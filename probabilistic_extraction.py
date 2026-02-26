import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MonteCarloResult:
    estimate: float
    standard_error: float
    wald_ci: Tuple[float, float]
    wilson_ci: Tuple[float, float]
    hits: int
    num_samples: int


def _validate_common_args(remasking: str, estimation_method: str) -> None:
    if remasking != 'low-confidence':
        raise NotImplementedError('Only low-confidence remasking is currently supported.')
    if estimation_method not in {'exact', 'monte-carlo'}:
        raise ValueError("estimation_method must be one of {'exact', 'monte-carlo'}")


def _suffix_attention_mask(prompt_attention_mask: Optional[torch.Tensor], suffix_len: int, device: torch.device) -> Optional[torch.Tensor]:
    if prompt_attention_mask is None:
        return None
    return torch.cat(
        [prompt_attention_mask.to(device), torch.ones((1, suffix_len), dtype=prompt_attention_mask.dtype, device=device)],
        dim=-1,
    )


def _uniform_cutoff_subsets(mask_positions: Sequence[int], confidences: Sequence[float], k: int) -> List[Tuple[Tuple[int, ...], float]]:
    if k == 0:
        return [(tuple(), 1.0)]

    value_pos = list(zip(confidences, mask_positions))
    sorted_values = sorted((v for v, _ in value_pos), reverse=True)
    kth = sorted_values[k - 1]

    higher = [p for v, p in value_pos if v > kth]
    equal = [p for v, p in value_pos if v == kth]

    must_take = len(higher)
    choose_needed = k - must_take

    if choose_needed < 0:
        raise RuntimeError('Invalid cutoff computation: choose_needed < 0.')
    if choose_needed == 0:
        return [(tuple(sorted(higher)), 1.0)]
    if choose_needed == len(equal):
        return [(tuple(sorted(higher + equal)), 1.0)]

    denom = math.comb(len(equal), choose_needed)
    out = []
    for subset in combinations(equal, choose_needed):
        selected = tuple(sorted(higher + list(subset)))
        out.append((selected, 1.0 / denom))
    return out


def _max_token_set(logits_1d: torch.Tensor) -> List[int]:
    max_value = logits_1d.max()
    idx = torch.nonzero(logits_1d == max_value, as_tuple=False).squeeze(-1)
    return idx.tolist()


def _safe_wald_and_wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    if n <= 0:
        return 0.0, float('nan'), (float('nan'), float('nan')), (float('nan'), float('nan'))
    p = hits / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    wald = (max(0.0, p - z * se), min(1.0, p + z * se))

    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    wilson = (max(0.0, center - half), min(1.0, center + half))
    return p, se, wald, wilson


@torch.no_grad()
def _exact_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
) -> float:
    device = model.device
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)
    suffix_len = target_tokens.shape[1]

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    # State is mask bitset over target suffix positions.
    # True => still masked.
    init_mask = tuple([True] * suffix_len)
    state_prob: Dict[Tuple[bool, ...], float] = {init_mask: 1.0}

    base = suffix_len // steps
    rem = suffix_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    for step_idx in range(steps):
        k_transfer = schedule[step_idx]
        next_state_prob: Dict[Tuple[bool, ...], float] = {}

        for mask_state, prob_mass in state_prob.items():
            if prob_mass == 0.0:
                continue

            x = torch.full((1, prompt_tokens.shape[1] + suffix_len), mask_id, dtype=torch.long, device=device)
            x[:, :prompt_tokens.shape[1]] = prompt_tokens
            for pos, is_masked in enumerate(mask_state):
                if not is_masked:
                    x[0, prompt_tokens.shape[1] + pos] = target_tokens[0, pos]

            logits = model(x, attention_mask=attn).logits[0]

            masked_positions = [i for i, m in enumerate(mask_state) if m]
            if len(masked_positions) != sum(mask_state):
                raise RuntimeError('Mask state invariant violated.')

            # Confidence under low-confidence mode with temperature=0:
            # confidence per masked position is max softmax probability.
            conf = []
            argmax_sets = {}
            for p in masked_positions:
                l = logits[prompt_tokens.shape[1] + p]
                probs = F.softmax(l, dim=-1)
                conf.append(float(probs.max().item()))
                argmax_sets[p] = _max_token_set(l)

            subsets = _uniform_cutoff_subsets(masked_positions, conf, k_transfer)

            for selected_subset, subset_prob in subsets:
                survive_prob = 1.0
                new_mask = list(mask_state)

                for p in selected_subset:
                    target_token = int(target_tokens[0, p].item())
                    tie_set = argmax_sets[p]
                    k = len(tie_set)
                    if target_token not in tie_set:
                        survive_prob = 0.0
                        break
                    survive_prob *= 1.0 / k
                    new_mask[p] = False

                if survive_prob == 0.0:
                    continue

                new_state = tuple(new_mask)
                next_state_prob[new_state] = next_state_prob.get(new_state, 0.0) + prob_mass * subset_prob * survive_prob

        state_prob = next_state_prob

    final_state = tuple([False] * suffix_len)
    return float(state_prob.get(final_state, 0.0))


@torch.no_grad()
def _monte_carlo_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
) -> MonteCarloResult:
    device = model.device
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)
    suffix_len = target_tokens.shape[1]
    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    rng = torch.Generator(device='cpu')
    if seed is not None:
        rng.manual_seed(seed)

    base = suffix_len // steps
    rem = suffix_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    hits = 0

    for _ in range(num_samples):
        suffix = torch.full((suffix_len,), mask_id, dtype=torch.long, device=device)
        alive = True

        for step_idx in range(steps):
            if not alive:
                break

            x = torch.cat([prompt_tokens[0], suffix], dim=0).unsqueeze(0)
            logits = model(x, attention_mask=attn).logits[0]

            masked_positions = (suffix == mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
            k_transfer = schedule[step_idx]

            conf = []
            argmax_sets = {}
            for p in masked_positions:
                l = logits[prompt_tokens.shape[1] + p]
                probs = F.softmax(l, dim=-1)
                conf.append(float(probs.max().item()))
                argmax_sets[p] = _max_token_set(l)

            subsets = _uniform_cutoff_subsets(masked_positions, conf, k_transfer)
            subset_weights = torch.tensor([w for _, w in subsets], dtype=torch.float64)
            subset_idx = int(torch.multinomial(subset_weights, 1, generator=rng).item())
            selected_subset = subsets[subset_idx][0]

            for p in selected_subset:
                tie_set = argmax_sets[p]
                if len(tie_set) == 1:
                    chosen = tie_set[0]
                else:
                    choice_idx = int(torch.randint(0, len(tie_set), (1,), generator=rng).item())
                    chosen = tie_set[choice_idx]

                suffix[p] = chosen

                if chosen != int(target_tokens[0, p].item()):
                    alive = False
                    break

        if alive and torch.equal(suffix, target_tokens[0]):
            hits += 1

    estimate, se, wald, wilson = _safe_wald_and_wilson(hits, num_samples)
    return MonteCarloResult(
        estimate=estimate,
        standard_error=se,
        wald_ci=wald,
        wilson_ci=wilson,
        hits=hits,
        num_samples=num_samples,
    )


@torch.no_grad()
def compute_probabilitic_extraction(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor] = None,
    mask_id: int = 126336,
    remasking: str = 'low-confidence',
    estimation_method: str = 'exact',
    num_samples: int = 1000,
    seed: Optional[int] = None,
):
    """
    Compute probabilistic extraction under LLaDA Algorithm-5 style low-confidence remasking.

    Parameters
    ----------
    model:
        Mask predictor model. Must return `.logits` from `model(input_ids, attention_mask=...)`.
    prompt_tokens:
        Tensor of shape (1, a).
    target_tokens:
        Tensor of shape (1, j) for suffix y.
    steps:
        Number of sampling steps N.
    remasking:
        String selector. Currently only 'low-confidence'.
    estimation_method:
        'exact' (branching over tie-breaks) or 'monte-carlo'.
    num_samples:
        Number of Monte Carlo samples when estimation_method='monte-carlo'.
    seed:
        RNG seed for Monte Carlo.
    """
    _validate_common_args(remasking=remasking, estimation_method=estimation_method)

    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
        raise ValueError('prompt_tokens must have shape (1, a).')
    if target_tokens.ndim != 2 or target_tokens.shape[0] != 1:
        raise ValueError('target_tokens must have shape (1, j).')
    if steps <= 0:
        raise ValueError('steps must be > 0.')
    if target_tokens.shape[1] < steps:
        raise ValueError('steps must be <= target suffix length for this scheduler.')

    if estimation_method == 'exact':
        return {
            'method': 'exact',
            'probability': _exact_probability(
                model=model,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
            ),
        }

    mc = _monte_carlo_probability(
        model=model,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        steps=steps,
        attention_mask=attention_mask,
        mask_id=mask_id,
        num_samples=num_samples,
        seed=seed,
    )
    return {
        'method': 'monte-carlo',
        'estimate': mc.estimate,
        'standard_error': mc.standard_error,
        'wald_ci': mc.wald_ci,
        'wilson_ci': mc.wilson_ci,
        'hits': mc.hits,
        'num_samples': mc.num_samples,
    }
