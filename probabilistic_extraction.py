import math
import secrets
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from get_log_likelihood import get_log_likelihood, get_log_likelihood_from_partially_masked

AUTOREGRESSIVE_MODEL_FAMILIES = {'llama', 'llama2', 'olmo', 'mistral'}
MAX_EXACT_LOW_CONFIDENCE_MASKED = 12


@dataclass
class MonteCarloResult:
    estimate: float
    standard_error: float
    wald_ci: Tuple[float, float]
    wilson_ci: Tuple[float, float]
    hits: int
    num_samples: int
    verbose_samples: Optional[List[Dict[str, object]]] = None


def validate_masked_indexes(
    masked_indexes: Optional[Sequence[int]],
    expected_count: Optional[int] = 50,
) -> Optional[List[int]]:
    if masked_indexes is None:
        return None

    normalized = [int(index) for index in masked_indexes]
    if expected_count is None:
        if not normalized:
            raise ValueError('--masked_indexes must contain at least one integer.')
    elif len(normalized) != expected_count:
        raise ValueError(f'--masked_indexes must contain exactly {expected_count} integers.')
    if any(index < 1 or index > 100 for index in normalized):
        raise ValueError('--masked_indexes entries must be 1-indexed positions in [1, 100].')
    if len(set(normalized)) != len(normalized):
        raise ValueError('--masked_indexes must not contain duplicates.')
    return normalized


def _unsupported_partially_masked_configuration(
    remasking: str,
    estimation_method: str,
    decoding_scheme: str,
) -> None:
    raise ValueError(
        '--masked_indexes is only supported for LLaDA configurations that use '
        '_elbo_probability, _path_sampling_random_probability, '
        '_path_sampling_low_confidence_probability_fast, '
        '_monte_carlo_probability_temperature_fast, or the exact '
        'low-confidence subset DP. '
        f'Got remasking={remasking!r}, estimation_method={estimation_method!r}, '
        f'decoding_scheme={decoding_scheme!r}.'
    )


def _validate_common_args(remasking: str, estimation_method: str) -> None:
    allowed_remasking = {'low-confidence', 'target-token-confidence', 'random', 'highest-index'}
    if remasking not in allowed_remasking:
        raise NotImplementedError(
            f"Unsupported remasking strategy: {remasking!r}. Supported strategies: {sorted(allowed_remasking)}"
        )
    if estimation_method not in {'exact', 'monte-carlo', 'path_sampling'}:
        raise ValueError("estimation_method must be one of {'exact', 'monte-carlo', 'path_sampling'}")


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


def _model_device(model) -> torch.device:
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def _elbo_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask_id: int,
) -> Dict[str, float]:
    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
        raise ValueError('prompt_tokens must have shape (1, a).')
    if target_tokens.ndim != 2 or target_tokens.shape[0] != 1:
        raise ValueError('target_tokens must have shape (1, j).')

    prompt_1d = prompt_tokens[0]
    target_1d = target_tokens[0]
    log_probability = get_log_likelihood(
        model=model,
        prompt=prompt_1d,
        answer=target_1d,
        mask_id=mask_id,
    )
    return {
        'probability': math.exp(log_probability),
        'log_probability': log_probability,
    }

@torch.no_grad()
def _elbo_probability_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,     # [1, 100]
    masked_indexes: list[int],         # 1-indexed masked positions
    mask_id: int,
) -> Dict[str, float]:
    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError('sequence_tokens must have shape (1, 100).')

    seq_len = sequence_tokens.shape[1]
    if seq_len != 100:
        raise ValueError(f'sequence_tokens must have shape (1, 100); got (1, {seq_len}).')

    masked_pos = sorted(set(int(i) for i in masked_indexes))
    if len(masked_pos) != 50:
        raise ValueError(
            f'Expected exactly 50 masked positions out of 100, got {len(masked_pos)}.'
        )
    if any(pos < 1 or pos > 100 for pos in masked_pos):
        raise ValueError('masked_indexes must be 1-indexed positions in [1, 100].')

    sequence_1d = sequence_tokens[0]
    log_probability = get_log_likelihood_from_partially_masked(
        model=model,
        prompt=sequence_1d,
        masked_indexes=masked_pos,
        mask_id=mask_id,
    )
    return {
        'probability': math.exp(log_probability),
        'log_probability': log_probability,
    }

@torch.no_grad()
def _exact_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
) -> float:
    device = _model_device(model)
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
def _exact_probability_target_token_confidence(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    temperature: float,
    decoding_scheme: str,
    k: int,
) -> Dict[str, float]:
    if temperature <= 0:
        raise ValueError('temperature must be > 0 for remasking="target-token-confidence".')

    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    suffix_len = target_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]
    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    suffix = torch.full((suffix_len,), mask_id, dtype=torch.long, device=device)
    logp = 0.0

    for _ in range(steps):
        x = torch.cat([prompt_tokens[0], suffix], dim=0).unsqueeze(0)
        logits = model(x, attention_mask=attn).logits[0]

        masked_positions = (suffix == mask_id).nonzero(as_tuple=False).squeeze(-1)
        candidate_probs: List[float] = []
        for pos in masked_positions.tolist():
            target_id = int(target_tokens[0, pos].item())
            scaled_logits = logits[prompt_len + pos] / temperature

            if decoding_scheme == 'top_k':
                top_k = min(k, scaled_logits.shape[-1])
                topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
                in_topk = bool((topk_idx == target_id).any().item())
                if in_topk:
                    selected_logit = scaled_logits[target_id]
                    log_denom = torch.logsumexp(topk_vals, dim=-1)
                    prob = float(torch.exp(selected_logit - log_denom).item())
                else:
                    prob = 0.0
            else:
                prob = float(F.softmax(scaled_logits, dim=-1)[target_id].item())

            candidate_probs.append(prob)
        candidate_probs_t = torch.tensor(candidate_probs, device=device, dtype=logits.dtype)

        best_idx = int(torch.argmax(candidate_probs_t).item())
        chosen_suffix_pos = int(masked_positions[best_idx].item())
        selected_prob = float(candidate_probs_t[best_idx].item())

        if selected_prob == 0.0:
            return {
                'probability': 0.0,
                'log_probability': float('-inf'),
            }
        logp += math.log(selected_prob)
        suffix[chosen_suffix_pos] = target_tokens[0, chosen_suffix_pos]

    return {
        'probability': float(math.exp(logp)),
        'log_probability': float(logp),
    }


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
    device = _model_device(model)
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

@torch.inference_mode()
def _path_sampling_probability_temperature1_tie_free(
    model,
    prompt_tokens: torch.Tensor,      # [1, P]
    target_tokens: torch.Tensor,      # [1, S]
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    temperature: float,
    decoding_scheme: str,
    k: int,
    mc_batch_size: int = 512,
) -> Dict[str, Any]:
    """
    Path-sampling / sequential-importance estimator for

        P_{theta,phi}(target_suffix | prompt_tokens)

    under:
      - one token revealed per step
      - low-confidence remasking
      - temperature > 0
      - proposal sampling either full-temperature or top-k-temperature

    Returns:
    {
        "probability": average_probability,
        "sample_probabilities": sample_probabilities,
        "num_samples": num_samples,
        "estimation_method": "path_sampling",
    }

    Notes
    -----
    At each step and for each masked position i:
      - p_i(v) = softmax(logits_i)[v]              # base confidence distribution
      - q_i(v) = proposal distribution used by decoding
                = softmax(logits_i / temperature)[v]
                or top-k-truncated version if decoding_scheme == "top_k"

    For a still-masked target position j with true token z_j:
      t_j     = p_j(z_j)
      alpha_j = q_j(z_j) * prod_{i != j} F_i(t_j)
      where F_i(t) = sum_v q_i(v) 1[p_i(v) < t]

    Then:
      s = sum_j alpha_j
      weight *= s
      sample next revealed position J ~ alpha / s
      reveal the correct token at J

    The final particle weight is an unbiased estimator of the target sequence probability.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device, non_blocking=True)
    target_tokens = target_tokens.to(device, non_blocking=True)

    assert prompt_tokens.ndim == 2 and prompt_tokens.shape[0] == 1
    assert target_tokens.ndim == 2 and target_tokens.shape[0] == 1

    prefix_len = prompt_tokens.shape[1]
    suffix_len = target_tokens.shape[1]

    # Specialization requested: one token revealed per step.
    if steps != suffix_len:
        raise ValueError(
            f"This optimized version assumes one token revealed per step, "
            f"so steps must equal suffix_len. Got steps={steps}, suffix_len={suffix_len}."
        )

    if decoding_scheme not in {"full", "top_k"}:
        raise ValueError(
            f"Unsupported decoding_scheme={decoding_scheme!r}. "
            f"Expected 'full' or 'top_k'."
        )

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    target_suffix = target_tokens[0]  # [S]
    sample_probabilities: List[float] = []

    # Small numerical floor for products / renormalization.
    tiny = torch.finfo(torch.float32).tiny

    for start in range(0, num_samples, mc_batch_size):
        bsz = min(mc_batch_size, num_samples - start)

        # Current partially revealed suffix for each particle.
        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # Running importance weights.
        weights = torch.ones(bsz, dtype=torch.float64, device=device)

        prompt_batch = prompt_tokens.expand(bsz, -1)
        x = torch.empty(
            (bsz, prefix_len + suffix_len),
            dtype=prompt_tokens.dtype,
            device=device,
        )
        x[:, :prefix_len] = prompt_batch

        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        for _step_idx in range(steps):
            if not alive.any():
                break

            x[:, prefix_len:] = suffix
            logits = model(x, attention_mask=attn).logits[:, prefix_len:, :]  # [B,S,V]

            # Base confidence distribution p_i(v) = softmax(logits_i)[v]
            base_probs = logits.softmax(dim=-1)  # [B,S,V]

            # Proposal distribution q_i(v)
            scaled_logits = logits / temperature
            if decoding_scheme == "top_k":
                vocab_size = scaled_logits.shape[-1]
                top_k = min(k, vocab_size)
                if top_k <= 0:
                    raise ValueError(f"k must be positive for top_k decoding. Got k={k}.")

                top_vals, top_idx = torch.topk(scaled_logits, k=top_k, dim=-1)  # [B,S,K], [B,S,K]
                top_q = top_vals.softmax(dim=-1)  # truncated proposal probs on top-k support

                proposal_probs = torch.zeros_like(base_probs)
                proposal_probs.scatter_(-1, top_idx, top_q)
            else:
                proposal_probs = scaled_logits.softmax(dim=-1)  # [B,S,V]

            masked = (suffix == mask_id) & alive[:, None]  # [B,S]

            # Gather target-token base confidence t_j = p_j(z_j)
            target_idx = target_suffix.view(1, suffix_len, 1).expand(bsz, -1, -1)  # [B,S,1]
            target_base_conf = base_probs.gather(dim=-1, index=target_idx).squeeze(-1)  # [B,S]
            target_proposal_prob = proposal_probs.gather(dim=-1, index=target_idx).squeeze(-1)  # [B,S]

            # alpha[b, j] = q_j(z_j) * prod_{i != j} F_i(t_j)
            alpha = torch.zeros(bsz, suffix_len, dtype=torch.float64, device=device)

            active_rows = torch.nonzero(alive, as_tuple=False).squeeze(-1)
            for b in active_rows.tolist():
                masked_pos = torch.nonzero(masked[b], as_tuple=False).squeeze(-1)
                m = masked_pos.numel()
                if m == 0:
                    continue

                # Extract only currently masked positions for this particle.
                # pb: [m, V], qb: [m, V]
                pb = base_probs[b, masked_pos, :]
                qb = proposal_probs[b, masked_pos, :]

                # Thresholds t_j for each candidate correct reveal j.
                # tj[q] = p_{masked_pos[q]}(z_{masked_pos[q]})
                tj = target_base_conf[b, masked_pos]         # [m]
                qj = target_proposal_prob[b, masked_pos]     # [m]

                # Build F_i(t) efficiently using sorting by p_i(v) and cumulative q_i(v).
                # For each masked position i:
                #   F_i(t) = sum_v q_i(v) * 1[p_i(v) < t]
                #
                # We compute all F_i(t_j) for all j.
                F = torch.empty((m, m), dtype=torch.float64, device=device)  # F[i, j] = F_i(t_j)

                for local_i in range(m):
                    p_row = pb[local_i]  # [V]
                    q_row = qb[local_i]  # [V]

                    sorted_p, perm = torch.sort(p_row)                   # ascending p_i(v)
                    sorted_q = q_row[perm].to(torch.float64)
                    cdf_q = torch.cumsum(sorted_q, dim=0)                # prefix sums in q-space

                    # count of tokens with p_i(v) < t_j
                    idx = torch.searchsorted(sorted_p, tj, right=False)  # [m], in [0, V]
                    Fi = torch.zeros(m, dtype=torch.float64, device=device)
                    valid = idx > 0
                    Fi[valid] = cdf_q[idx[valid] - 1]
                    F[local_i] = Fi

                # For candidate j, alpha_j = q_j(z_j) * prod_{i != j} F_i(t_j)
                # Use logs for stability.
                logF = torch.log(torch.clamp(F, min=tiny))               # [m, m]
                sum_logF = logF.sum(dim=0)                               # [m] = sum_i log F_i(t_j)

                # Remove i == j term from the product.
                diag_logF = torch.diagonal(logF, dim1=0, dim2=1)         # [m]
                log_alpha_local = (
                    torch.log(torch.clamp(qj.to(torch.float64), min=tiny))
                    + (sum_logF - diag_logF)
                )
                alpha_local = torch.exp(log_alpha_local)                 # [m]

                # Only masked positions are candidates.
                alpha[b, masked_pos] = alpha_local

            # s_b = sum_j alpha_bj
            s = alpha.sum(dim=-1)  # [B]

            # Dead particles contribute zero thereafter.
            zero_survival = alive & (s <= 0)
            if zero_survival.any():
                weights[zero_survival] = 0.0
                alive = alive & (~zero_survival)

            if not alive.any():
                break

            # Update importance weights: w *= s
            weights[alive] *= s[alive]

            # Sample next correctly revealed position J ~ alpha / s
            alpha_alive = alpha[alive]                                   # [B_alive, S]
            s_alive = s[alive].unsqueeze(-1)                             # [B_alive, 1]
            proposal_next_pos = alpha_alive / torch.clamp(s_alive, min=tiny)

            selected_pos_alive = torch.multinomial(
                proposal_next_pos.to(torch.float32),
                num_samples=1,
                generator=rng,
            ).squeeze(-1)                                                # [B_alive]

            # Reveal the correct target token at the sampled position.
            alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
            suffix[alive_idx, selected_pos_alive] = target_suffix[selected_pos_alive]

        sample_probabilities.extend(weights.detach().cpu().tolist())

    average_probability = float(sum(sample_probabilities) / max(1, num_samples))

    return {
        "probability": average_probability,
        "sample_probabilities": sample_probabilities,
        "num_samples": num_samples,
        "estimation_method": "path_sampling",
    }



@torch.inference_mode()
def _monte_carlo_probability_temperature_fast(
    model,
    prompt_tokens: torch.Tensor,      # [1, P]
    target_tokens: torch.Tensor,      # [1, S]
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    temperature: float,
    decoding_scheme: str,
    k: int,
    mc_batch_size: int = 512,
) -> MonteCarloResult:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device, non_blocking=True)
    target_tokens = target_tokens.to(device, non_blocking=True)

    assert prompt_tokens.ndim == 2 and prompt_tokens.shape[0] == 1
    assert target_tokens.ndim == 2 and target_tokens.shape[0] == 1

    prefix_len = prompt_tokens.shape[1]
    suffix_len = target_tokens.shape[1]

    # Your requested specialization: one token revealed per step.
    if steps != suffix_len:
        raise ValueError(
            f"This optimized version assumes one token revealed per step, "
            f"so steps must equal suffix_len. Got steps={steps}, suffix_len={suffix_len}."
        )

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    target_suffix = target_tokens[0]                            # [S]
    hits = 0
    vocab_arange_cache = None

    for start in range(0, num_samples, mc_batch_size):
        bsz = min(mc_batch_size, num_samples - start)

        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        prompt_batch = prompt_tokens.expand(bsz, -1)

        x = torch.empty(
            (bsz, prefix_len + suffix_len),
            dtype=prompt_tokens.dtype,
            device=device,
        )
        x[:, :prefix_len] = prompt_batch

        for _step_idx in range(steps):
            if not alive.any():
                break

            x[:, prefix_len:] = suffix

            logits = model(x, attention_mask=attn).logits[:, prefix_len:, :]  
            probs = logits.softmax(dim=-1)                                      
            masked = (suffix == mask_id) & alive[:, None]                       
            scaled_logits = logits / temperature                                

            if decoding_scheme == "top_k":
                top_k = min(k, scaled_logits.shape[-1])
                top_vals, top_idx = torch.topk(scaled_logits, k=top_k, dim=-1)  
                top_probs = top_vals.softmax(dim=-1)                            

                sampled_local = torch.multinomial(
                    top_probs.reshape(-1, top_k),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, suffix_len)                                       # [B,S]

                sampled_tokens = top_idx.gather(
                    dim=-1,
                    index=sampled_local.unsqueeze(-1),
                ).squeeze(-1)                                                    
            else:
                vocab_size = scaled_logits.shape[-1]
                sampled_tokens = torch.multinomial(
                    scaled_logits.softmax(dim=-1).reshape(-1, vocab_size),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, suffix_len)                                       
            chosen_prob = probs.gather(
                dim=-1,
                index=sampled_tokens.unsqueeze(-1),
            ).squeeze(-1)                                                        
            neg_inf = torch.full_like(chosen_prob, float("-inf"))
            confidence = torch.where(masked, chosen_prob, neg_inf)               
            selected_pos = confidence.argmax(dim=-1)                             

            batch_idx = torch.arange(bsz, device=device)
            selected_token = sampled_tokens[batch_idx, selected_pos]             
            selected_target = target_suffix[selected_pos]                        
            active = alive

            suffix[batch_idx[active], selected_pos[active]] = selected_token[active]

            mismatch = active & (selected_token != selected_target)
            alive = alive & (~mismatch)

        if alive.any():
            hits += (alive & (suffix == target_suffix.unsqueeze(0)).all(dim=-1)).sum().item()

    estimate, se, wald, wilson = _safe_wald_and_wilson(hits, num_samples)
    return MonteCarloResult(
        estimate=estimate,
        standard_error=se,
        wald_ci=wald,
        wilson_ci=wilson,
        hits=hits,
        num_samples=num_samples,
    )


@torch.inference_mode()
def highest_index_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    decoding_scheme: str,
    k: int,
    temperature: float,
) -> Dict[str, object]:
    """
    Exact probability for the deterministic reveal path:
        suffix position 0, then 1, then 2, ..., then suffix_len - 1.

    If steps == suffix_len, this reveals one token per step.
    If steps < suffix_len, this reveals contiguous chunks from left to right.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    suffix_len = target_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    base = suffix_len // steps
    rem = suffix_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    prompt_row = prompt_tokens[0]  # [prompt_len]
    target_row = target_tokens[0]  # [suffix_len]

    suffix = torch.full(
        (1, suffix_len),
        mask_id,
        dtype=torch.long,
        device=device,
    )

    log_probability = torch.zeros((), dtype=torch.float64, device=device)
    alive = True

    start = 0

    for step_size in schedule:
        if step_size == 0:
            continue

        # Deterministic left-to-right reveal positions.
        reveal_positions = torch.arange(
            start,
            start + step_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)  # [1, step_size]

        start += step_size

        x = torch.cat(
            [prompt_row.unsqueeze(0), suffix],
            dim=1,
        )  # [1, prompt_len + suffix_len]

        batched_attn = None
        if attn is not None:
            if attn.shape[0] == 1:
                batched_attn = attn
            else:
                batched_attn = attn[:1]

        logits = model(x, attention_mask=batched_attn).logits
        suffix_logits = logits[:, prompt_len:, :]  # [1, suffix_len, vocab]

        vocab_size = suffix_logits.shape[-1]
        gather_index = reveal_positions.unsqueeze(-1).expand(-1, -1, vocab_size)

        step_logits = torch.gather(
            suffix_logits,
            dim=1,
            index=gather_index,
        )  # [1, step_size, vocab]

        target_ids = torch.gather(
            target_row.unsqueeze(0),
            dim=1,
            index=reveal_positions,
        )  # [1, step_size]

        scaled_logits = step_logits if temperature <= 0 else step_logits / temperature

        if decoding_scheme == "top_k":
            top_k = min(k, scaled_logits.shape[-1])

            if top_k <= 0:
                alive = False
                break

            topk_vals, topk_idx = torch.topk(
                scaled_logits,
                k=top_k,
                dim=-1,
            )  # [1, step_size, top_k]

            in_topk = (topk_idx == target_ids.unsqueeze(-1)).any(dim=-1)

            target_logits = torch.gather(
                scaled_logits,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)  # [1, step_size]

            token_log_probs = target_logits - torch.logsumexp(topk_vals, dim=-1)

            token_log_probs = torch.where(
                in_topk,
                token_log_probs,
                torch.full_like(token_log_probs, float("-inf")),
            )

        else:
            target_logits = torch.gather(
                scaled_logits,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)  # [1, step_size]

            token_log_probs = target_logits - torch.logsumexp(
                scaled_logits,
                dim=-1,
            )

        if torch.isneginf(token_log_probs).any():
            alive = False
            break

        log_probability = log_probability + token_log_probs.double().sum()

        # Reveal the true target tokens into the suffix.
        suffix.scatter_(
            dim=1,
            index=reveal_positions,
            src=target_ids,
        )

    if alive:
        probability = float(torch.exp(log_probability).detach().cpu())
        log_probability_value = float(log_probability.detach().cpu())
    else:
        probability = 0.0
        log_probability_value = float("-inf")

    return {
        "probability": probability,
        "log_probability": log_probability_value,
        "sample_probabilities": [probability],
        "num_samples": 1,
        "estimation_method": "highest_index_exact",
    }

@torch.inference_mode()
def highest_index_probability_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, 100]
    masked_indexes: list[int],              # 1-indexed masked positions in sequence_tokens
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    decoding_scheme: str,
    k: int,
    temperature: float,
) -> Dict[str, object]:
    """
    Exact probability for partially masked conditioning using the deterministic
    lowest-index-to-highest-index reveal path.

    High-level behavior:
      - sequence_tokens is the full target sequence z, shape [1, 100]
      - masked_indexes specifies the positions to regenerate, using 1-indexing
      - unmasked positions remain observed / conditioning tokens
      - masked positions are revealed from lowest absolute index to highest
      - if steps == number of masked positions, this reveals one token per step
      - supports 'top_k' and full-softmax decoding

    Example:
      masked_indexes = [51, 52, ..., 100]
      steps = 50

      Reveal order is:
        51, 52, 53, ..., 100

      using 1-indexed sequence positions.
    """

    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device)

    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError(
            f"sequence_tokens must have shape [1, 100], got {tuple(sequence_tokens.shape)}"
        )

    seq_len = sequence_tokens.shape[1]
    if seq_len != 100:
        raise ValueError(f"Expected sequence length 100, got {seq_len}")

    if steps <= 0:
        raise ValueError("steps must be positive")

    # Convert 1-indexed masked positions to 0-indexed absolute positions.
    # Sorting gives the deterministic lowest-index-to-highest-index reveal path.
    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))

    if len(masked_pos) != 50:
        raise ValueError(
            f"Expected exactly 50 masked positions out of 100, got {len(masked_pos)}"
        )

    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError("masked_indexes must be 1-indexed positions in [1, 100]")

    masked_len = len(masked_pos)
    masked_pos_t = torch.tensor(masked_pos, dtype=torch.long, device=device)  # [50]

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], got {tuple(attention_mask.shape)}"
            )

    base = masked_len // steps
    rem = masked_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    full_target_row = sequence_tokens[0]               # [100]
    masked_target_row = full_target_row[masked_pos_t]  # [50]

    # Current sequence state.
    # Observed positions stay fixed. Masked positions start as mask_id.
    x = sequence_tokens.clone()                        # [1, 100]
    x[:, masked_pos_t] = mask_id

    log_probability = torch.zeros((), dtype=torch.float64, device=device)
    alive = True

    start = 0

    for step_size in schedule:
        if step_size == 0:
            continue

        # Deterministic reveal slots into masked_pos_t / masked_target_row.
        # Since masked_pos_t is sorted, this reveals lowest absolute index first.
        reveal_slots = torch.arange(
            start,
            start + step_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)  # [1, step_size]

        start += step_size

        logits = model(x, attention_mask=attention_mask).logits  # [1, 100, vocab]
        vocab_size = logits.shape[-1]

        # Map masked slots to absolute sequence positions.
        reveal_abs_positions = masked_pos_t[reveal_slots]        # [1, step_size]

        gather_index = reveal_abs_positions.unsqueeze(-1).expand(
            -1,
            -1,
            vocab_size,
        )

        step_logits = torch.gather(
            logits,
            dim=1,
            index=gather_index,
        )  # [1, step_size, vocab]

        target_ids = torch.gather(
            masked_target_row.unsqueeze(0),
            dim=1,
            index=reveal_slots,
        )  # [1, step_size]

        scaled_logits = step_logits if temperature <= 0 else step_logits / temperature

        if decoding_scheme == "top_k":
            top_k = min(k, scaled_logits.shape[-1])

            if top_k <= 0:
                alive = False
                break

            topk_vals, topk_idx = torch.topk(
                scaled_logits,
                k=top_k,
                dim=-1,
            )  # [1, step_size, top_k]

            in_topk = (topk_idx == target_ids.unsqueeze(-1)).any(dim=-1)

            target_logits = torch.gather(
                scaled_logits,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)  # [1, step_size]

            token_log_probs = target_logits - torch.logsumexp(topk_vals, dim=-1)

            token_log_probs = torch.where(
                in_topk,
                token_log_probs,
                torch.full_like(token_log_probs, float("-inf")),
            )

        else:
            target_logits = torch.gather(
                scaled_logits,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)  # [1, step_size]

            token_log_probs = target_logits - torch.logsumexp(
                scaled_logits,
                dim=-1,
            )

        # If any token has zero probability, this deterministic path has probability 0.
        if torch.isneginf(token_log_probs).any():
            alive = False
            break

        log_probability = log_probability + token_log_probs.double().sum()

        # Reveal the true target tokens into x.
        x.scatter_(
            dim=1,
            index=reveal_abs_positions,
            src=target_ids,
        )

    if alive:
        probability = float(torch.exp(log_probability).detach().cpu())
        log_probability_value = float(log_probability.detach().cpu())
    else:
        probability = 0.0
        log_probability_value = float("-inf")

    return {
        "probability": probability,
        "log_probability": log_probability_value,
        "sample_probabilities": [probability],
        "num_samples": 1,
        "estimation_method": "highest_index_exact_from_partially_masked",
    }
@torch.inference_mode()
def _path_sampling_random_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    decoding_scheme: str,
    k: int,
    temperature: float,
    batch_size: int = 64,
) -> Dict[str, object]:
    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    suffix_len = target_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    # Keep CPU RNG for seeded reproducibility style close to the original.
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    base = suffix_len // steps
    rem = suffix_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    prompt_row = prompt_tokens[0]   # [prompt_len]
    target_row = target_tokens[0]   # [suffix_len]

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []

    for batch_start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - batch_start)

        # Current suffix states for all samples in this batch.
        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # Accumulated log-probability for each sample.
        log_path_probability = torch.zeros(bsz, dtype=torch.float64, device=device)

        # Whether the sample is still alive (not zero-probability yet).
        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        # Random reveal permutations, one per sample.
        # Generated on CPU using the seeded CPU RNG, then moved to device.
        perm_scores = torch.rand((bsz, suffix_len), generator=rng, device="cpu")
        permutation = perm_scores.argsort(dim=-1).to(device)  # [bsz, suffix_len]

        start = 0
        for step_size in schedule:
            reveal_positions = permutation[:, start:start + step_size]  # [bsz, step_size]
            start += step_size

            # Build model input.
            x = torch.cat(
                [prompt_row.unsqueeze(0).expand(bsz, -1), suffix],
                dim=1,
            )  # [bsz, prompt_len + suffix_len]

            # Repeat attention mask across batch if needed.
            batched_attn = None
            if attn is not None:
                if attn.shape[0] == bsz:
                    batched_attn = attn
                else:
                    batched_attn = attn.expand(bsz, *attn.shape[1:])

            logits = model(x, attention_mask=batched_attn).logits  # [bsz, total_len, vocab]
            suffix_logits = logits[:, prompt_len:, :]              # [bsz, suffix_len, vocab]

            # Gather logits for all revealed positions in one shot.
            vocab_size = suffix_logits.shape[-1]
            gather_index = reveal_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
            step_logits = torch.gather(suffix_logits, dim=1, index=gather_index)  # [bsz, step_size, vocab]

            target_ids = torch.gather(
                target_row.unsqueeze(0).expand(bsz, -1),
                dim=1,
                index=reveal_positions,
            )  # [bsz, step_size]

            scaled_logits = step_logits if temperature <= 0 else (step_logits / temperature)

            if decoding_scheme == "top_k":
                top_k = min(k, scaled_logits.shape[-1])
                topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)  # [bsz, step_size, top_k]

                in_topk = (topk_idx == target_ids.unsqueeze(-1)).any(dim=-1)  # [bsz, step_size]

                target_logits = torch.gather(
                    scaled_logits,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)  # [bsz, step_size]

                token_log_probs = target_logits - torch.logsumexp(topk_vals, dim=-1)
                token_log_probs = torch.where(
                    in_topk,
                    token_log_probs,
                    torch.full_like(token_log_probs, float("-inf")),
                )
            else:
                target_logits = torch.gather(
                    scaled_logits,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)  # [bsz, step_size]

                token_log_probs = target_logits - torch.logsumexp(scaled_logits, dim=-1)

            # If any revealed token has zero probability, the whole path becomes zero.
            step_has_zero = torch.isneginf(token_log_probs).any(dim=-1)  # [bsz]

            # Sum token log-probs for alive paths only.
            safe_token_log_probs = torch.where(
                torch.isfinite(token_log_probs),
                token_log_probs,
                torch.zeros_like(token_log_probs),
            )
            step_log_prob = safe_token_log_probs.sum(dim=-1)  # [bsz]

            log_path_probability = torch.where(
                alive & (~step_has_zero),
                log_path_probability + step_log_prob,
                log_path_probability,
            )

            alive = alive & (~step_has_zero)

            # Update suffix with the revealed target tokens for all samples.
            suffix.scatter_(dim=1, index=reveal_positions, src=target_ids)

        batch_log_probs = torch.where(
            alive,
            log_path_probability,
            torch.full_like(log_path_probability, float("-inf")),
        )

        sample_log_probabilities.extend(batch_log_probs.detach().cpu().tolist())

        batch_probabilities = torch.where(
            torch.isfinite(batch_log_probs),
            torch.exp(batch_log_probs),
            torch.zeros_like(batch_log_probs),
        )
        sample_probabilities.extend(batch_probabilities.detach().cpu().tolist())

    if sample_log_probabilities:
        finite_logs = [lp for lp in sample_log_probabilities if not math.isinf(lp)]
        if not finite_logs:
            average_probability = 0.0
        else:
            max_log = max(finite_logs)
            scaled_sum = sum(math.exp(lp - max_log) for lp in finite_logs)
            average_probability = float(
                math.exp(max_log) * (scaled_sum / len(sample_log_probabilities))
            )
    else:
        average_probability = 0.0

    return {
        "probability": average_probability,
        "sample_probabilities": sample_probabilities,
        "num_samples": num_samples,
        "estimation_method": "path_sampling",
    }

@torch.no_grad()
def _autoregressive_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    decoding_scheme: str,
    k: int,
    temperature: float,
    return_token_details: bool,
):
    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    if prompt_tokens.shape[1] == 0:
        raise ValueError('For autoregressive model families, prompt_tokens must contain at least one token.')
    if decoding_scheme not in {'top_k', 'full', 'greedy'}:
        raise ValueError("decoding_scheme must be one of {'top_k', 'full', 'greedy'} for autoregressive model families.")
    if decoding_scheme == 'top_k' and k <= 0:
        raise ValueError('k must be > 0 when decoding_scheme="top_k".')
    if decoding_scheme in {'top_k', 'full'} and temperature <= 0:
        raise ValueError('temperature must be > 0 when decoding_scheme is "top_k" or "full".')

    full_tokens = torch.cat([prompt_tokens, target_tokens], dim=1)
    full_attention_mask = _suffix_attention_mask(attention_mask, target_tokens.shape[1], device)
    logits = model(full_tokens, attention_mask=full_attention_mask).logits[0]

    prompt_len = prompt_tokens.shape[1]
    log_prob_total = 0.0
    total_prob_zero = False
    token_details: List[Dict[str, float]] = []

    for t in range(target_tokens.shape[1]):
        pred_logits = logits[prompt_len + t - 1]
        target_id = int(target_tokens[0, t].item())

        if decoding_scheme == 'greedy':
            greedy_id = int(torch.argmax(pred_logits).item())
            step_prob = 1.0 if greedy_id == target_id else 0.0
        elif decoding_scheme == 'top_k':
            scaled_logits = pred_logits / temperature
            top_k = min(k, scaled_logits.shape[-1])
            topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
            in_topk = bool((topk_idx == target_id).any().item())
            if in_topk:
                selected_logit = scaled_logits[target_id]
                log_denom = torch.logsumexp(topk_vals, dim=-1)
                step_prob = float(torch.exp(selected_logit - log_denom).item())
            else:
                step_prob = 0.0
        else:
            scaled_logits = pred_logits / temperature
            step_prob = float(F.softmax(scaled_logits, dim=-1)[target_id].item())

        if step_prob == 0.0:
            total_prob_zero = True
            log_prob_total = float('-inf')
        elif not total_prob_zero:
            log_prob_total += math.log(step_prob)

        if return_token_details:
            token_details.append(
                {
                    'position': t,
                    'token_id': target_id,
                    'step_probability': step_prob,
                }
            )

    result = {
        'method': 'autoregressive',
        'model_family': 'llama',
        'decoding_scheme': decoding_scheme,
        'probability': 0.0 if total_prob_zero else float(math.exp(log_prob_total)),
        'log_probability': float(log_prob_total),
    }
    if return_token_details:
        result['token_details'] = token_details
    return result

def _logmeanexp_and_stderr(log_values: List[float]) -> Tuple[float, float]:
    """
    Returns:
        mean estimate in probability space,
        Monte Carlo standard error in probability space.

    Uses max-log scaling for numerical stability.
    """
    if not log_values:
        return 0.0, float("nan")

    max_log = max(log_values)
    n = len(log_values)

    if math.isinf(max_log) and max_log < 0:
        return 0.0, 0.0

    scaled = [math.exp(x - max_log) for x in log_values]
    scaled_mean = sum(scaled) / n
    mean = math.exp(max_log) * scaled_mean

    if n <= 1:
        return mean, float("nan")

    scaled_var = sum((x - scaled_mean) ** 2 for x in scaled) / (n - 1)
    stderr = math.exp(max_log) * math.sqrt(scaled_var / n)

    return float(mean), float(stderr)


import math
from typing import Dict, List, Optional

import torch


import math
from typing import Dict, List, Optional

import torch

import math
from typing import Dict, List, Optional

import torch


_LOW_CONFIDENCE_LOG_TOL = 1e-10


def _low_confidence_eval_mode(function):
    """Use deterministic eval execution and restore the caller's settings."""
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        modes = [(module, module.training) for module in model.modules()]
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        try:
            model.eval()
            # Fail rather than silently permit a known nondeterministic kernel.
            # CUDA may require CUBLAS_WORKSPACE_CONFIG before process startup;
            # PyTorch reports that requirement if an affected operation is used.
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            return function(model, *args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
            for module, training in modes:
                module.training = training
    return wrapped


def _check_low_confidence_log_mass(values, name, context):
    """Allow zero mass (-inf), but never silently turn invalid math into failure."""
    invalid = torch.isnan(values) | torch.isposinf(values)
    invalid |= values > _LOW_CONFIDENCE_LOG_TOL
    if bool(invalid.any().item()):
        raise FloatingPointError(
            f"Invalid {name} ({context}): expected log mass <= 0 or -inf; "
            f"got {values[invalid].detach().cpu().tolist()}"
        )


@dataclass
class _LowConfidenceDistribution:
    logits: torch.Tensor
    log_Z_conf: torch.Tensor
    log_Z_sample: torch.Tensor
    sorted_logits: torch.Tensor
    sorted_token_ids: torch.Tensor
    log_cdf: torch.Tensor


def _low_confidence_distribution(logits_native, temperature, context):
    """Canonical FP64 arithmetic for ONE state's [remaining, vocabulary] logits."""
    if logits_native.ndim != 2 or logits_native.shape[-1] == 0:
        raise ValueError(f"Expected nonempty [remaining, vocabulary] logits ({context}).")
    logits = logits_native.to(torch.float64)
    if bool((torch.isnan(logits) | torch.isposinf(logits)).any().item()):
        raise FloatingPointError(f"Invalid model logits ({context}): NaN or +inf.")
    log_Z_conf = torch.logsumexp(logits, dim=-1)
    log_Z_sample = (log_Z_conf if temperature == 1.0 else
                    torch.logsumexp(logits / temperature, dim=-1))
    if not bool((torch.isfinite(log_Z_conf) & torch.isfinite(log_Z_sample)).all().item()):
        raise FloatingPointError(f"Nonfinite distribution normalizer ({context}).")

    if temperature != 1.0:
        confidence_logs = logits - log_Z_conf.unsqueeze(-1)
        _check_low_confidence_log_mass(confidence_logs, 'confidence probabilities', context)
        confidence_total = torch.logsumexp(confidence_logs, dim=-1)
        if not bool((torch.isfinite(confidence_total) &
                     (confidence_total.abs() <= _LOW_CONFIDENCE_LOG_TOL)).all().item()):
            raise FloatingPointError(f"Confidence distribution is not normalized ({context}).")
        del confidence_logs

    sample_logs = logits / temperature - log_Z_sample.unsqueeze(-1)
    _check_low_confidence_log_mass(sample_logs, 'token probabilities', context)
    log_total = torch.logsumexp(sample_logs, dim=-1)
    if not bool((torch.isfinite(log_total) & (log_total.abs() <= _LOW_CONFIDENCE_LOG_TOL)).all().item()):
        raise FloatingPointError(f"Sampling distribution is not normalized ({context}).")
    del sample_logs

    sorted_native, sorted_token_ids = torch.sort(logits_native, dim=-1)
    sorted_logits = sorted_native.to(torch.float64)
    log_cdf = torch.logcumsumexp(
        sorted_logits if temperature == 1.0 else sorted_logits / temperature,
        dim=-1,
    ) - log_Z_sample.unsqueeze(-1)
    _check_low_confidence_log_mass(log_cdf, 'sampling CDF', context)
    if not bool((log_cdf[:, -1].abs() <= _LOW_CONFIDENCE_LOG_TOL).all().item()):
        raise FloatingPointError(f"Sampling CDF is not normalized ({context}).")
    if bool((log_cdf[:, 1:] < log_cdf[:, :-1]).any().item()):
        raise FloatingPointError(f"Sampling CDF is not monotone ({context}).")
    log_cdf.clamp_max_(0.0)
    return _LowConfidenceDistribution(
        logits, log_Z_conf, log_Z_sample, sorted_logits, sorted_token_ids, log_cdf,
    )


class _LowConfidenceStateEvaluator:
    """Canonical singleton forwards with a per-call, bounded CPU logits cache.

    The enclosing estimator enforces eval mode. Custom stochastic/stateful eval
    forwards are unsupported. The cache retains native active logits, not full
    sequence outputs or FP64 vocabulary tensors, and never changes forward shape.
    """
    def __init__(self, model, attention_mask, masked_positions, use_cache=True,
                 cache_max_bytes=64 * 1024 * 1024):
        self.model = model
        self.device = _model_device(model)
        self.attention_mask = attention_mask
        self.masked_positions = masked_positions
        self.cache_max_bytes = cache_max_bytes if use_cache else 0
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_writes_enabled = True
        self.forward_rows = 0

    def context(self, revealed):
        positions = self.masked_positions[revealed].detach().cpu().tolist()
        return f"step={len(positions)}, revealed_indices={[p + 1 for p in positions]}"

    def distribution(self, x_row, revealed, temperature):
        key = tuple(bool(v) for v in revealed.detach().cpu().tolist())
        logits = self.cache.get(key)
        if logits is None:
            autocast = (torch.autocast(device_type=self.device.type, enabled=False)
                        if self.device.type in {'cpu', 'cuda'} else nullcontext())
            with autocast:
                outputs = self.model(x_row.unsqueeze(0).contiguous(),
                                     attention_mask=self.attention_mask)
            logits = outputs.logits[0, self.masked_positions[~revealed], :].contiguous()
            del outputs
            self.forward_rows += 1
            size = logits.numel() * logits.element_size()
            if self.cache_writes_enabled and size <= self.cache_max_bytes:
                while self.cache and self.cache_bytes + size > self.cache_max_bytes:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= old.numel() * old.element_size()
                self.cache[key] = logits.detach().to(device='cpu', copy=True)
                self.cache_bytes += size
        else:
            self.cache.move_to_end(key)
            logits = logits.to(self.device)
        return _low_confidence_distribution(logits, temperature, self.context(revealed))


def _target_probability_state(
    model, x_row, active_abs_positions, active_target_ids, attention_mask,
    temperature, context, decoding_scheme="full", k=1, need_confidence=False,
):
    """STS-shaped singleton forward and FP64 target scoring, without a CDF.

    Random remasking and DUEL do not integrate sampled-confidence competition,
    so vocabulary sorting/CDF construction would be unused work. Full-distribution
    normalizers and target log probabilities use STS's exact arithmetic and
    [remaining, vocabulary] shape. Top-k is a separate supported decoder policy.
    """
    device = x_row.device
    autocast = (torch.autocast(device_type=device.type, enabled=False)
                if device.type in {"cpu", "cuda"} else nullcontext())
    with autocast:
        outputs = model(x_row.unsqueeze(0).contiguous(), attention_mask=attention_mask)
    logits_native = outputs.logits[0, active_abs_positions, :].contiguous()
    del outputs
    logits = logits_native.to(torch.float64)
    del logits_native
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError(f"Expected nonempty [remaining, vocabulary] logits ({context}).")
    if bool((torch.isnan(logits) | torch.isposinf(logits)).any().item()):
        raise FloatingPointError(f"Invalid model logits ({context}): NaN or +inf.")
    tau = float(temperature)
    log_Z_conf = torch.logsumexp(logits, dim=-1)
    if not bool(torch.isfinite(log_Z_conf).all().item()):
        raise FloatingPointError(f"Nonfinite distribution normalizer ({context}).")
    highest_log_confidence = None
    if need_confidence:
        highest_log_confidence = logits.max(dim=-1).values - log_Z_conf
        if tau != 1.0:
            confidence_logs = logits - log_Z_conf.unsqueeze(-1)
            _check_low_confidence_log_mass(confidence_logs, "confidence probabilities", context)
            total = torch.logsumexp(confidence_logs, dim=-1)
            if not bool((torch.isfinite(total) & (total.abs() <= _LOW_CONFIDENCE_LOG_TOL)).all().item()):
                raise FloatingPointError(f"Confidence distribution is not normalized ({context}).")
            del confidence_logs

    target_raw_logits = logits.gather(-1, active_target_ids.unsqueeze(-1)).squeeze(-1)
    in_topk = None
    if decoding_scheme == "top_k" and k < logits.shape[-1]:
        sample_logits, topk_indices = torch.topk(logits, k=int(k), dim=-1)
        in_topk = (topk_indices == active_target_ids.unsqueeze(-1)).any(dim=-1)
        log_Z_sample = torch.logsumexp(sample_logits / tau, dim=-1)
    else:
        sample_logits = logits
        log_Z_sample = (log_Z_conf if tau == 1.0 else
                        torch.logsumexp(logits / tau, dim=-1))
    if not bool(torch.isfinite(log_Z_sample).all().item()):
        raise FloatingPointError(f"Nonfinite distribution normalizer ({context}).")
    sample_logs = sample_logits / tau - log_Z_sample.unsqueeze(-1)
    _check_low_confidence_log_mass(sample_logs, "token probabilities", context)
    total = torch.logsumexp(sample_logs, dim=-1)
    if not bool((torch.isfinite(total) & (total.abs() <= _LOW_CONFIDENCE_LOG_TOL)).all().item()):
        raise FloatingPointError(f"Sampling distribution is not normalized ({context}).")
    target_log_probs = target_raw_logits / tau - log_Z_sample
    if in_topk is not None:
        target_log_probs = target_log_probs.masked_fill(~in_topk, -math.inf)
    _check_low_confidence_log_mass(target_log_probs, "target token probabilities", context)
    return target_log_probs, highest_log_confidence


def _random_remasking_target_log_probs(
    logits, positions, target_ids, temperature, decoding_scheme, k, context,
    normalization_batch_size=128,
):
    """Score only requested positions, in bounded FP64 vocabulary chunks.

    Centered log_softmax avoids subtracting a large raw-logit normalizer from
    the target, retains tiny probabilities as finite log values, and never
    requires materializing probabilities or the full output in double precision.
    """
    batch_size, step_size = positions.shape
    row_ids = torch.arange(batch_size, device=logits.device)[:, None].expand_as(positions).reshape(-1)
    flat_positions, flat_targets = positions.reshape(-1), target_ids.reshape(-1)
    result = torch.empty(flat_targets.numel(), dtype=torch.float64, device=logits.device)
    vocab_size = logits.shape[-1]
    for start in range(0, flat_targets.numel(), normalization_batch_size):
        end = min(start + normalization_batch_size, flat_targets.numel())
        selected = logits[row_ids[start:end], flat_positions[start:end], :].to(torch.float64)
        if bool((torch.isnan(selected) | torch.isposinf(selected)).any().item()):
            raise FloatingPointError(f"Invalid model logits ({context}): NaN or +inf.")
        targets = flat_targets[start:end]
        if decoding_scheme == "top_k" and k < vocab_size:
            selected, token_ids = torch.topk(selected, k=int(k), dim=-1)
            target_matches = token_ids == targets[:, None]
            target_columns = target_matches.to(torch.int64).argmax(dim=-1)
        else:
            target_matches = None
            target_columns = targets
        if temperature != 1.0:
            selected.div_(temperature)
        log_probs = torch.log_softmax(selected, dim=-1)
        _check_low_confidence_log_mass(log_probs, "token probabilities", context)
        target_logs = log_probs.gather(-1, target_columns[:, None]).squeeze(-1)
        if target_matches is not None:
            target_logs = target_logs.masked_fill(~target_matches.any(dim=-1), -math.inf)
        result[start:end] = target_logs
        del selected, log_probs
    return result.view(batch_size, step_size)


@torch.inference_mode()
@_low_confidence_eval_mode
def _path_sampling_random_probability_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,
    masked_indexes: list[int],
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    decoding_scheme: str,
    k: int,
    temperature: float,
    batch_size: int = 128,
    normalization_batch_size: int = 128,
) -> Dict[str, object]:
    """Fast random reveal-order sampling with FP64 log-probability scoring.

    A uniform permutation and fixed schedule select reveal blocks; all tokens
    in each block are scored before any of them are revealed. Model forwards
    run in native dtype, in temporary deterministic eval mode without autocast.
    Real model batches default to 128 for the A100 80GB. The common initial
    state is evaluated once across all samples, and exactly zero paths stop.

    Only selected token positions are normalized, using stable FP64 log_softmax
    in bounded chunks. Path products and the arithmetic mean stay in log space.
    There is no probability floor or small-weight pruning. Numerical identity
    to singleton STS logits is not promised across different model batch shapes.
    STS and DUEL retain their separate execution policies.
    """
    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device)
    if sequence_tokens.shape != (1, 100):
        raise ValueError("sequence_tokens must have shape [1, 100].")
    raw_positions = [int(i) - 1 for i in masked_indexes]
    if len(raw_positions) != len(set(raw_positions)):
        raise ValueError("masked_indexes must not contain duplicate positions.")
    masked_pos = sorted(raw_positions)
    if len(masked_pos) != 50:
        raise ValueError(f"Expected exactly 50 masked positions out of 100, got {len(masked_pos)}")
    if any(pos < 0 or pos >= 100 for pos in masked_pos):
        raise ValueError("masked_indexes must be 1-indexed positions in [1, 100].")
    masked_len = len(masked_pos)
    if not 1 <= steps <= masked_len:
        raise ValueError(f"steps must be in [1, {masked_len}], got {steps}")
    if num_samples <= 0 or batch_size <= 0 or normalization_batch_size <= 0:
        raise ValueError("num_samples, batch_size, and normalization_batch_size must be positive.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and strictly positive.")
    if decoding_scheme not in {"full", "top_k"}:
        raise ValueError("decoding_scheme must be either 'full' or 'top_k'.")
    if decoding_scheme == "top_k" and k <= 0:
        raise ValueError("k must be positive when decoding_scheme='top_k'.")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.shape != (1, 100):
            raise ValueError("attention_mask must have shape [1, 100].")

    tau = float(temperature)
    masked_pos_t = torch.tensor(masked_pos, dtype=torch.long, device=device)
    masked_target_row = sequence_tokens[0, masked_pos_t]
    base, rem = divmod(masked_len, steps)
    schedule = [base + (step < rem) for step in range(steps)]
    rng = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
    # All trajectories start at exactly the same state. Score all possible
    # first reveals once instead of repeating the full model forward 500 times.
    initial_x = sequence_tokens.clone()
    initial_x[:, masked_pos_t] = mask_id
    autocast = (torch.autocast(device_type=device.type, enabled=False)
                if device.type in {"cpu", "cuda"} else nullcontext())
    with autocast:
        outputs = model(initial_x, attention_mask=attention_mask)
    initial_scores = _random_remasking_target_log_probs(
        outputs.logits, masked_pos_t[None, :], masked_target_row[None, :], tau,
        decoding_scheme, k, "step=0, revealed_indices=[]", normalization_batch_size,
    )[0]
    del outputs
    forward_calls, forward_rows, max_forward_batch = 1, 1, 1
    sample_logs = []
    for batch_start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - batch_start)
        permutations = torch.argsort(torch.rand(
            (bsz, masked_len), generator=rng, device="cpu", dtype=torch.float64,
        ), dim=-1).to(device)
        x = initial_x.expand(bsz, -1).clone()
        log_weights = torch.zeros(bsz, dtype=torch.float64, device=device)
        start = 0
        for step, step_size in enumerate(schedule):
            slots = permutations[:, start:start + step_size]
            if step == 0:
                # Every target in a simultaneous reveal block is scored before
                # any of the targets in that block are revealed.
                log_weights += initial_scores[slots].sum(dim=-1)
                x.scatter_(1, masked_pos_t[slots], masked_target_row[slots])
            else:
                # Only -inf paths are dropped. Never exp() a weight to decide
                # whether to retain it: arbitrarily small finite logs live.
                active_rows = torch.nonzero(~torch.isneginf(log_weights), as_tuple=False).flatten()
                active_count = active_rows.numel()
                if active_count == 0:
                    break
                active_slots = slots[active_rows]
                positions = masked_pos_t[active_slots]
                targets = masked_target_row[active_slots]
                active_x = x.index_select(0, active_rows)
                active_attn = (None if attention_mask is None else
                               attention_mask.expand(active_count, -1))
                autocast = (torch.autocast(device_type=device.type, enabled=False)
                            if device.type in {"cpu", "cuda"} else nullcontext())
                with autocast:
                    outputs = model(active_x, attention_mask=active_attn)
                forward_calls += 1
                forward_rows += active_count
                max_forward_batch = max(max_forward_batch, active_count)
                token_logs = _random_remasking_target_log_probs(
                    outputs.logits, positions, targets, tau, decoding_scheme, k,
                    f"step={step}, trajectory_batch_start={batch_start}",
                    normalization_batch_size,
                )
                del outputs, active_x
                log_weights[active_rows] += token_logs.sum(dim=-1)
                x[active_rows[:, None], positions] = targets
            start += step_size
        sample_logs.extend(log_weights.detach().cpu().tolist())

    # Aggregate once in sample order. Model batch shapes may change logits
    # slightly, but seeded reveal permutations are independent of batch size.
    log_values = torch.tensor(sample_logs, dtype=torch.float64)
    _check_low_confidence_log_mass(log_values, "random path probabilities", "completed paths")
    log_probability = float((torch.logsumexp(log_values, dim=0) - math.log(num_samples)).item())
    _check_low_confidence_log_mass(
        torch.tensor(log_probability, dtype=torch.float64), "random mean probability", "completed paths",
    )
    return {
        "probability": 0.0 if log_probability == -math.inf else math.exp(log_probability),
        "log_probability": log_probability,
        "sample_probabilities": torch.exp(log_values).tolist(),
        "sample_log_probabilities": sample_logs,
        "num_samples": num_samples,
        "estimation_method": "path_sampling",
        "decoding_scheme": decoding_scheme,
        "temperature": tau,
        "model_forward_calls": forward_calls,
        "model_forward_batch_size": batch_size,
        "model_forward_max_batch_size": max_forward_batch,
        "model_forward_rows": forward_rows,
        "normalization_batch_size": normalization_batch_size,
        "initial_state_reused": True,
        "model_forward_dtype": "native",
        "model_eval_mode": True,
        "estimator_dtype_after_logits": "float64",
        "trajectory_batch_size": batch_size,
    }


def _verbose_step_record(verbose_batch, log_A, row, step, compact):
    indices = verbose_batch['sequence_indices'][row].tolist()
    log_a = verbose_batch['log_a_active'][row].tolist()
    target_logs = verbose_batch['target_sample_log_probs_64'][row].tolist()
    products = verbose_batch['log_product'][row].tolist()
    possible_mask = verbose_batch['highest_possible'][row].tolist()
    sampled_mask = verbose_batch['highest_sampled'][row].tolist()
    record = {
        'step_index': step,
        'log_A': float(log_A[row].item()),
        'highest_possible_confidence_indices': [
            int(index) for index, selected in zip(indices, possible_mask) if selected
        ],
        'highest_sampled_confidence_indices': [
            int(index) for index, selected in zip(indices, sampled_mask) if selected
        ],
    }
    if compact:
        record.update({
            'sequence_indices': [int(index) for index in indices],
            'log_a_active': log_a,
            'target_sample_log_probs_64': target_logs,
            'log_product': products,
        })
    else:
        record['candidates'] = [
            {
                'sequence_index': int(index),
                'log_a_active': float(log_a_value),
                'target_sample_log_probs_64': float(target_log),
                'log_product': float(product),
            }
            for index, log_a_value, target_log, product
            in zip(indices, log_a, target_logs, products)
        ]
    return record, int(verbose_batch['tie_count'][row].item())


@torch.inference_mode()
@_low_confidence_eval_mode
def _path_sampling_low_confidence_probability_fast_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, 100], full target sequence z
    masked_indexes: list[int],              # 1-indexed masked positions M
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    temperature: float,
    batch_size: int = 1024,
    validate_no_ties: bool = False,  # retained for API compatibility; ties use smallest index
    return_samples: bool = True,
    verbose: bool = False,
    verbose_compact: bool = False,
    use_state_cache: bool = True,
) -> Dict[str, object]:
    """
    Fast successful-trajectory estimator for low-confidence remasking.

    Estimates

        p_{theta, phi, M}(z)
        =
        p_{theta, phi, M}(z_M | z_not_M)

    in the one-token-per-step setting.

    The estimator is

        hat p_z = prod_t A(S_{t-1}),

    where ties at the maximum confidence are resolved by taking the smallest
    sequence index.  For a proposed successful winner i, define

        L_{j,i}(S)  = P[c_j(V_j | S) <  c_i^*(S)]
        LE_{j,i}(S) = P[c_j(V_j | S) <= c_i^*(S)].

    Then a smaller-index competitor j < i must be strictly below i, while a
    larger-index competitor j > i may be below or tied with i.  Thus a_i(S) is
    the target-token sampling probability p_i(z_i | S) times

        prod_{j < i} L_{j,i}(S) * prod_{j > i} LE_{j,i}(S).

    All confidence comparisons are performed directly in FP64 log-confidence
    space; no raw-logit confidence threshold is reconstructed.

        A(S) = sum_i a_i(S),

        q(i | S) = a_i(S) / A(S).

    IMPORTANT TEMPERATURE DISTINCTION
    ---------------------------------
    Candidate sampling uses

        p_i(v | S)
        =
        softmax(logits_i / temperature)_v,

    while low-confidence ranking uses the UNTEMPERED confidence

        c_i(v | S)
        =
        softmax(logits_i)_v.

    NUMERICAL STRATEGY
    ------------------
    To avoid the VRAM cost of model.double():
      - the model forward stays in the model's existing/native dtype;
      - autocast is disabled around the forward so an outer autocast context
        cannot silently change that dtype;
      - logits at active masked positions are immediately promoted to FP64;
      - from that point onward, all full-vocabulary normalizers, sorting, CDFs,
        log-confidence comparisons, successful-transition probabilities, proposal
        probabilities, trajectory log weights, and Monte Carlo averaging are FP64;
      - all multiplicative probabilities are accumulated in log-space.

    STATE CACHE
    -----------
    When use_state_cache=True, log a(S) is memoized by the revealed-set state
    S.  Identical states are deduplicated within a Monte Carlo batch and
    reused across later batches, so the expensive estimator computation is
    performed only once per unique state encountered.

    In verbose mode, deterministic diagnostic subvalues are cached alongside
    log a(S): target log-probabilities, the tie-adjusted competitor win mass,
    highest-possible-position flags, and tie counts.  highest_sampled is NOT
    cached: it is freshly resampled for every trajectory occurrence.  Cache
    misses draw those samples while the full-vocabulary CDF is already present;
    cache hits use the shared state evaluator for diagnostic resampling without
    recomputing A(S). Native active logits have a separate 64 MiB CPU LRU cache,
    shared in implementation with direct MC. use_state_cache=False disables both
    caches for this call.

    Model forwards always use batch size one and temporary eval mode; the
    caller's module training flags are restored on return or error. batch_size
    controls trajectory sampling only. Both estimators use identical per-state
    FP64 distribution calculations. Custom stochastic/stateful eval forwards
    remain unsupported.

    The default trajectory batch is 1024, covering typical 300-1000-sample runs
    in one batch. Identical states are grouped within each step. State results
    are retained only when later trajectory batches can reuse them. Native
    logits are cached only for verbose diagnostics in those later batches;
    ordinary estimation reuses the much smaller log-a cache instead. Changing
    batch_size changes seeded proposal draws, but not the state evaluator or
    the STS proposal and importance-weight formulas.

    TIE BREAKING
    ------------
    If multiple sampled positions share the maximum confidence, the decoder
    chooses the smallest sequence index deterministically.  The estimator
    marginalizes this rule exactly inside a_i(S): smaller-index competitors use
    strict '<' mass and larger-index competitors use '<=' mass.

    Note: validate_no_ties is retained only for backwards API compatibility and
    no longer changes the estimator; ties are supported exactly.

    Assumptions:
      - sequence_tokens has shape [1, 100];
      - masked_indexes contains between 1 and 100 valid, unique,
        1-indexed positions;
      - steps == len(masked_indexes);
      - attention_mask, if provided, has shape [1, 100].
    """

    device = _model_device(model)

    # Keep model parameters/buffers in their existing dtype to avoid the VRAM
    # cost of model.double().  Only the active-position logits are promoted to
    # FP64 after the forward pass.
    sequence_tokens = sequence_tokens.to(device)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError(
            f"sequence_tokens must have shape [1, 100], "
            f"got {tuple(sequence_tokens.shape)}"
        )

    seq_len = sequence_tokens.shape[1]

    if seq_len != 100:
        raise ValueError(
            f"Expected sequence length 100, got {seq_len}"
        )

    # Convert 1-indexed masked positions to sorted 0-indexed slots.
    raw_masked_pos = [int(i) - 1 for i in masked_indexes]
    if not raw_masked_pos:
        raise ValueError("masked_indexes must contain at least one position.")
    if len(raw_masked_pos) != len(set(raw_masked_pos)):
        raise ValueError("masked_indexes must not contain duplicate positions.")
    masked_pos = sorted(raw_masked_pos)

    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError(
            "masked_indexes must be 1-indexed positions in [1, 100]"
        )

    masked_len = len(masked_pos)

    if steps != masked_len:
        raise ValueError(
            "Low-confidence path estimator reveals exactly one masked "
            "token per step, so steps must equal "
            f"len(masked_indexes)={masked_len}."
        )

    if (
        not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError(
            "Low-confidence full-distribution estimator requires "
            "finite temperature > 0."
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

        if attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )

    tau = float(temperature)

    masked_pos_t = torch.tensor(
        masked_pos,
        dtype=torch.long,
        device=device,
    )  # [masked_len]

    full_target_row = sequence_tokens[0]
    masked_target_row = full_target_row[masked_pos_t]
    # [masked_len]

    # ------------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------------

    if device.type in {"cuda", "cpu"}:
        rng_device = device
        sample_on_device = True
    else:
        rng_device = torch.device("cpu")
        sample_on_device = False

    # seed=None means use PyTorch's global RNG rather than constructing a
    # generator with its fixed default seed.
    if seed is None:
        rng = None
    else:
        rng = torch.Generator(
            device=rng_device
        )
        rng.manual_seed(
            int(seed)
        )

    diagnostic_rng = None
    if verbose:
        diagnostic_seed = (
            secrets.randbits(63)
            if seed is None
            else (int(seed) + 0x5DEECE66D) % (2**63 - 1)
        )
        diagnostic_rng = torch.Generator(device=rng_device)
        diagnostic_rng.manual_seed(diagnostic_seed)

    # ------------------------------------------------------------------
    # Output accumulators
    # ------------------------------------------------------------------

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []
    verbose_samples: List[Dict[str, object]] = []

    running_log_sum = torch.tensor(
        float("-inf"),
        dtype=torch.float64,
        device=device,
    )

    num_accumulated = 0

    # ------------------------------------------------------------------
    # Small static tensors reused at every denoising step.  Keeping these
    # outside the hot loop avoids repeated CUDA allocations/kernel launches.
    # ------------------------------------------------------------------
    max_bsz = min(batch_size, num_samples)
    slot_grid_base = torch.arange(
        masked_len, dtype=torch.long, device=device
    ).unsqueeze(0)
    eye_base = torch.eye(
        masked_len, dtype=torch.bool, device=device
    )
    reveal_true_base = torch.ones(
        (max_bsz, 1), dtype=torch.bool, device=device
    )

    # The estimator cache stays on the model device: each entry is only a
    # masked_len-element FP64 log-a vector.  Revealed-set keys are tuples of
    # booleans, so the cache supports any mask count without int64 overflow.
    # Verbose deterministic diagnostics are tiny and are cached on CPU so
    # verbose caching adds negligible VRAM.
    #
    # highest_sampled is intentionally never cached.  It is a stochastic
    # diagnostic and is resampled independently for every trajectory occurrence.
    cache_enabled = bool(use_state_cache)
    retain_state_results = True
    state_evaluator = _LowConfidenceStateEvaluator(
        model, attention_mask, masked_pos_t, use_cache=use_state_cache,
    )
    state_log_a_cache: Dict[Tuple[bool, ...], torch.Tensor] = {}
    state_verbose_cache: Dict[Tuple[bool, ...], Dict[str, torch.Tensor]] = {}
    cache_requests = 0
    cache_hits = 0
    cache_misses = 0
    cache_forward_rows_saved = 0
    verbose_diagnostic_forward_rows = 0

    # ==================================================================
    # Compute log a_i(S)
    # ==================================================================

    def _compute_log_a_for_state(
        x: torch.Tensor,              # [bsz, 100]
        revealed: torch.Tensor,       # [bsz, masked_len]
        alive: torch.Tensor,          # [bsz]
        num_unrevealed: int,
        verbose_draw_counts: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, object]]]:
        """
        Returns

            log_a_full : [bsz, masked_len], float64

        where revealed positions and dead trajectories are -inf.

        Full-vocab work is vectorized across all currently masked positions.
        """

        bsz = x.shape[0]
        m = num_unrevealed

        unrevealed = ~revealed

        # --------------------------------------------------------------
        # Identify active masked slots.
        #
        # Every trajectory has exactly m unrevealed slots because exactly
        # one slot is marked revealed at every step, including dead rows.
        # --------------------------------------------------------------

        slot_grid = slot_grid_base.expand(bsz, -1)

        active_slots = slot_grid[
            unrevealed
        ].view(
            bsz,
            m,
        )
        # [bsz, m]

        active_abs_positions = masked_pos_t[
            active_slots
        ]
        # [bsz, m]

        active_target_ids = masked_target_row[
            active_slots
        ]
        # [bsz, m]

        # This function always receives one state. Both estimators use the
        # same singleton forward and [remaining, vocabulary] reductions.
        distribution = state_evaluator.distribution(x[0], revealed[0], tau)
        active_logits = distribution.logits.unsqueeze(0)
        log_Z_conf = distribution.log_Z_conf.unsqueeze(0)
        log_Z_sample = distribution.log_Z_sample.unsqueeze(0)
        vocab_size = active_logits.shape[-1]

        # Target raw logits l_i(z_i).
        target_raw_logits = torch.gather(
            active_logits,
            dim=-1,
            index=active_target_ids.unsqueeze(-1),
        ).squeeze(-1)
        # [bsz, m]

        # log p_i(z_i):
        target_sample_log_probs = (
            target_raw_logits / tau
            - log_Z_sample
        )
        # [bsz, m], FP64

        # log c_i^* = log c_i(z_i):
        target_conf_log_probs = (
            target_raw_logits
            - log_Z_conf
        )
        # [bsz, m], FP64

        # --------------------------------------------------------------
        # With one masked position there are no competitors. Reuse the shared
        # distribution, but skip the successful-winner competition machinery.
        # --------------------------------------------------------------
        if m == 1:
            log_a_active = target_sample_log_probs
            log_product = torch.zeros_like(log_a_active)
            target_sample_log_probs_64 = target_sample_log_probs

            log_a_full = torch.full(
                (bsz, masked_len),
                float("-inf"),
                dtype=torch.float64,
                device=device,
            )
            log_a_full.scatter_(
                dim=1,
                index=active_slots,
                src=log_a_active,
            )
            log_a_full.masked_fill_(~alive.unsqueeze(-1), float("-inf"))

            verbose_batch = None
            if verbose:
                active_alive = alive.unsqueeze(-1)
                only_position = torch.ones(
                    (bsz, 1), dtype=torch.bool, device=device
                )
                draw_counts = (
                    [1] * bsz
                    if verbose_draw_counts is None
                    else [int(v) for v in verbose_draw_counts]
                )
                if len(draw_counts) != bsz or any(v <= 0 for v in draw_counts):
                    raise ValueError(
                        "verbose_draw_counts must contain one positive count per row."
                    )

                # With one remaining position, highest_sampled is deterministically
                # that position and consumes no diagnostic RNG.
                highest_sampled_draws = [
                    torch.ones((count, 1), dtype=torch.bool)
                    for count in draw_counts
                ]

                verbose_batch = {
                    'sequence_indices': (active_abs_positions + 1).detach().cpu(),
                    'log_a_active': log_a_active.detach().cpu(),
                    'target_sample_log_probs_64': target_sample_log_probs_64.detach().cpu(),
                    'log_product': log_product.detach().cpu(),
                    'highest_possible': (only_position & active_alive).detach().cpu(),
                    'highest_sampled': (only_position & active_alive).detach().cpu(),
                    'tie_count': torch.zeros(
                        bsz, dtype=torch.long, device=device
                    ).detach().cpu(),
                    # Internal-only: one [num_occurrences, m] tensor per state row.
                    '_highest_sampled_draws': highest_sampled_draws,
                }

            del active_logits
            return log_a_full, verbose_batch

        highest_possible_active = None
        highest_sampled_active = None
        if verbose:
            maximum_confidence = active_logits.max(dim=-1).values - log_Z_conf
            highest_possible_active = maximum_confidence == maximum_confidence.max(
                dim=-1, keepdim=True
            ).values
            del maximum_confidence

        sorted_logits = distribution.sorted_logits.unsqueeze(0)
        log_cdf = distribution.log_cdf.unsqueeze(0)
        del active_logits

        highest_sampled_draws = None
        if verbose:
            draw_counts = (
                [1] * bsz
                if verbose_draw_counts is None
                else [int(v) for v in verbose_draw_counts]
            )
            if len(draw_counts) != bsz or any(v <= 0 for v in draw_counts):
                raise ValueError(
                    "verbose_draw_counts must contain one positive count per row."
                )

            # Draw independently for every occurrence represented by this row.
            # For cache misses this lets one unique-state estimator evaluation
            # supply fresh highest_sampled diagnostics to all duplicate rows.
            highest_sampled_draws = []
            first_draws = []

            for row in range(bsz):
                count = draw_counts[row]

                if sample_on_device:
                    diagnostic_uniform = torch.rand(
                        (m, count),
                        device=device,
                        dtype=torch.float64,
                        generator=diagnostic_rng,
                    )
                else:
                    diagnostic_uniform = torch.rand(
                        (m, count),
                        device='cpu',
                        dtype=torch.float64,
                        generator=diagnostic_rng,
                    ).to(device)

                sampled_sorted_slots = torch.searchsorted(
                    log_cdf[row],
                    diagnostic_uniform.log(),
                    right=False,
                ).clamp_max(vocab_size - 1)
                # [m, count]

                sampled_logits = torch.gather(
                    sorted_logits[row],
                    dim=-1,
                    index=sampled_sorted_slots,
                )
                sampled_confidence = (
                    sampled_logits
                    - log_Z_conf[row].unsqueeze(-1)
                )
                highest = (
                    sampled_confidence
                    == sampled_confidence.max(dim=0, keepdim=True).values
                ).transpose(0, 1)
                # [count, m]

                first_draws.append(highest[0])
                highest_sampled_draws.append(highest.detach().cpu())

            highest_sampled_active = torch.stack(first_draws, dim=0)


        # --------------------------------------------------------------
        # Direct FP64 log-confidence comparison.
        #
        # For competitor position j and proposed successful winner i,
        # compare
        #
        #     log c_j(V_j)
        #
        # directly with
        #
        #     log c_i^*.
        #
        # Because sorted_logits is already ordered within each position,
        # subtracting that position's log Z preserves the ordering.  We can
        # therefore search directly in sorted log-confidence space without
        # reconstructing a raw-logit threshold.
        # --------------------------------------------------------------

        sorted_log_confidence = (
            sorted_logits
            - log_Z_conf.unsqueeze(-1)
        )
        # [bsz, competitor j, vocab], FP64

        target_conf_values = (
            target_conf_log_probs
            .unsqueeze(1)
            .expand(-1, m, -1)
            .contiguous()
        )
        # [bsz, competitor j, proposed winner i], FP64

        # left_idx: first competitor token with confidence >= c_i*.
        # right_idx: first competitor token with confidence >  c_i*.
        left_idx = torch.searchsorted(
            sorted_log_confidence,
            target_conf_values,
            right=False,
        )
        right_idx = torch.searchsorted(
            sorted_log_confidence,
            target_conf_values,
            right=True,
        )
        # [bsz, competitor j, proposed winner i]

        # --------------------------------------------------------------
        # L_{j,i} = P(confidence < c_i*) in log-space.
        # --------------------------------------------------------------

        left_gather_idx = (left_idx - 1).clamp(
            min=0,
            max=vocab_size - 1,
        )

        log_L = torch.gather(
            log_cdf,
            dim=-1,
            index=left_gather_idx,
        )

        log_L.masked_fill_(left_idx == 0, float("-inf"))
        log_L.masked_fill_(left_idx == vocab_size, 0.0)
        # [bsz, competitor j, proposed winner i], FP64

        # --------------------------------------------------------------
        # LE_{j,i} = P(confidence <= c_i*) in log-space.
        # --------------------------------------------------------------

        right_gather_idx = (right_idx - 1).clamp(
            min=0,
            max=vocab_size - 1,
        )

        log_LE = torch.gather(
            log_cdf,
            dim=-1,
            index=right_gather_idx,
        )

        log_LE.masked_fill_(right_idx == 0, float("-inf"))
        log_LE.masked_fill_(right_idx == vocab_size, 0.0)
        # [bsz, competitor j, proposed winner i], FP64

        # Potential exact cross-position ties are retained as a diagnostic.
        equal_count = right_idx - left_idx
        has_equal = equal_count > 0
        eye_m = eye_base[:m, :m].unsqueeze(0)
        cross_tie_mask = has_equal & (~eye_m)

        tie_count = torch.zeros(bsz, dtype=torch.long, device=device)
        if verbose and bool(cross_tie_mask.any().item()):
            tie_count = (
                cross_tie_mask & alive.view(bsz, 1, 1)
            ).sum(dim=(1, 2))

        # validate_no_ties is deliberately ignored.  It remains in the public
        # signature only so existing callers do not break; ties are handled
        # exactly by deterministic smallest-index tie-breaking.
        _ = validate_no_ties

        # --------------------------------------------------------------
        # Smallest-index tie rule.
        #
        # active_slots is in increasing masked-slot order, and masked_pos is
        # sorted, so local active index order is sequence-index order.
        # For proposed winner i:
        #   competitor j < i  -> must satisfy confidence <  c_i*
        #   competitor j > i  -> may satisfy confidence <= c_i*
        #   competitor j == i -> neutral factor 1
        # --------------------------------------------------------------

        proposed_i = torch.arange(
            m,
            dtype=torch.long,
            device=device,
        ).view(1, m)

        log_smallest_index_win_mass = torch.zeros(
            (bsz, m),
            dtype=torch.float64,
            device=device,
        )

        # Accumulate in competitor order, preserving the previous no-tie
        # multiplication/addition order as closely as possible.
        for competitor_j in range(m):
            strict_for_smaller = (competitor_j < proposed_i)
            non_strict_for_larger = (competitor_j > proposed_i)

            competitor_factor = torch.where(
                strict_for_smaller,
                log_L[:, competitor_j, :],
                torch.where(
                    non_strict_for_larger,
                    log_LE[:, competitor_j, :],
                    torch.zeros(
                        (),
                        dtype=torch.float64,
                        device=device,
                    ),
                ),
            )

            log_smallest_index_win_mass = (
                log_smallest_index_win_mass
                + competitor_factor
            )

        log_a_active = (
            target_sample_log_probs
            + log_smallest_index_win_mass
        )
        # [bsz, m], FP64

        # Retain the existing diagnostic field name/shape.  It is the log
        # probability that the competitor field permits i to win under the
        # deterministic smallest-index tie rule.
        log_product = log_smallest_index_win_mass
        target_sample_log_probs_64 = target_sample_log_probs

        # No longer needed before returning.
        del sorted_logits
        del sorted_log_confidence
        del target_conf_values
        del log_cdf
        del log_L
        del log_LE
        del left_idx
        del right_idx
        del left_gather_idx
        del right_gather_idx
        del equal_count
        del has_equal
        del eye_m
        del cross_tie_mask
        del proposed_i
        del competitor_factor
        del log_smallest_index_win_mass

        # --------------------------------------------------------------
        # Scatter back into the fixed masked-slot representation.
        # --------------------------------------------------------------

        log_a_full = torch.full(
            (bsz, masked_len),
            float("-inf"),
            dtype=torch.float64,
            device=device,
        )

        log_a_full.scatter_(
            dim=1,
            index=active_slots,
            src=log_a_active,
        )

        # Dead trajectories have zero contribution.
        log_a_full.masked_fill_(~alive.unsqueeze(-1), float("-inf"))

        verbose_batch = None
        if verbose:
            active_alive = alive.unsqueeze(-1)
            verbose_batch = {
                'sequence_indices': (active_abs_positions + 1).detach().cpu(),
                'log_a_active': log_a_active.detach().cpu(),
                'target_sample_log_probs_64': target_sample_log_probs_64.detach().cpu(),
                'log_product': log_product.detach().cpu(),
                'highest_possible': (
                    highest_possible_active & active_alive
                ).detach().cpu(),
                'highest_sampled': (
                    highest_sampled_active & active_alive
                ).detach().cpu(),
                'tie_count': tie_count.detach().cpu(),
                # Internal-only; omitted from user-visible step records.
                '_highest_sampled_draws': highest_sampled_draws,
            }

        return log_a_full, verbose_batch

    def _compute_log_a_for_batch_uncached(
        x, revealed, alive, num_unrevealed, verbose_draw_counts=None,
    ):
        # Trajectories may be batched, but model and vocabulary arithmetic
        # always operate on exactly one state, just as in direct MC.
        log_a_rows = []
        diagnostic_rows = []
        for row in range(x.shape[0]):
            if not bool(alive[row].item()):
                # Dummy reveal paths after zero success mass never need a
                # forward, even when caching is disabled.
                log_a_rows.append(torch.full(
                    (1, masked_len), float('-inf'), dtype=torch.float64, device=device,
                ))
                if verbose:
                    positions = masked_pos_t[~revealed[row]] + 1
                    zeros = torch.zeros((1, num_unrevealed), dtype=torch.bool)
                    minus_inf = torch.full((1, num_unrevealed), float('-inf'),
                                           dtype=torch.float64)
                    count = 1 if verbose_draw_counts is None else verbose_draw_counts[row]
                    diagnostic_rows.append({
                        'sequence_indices': positions.unsqueeze(0).detach().cpu(),
                        'log_a_active': minus_inf,
                        'target_sample_log_probs_64': minus_inf,
                        'log_product': minus_inf,
                        'highest_possible': zeros,
                        'highest_sampled': zeros,
                        'tie_count': torch.zeros(1, dtype=torch.long),
                        '_highest_sampled_draws': [zeros.expand(count, -1)],
                    })
                continue
            log_a_row, diagnostic = _compute_log_a_for_state(
                x[row:row + 1], revealed[row:row + 1], alive[row:row + 1],
                num_unrevealed,
                None if verbose_draw_counts is None else [verbose_draw_counts[row]],
            )
            _check_low_confidence_log_mass(
                log_a_row, 'successful transition masses',
                state_evaluator.context(revealed[row]),
            )
            log_a_rows.append(log_a_row)
            if verbose:
                diagnostic_rows.append(diagnostic)
        merged = None
        if verbose:
            merged = {
                key: (sum((row[key] for row in diagnostic_rows), [])
                      if key == '_highest_sampled_draws' else
                      torch.cat([row[key] for row in diagnostic_rows], dim=0))
                for key in diagnostic_rows[0]
            }
        return torch.cat(log_a_rows, dim=0), merged

    def _resample_highest_sampled_for_cached_states(
        x, revealed, representative_rows, draw_counts, num_unrevealed,
    ):
        """Fresh diagnostics using the same state evaluator and distribution."""
        nonlocal verbose_diagnostic_forward_rows
        draws = []
        for row, count in zip(representative_rows, draw_counts):
            if num_unrevealed == 1:
                draws.append(torch.ones((int(count), 1), dtype=torch.bool))
                continue
            previous_forwards = state_evaluator.forward_rows
            distribution = state_evaluator.distribution(x[row], revealed[row], tau)
            verbose_diagnostic_forward_rows += state_evaluator.forward_rows - previous_forwards
            uniforms = torch.rand(
                (num_unrevealed, int(count)), dtype=torch.float64,
                device=rng_device, generator=diagnostic_rng,
            ).to(device)
            sampled_slots = torch.searchsorted(
                distribution.log_cdf, uniforms.log(), right=False,
            ).clamp_max(distribution.sorted_logits.shape[-1] - 1)
            sampled_logs = torch.gather(
                distribution.sorted_logits, -1, sampled_slots,
            ) - distribution.log_Z_conf.unsqueeze(-1)
            highest = sampled_logs == sampled_logs.max(dim=0, keepdim=True).values
            draws.append(highest.transpose(0, 1).detach().cpu())
        return draws

    def compute_log_a_for_batch(
        x: torch.Tensor,
        revealed: torch.Tensor,
        alive: torch.Tensor,
        num_unrevealed: int,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Memoized wrapper around the expensive state evaluation.

        S is exactly the revealed subset because revealed positions are always
        fixed to target z_i and unrevealed positions stay masked.

        In verbose mode:
          * deterministic state diagnostics are cached;
          * highest_sampled is always freshly drawn per trajectory occurrence;
          * cache misses draw while the estimator CDF is already in memory;
          * cache hits use the lightweight diagnostic resampling path above.
        """
        nonlocal cache_requests, cache_hits, cache_misses, cache_forward_rows_saved

        if not cache_enabled:
            return _compute_log_a_for_batch_uncached(
                x=x,
                revealed=revealed,
                alive=alive,
                num_unrevealed=num_unrevealed,
                verbose_draw_counts=None,
            )

        bsz = x.shape[0]
        m = num_unrevealed

        out = torch.full(
            (bsz, masked_len),
            float("-inf"),
            dtype=torch.float64,
            device=device,
        )

        # Prepare verbose outputs for every row.  Dead rows remain explicit
        # placeholders (-inf/False/0); live rows are filled from cache/evaluation.
        verbose_out = None
        highest_sampled_out = None
        active_slots_all = None
        if verbose:
            slot_grid = slot_grid_base.expand(bsz, -1)
            active_slots_all = slot_grid[
                ~revealed
            ].view(bsz, m)
            sequence_indices_cpu = (
                masked_pos_t[active_slots_all] + 1
            ).detach().cpu()

            verbose_out = {
                'sequence_indices': sequence_indices_cpu,
                'log_a_active': torch.full(
                    (bsz, m), float("-inf"), dtype=torch.float64
                ),
                'target_sample_log_probs_64': torch.full(
                    (bsz, m), float("-inf"), dtype=torch.float64
                ),
                'log_product': torch.full(
                    (bsz, m), float("-inf"), dtype=torch.float64
                ),
                'highest_possible': torch.zeros(
                    (bsz, m), dtype=torch.bool
                ),
                'highest_sampled': torch.zeros(
                    (bsz, m), dtype=torch.bool
                ),
                'tie_count': torch.zeros(
                    bsz, dtype=torch.long
                ),
            }
            highest_sampled_out = verbose_out['highest_sampled']

        live_rows_t = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        num_live = int(live_rows_t.numel())

        if num_live == 0:
            return out, verbose_out

        # One synchronization per step/batch: revealed-set keys for live
        # trajectories are transferred to CPU.
        live_rows = live_rows_t.detach().cpu().tolist()
        live_states = revealed[live_rows_t].detach().cpu().tolist()
        cache_requests += num_live

        # Group rows by state while preserving first-occurrence order.
        rows_by_state: Dict[Tuple[bool, ...], List[int]] = {}
        for row, state in zip(live_rows, live_states):
            key = tuple(bool(value) for value in state)
            rows_by_state.setdefault(key, []).append(int(row))

        missing_keys: List[Tuple[bool, ...]] = []
        missing_representative_rows: List[int] = []
        missing_draw_counts: List[int] = []

        cached_diag_keys: List[Tuple[bool, ...]] = []
        cached_diag_representative_rows: List[int] = []
        cached_diag_draw_counts: List[int] = []

        def _fill_verbose_deterministic(
            rows: List[int],
            entry: Dict[str, torch.Tensor],
        ) -> None:
            if not verbose or verbose_out is None:
                return

            row_idx_cpu = torch.tensor(rows, dtype=torch.long)

            for field in (
                'target_sample_log_probs_64',
                'log_product',
                'highest_possible',
            ):
                value = entry[field]
                verbose_out[field].index_copy_(
                    0,
                    row_idx_cpu,
                    value.unsqueeze(0).expand(len(rows), -1),
                )

            verbose_out['tie_count'].index_copy_(
                0,
                row_idx_cpu,
                entry['tie_count'].reshape(1).expand(len(rows)),
            )

        # First satisfy states already cached.
        for key, rows in rows_by_state.items():
            cached = state_log_a_cache.get(key)

            if cached is not None:
                row_idx = torch.tensor(rows, dtype=torch.long, device=device)
                out.index_copy_(
                    0,
                    row_idx,
                    cached.unsqueeze(0).expand(len(rows), -1),
                )
                cache_hits += len(rows)
                cache_forward_rows_saved += len(rows)

                if verbose:
                    deterministic = state_verbose_cache.get(key)
                    if deterministic is None:
                        raise RuntimeError(
                            "Verbose state cache entry missing deterministic diagnostics."
                        )
                    _fill_verbose_deterministic(rows, deterministic)

                    cached_diag_keys.append(key)
                    cached_diag_representative_rows.append(rows[0])
                    cached_diag_draw_counts.append(len(rows))
            else:
                missing_keys.append(key)
                missing_representative_rows.append(rows[0])
                missing_draw_counts.append(len(rows))

                # Only one representative state needs the expensive estimator
                # computation.  The remaining duplicate rows are saved forwards.
                duplicate_rows = max(0, len(rows) - 1)
                cache_hits += duplicate_rows
                cache_forward_rows_saved += duplicate_rows

        # Evaluate each genuinely new state once.
        if missing_keys:
            rep_idx = torch.tensor(
                missing_representative_rows,
                dtype=torch.long,
                device=device,
            )
            missing_x = x.index_select(0, rep_idx)
            missing_revealed = revealed.index_select(0, rep_idx)
            missing_alive = torch.ones(
                len(missing_keys), dtype=torch.bool, device=device
            )

            missing_log_a, missing_verbose = _compute_log_a_for_batch_uncached(
                x=missing_x,
                revealed=missing_revealed,
                alive=missing_alive,
                num_unrevealed=num_unrevealed,
                verbose_draw_counts=(
                    missing_draw_counts if verbose else None
                ),
            )

            cache_misses += len(missing_keys)

            if verbose:
                if missing_verbose is None:
                    raise RuntimeError(
                        "Verbose diagnostics were not produced for cache misses."
                    )
                missing_draws = missing_verbose.get('_highest_sampled_draws')
                if missing_draws is None:
                    raise RuntimeError(
                        "Fresh highest_sampled draws were not produced."
                    )

            for local_idx, key in enumerate(missing_keys):
                value = missing_log_a[local_idx]
                if retain_state_results:
                    value = value.clone()
                    state_log_a_cache[key] = value

                rows = rows_by_state[key]
                row_idx = torch.tensor(rows, dtype=torch.long, device=device)
                out.index_copy_(
                    0,
                    row_idx,
                    value.unsqueeze(0).expand(len(rows), -1),
                )

                if verbose:
                    deterministic = {
                        'target_sample_log_probs_64':
                            missing_verbose['target_sample_log_probs_64'][local_idx].clone(),
                        'log_product':
                            missing_verbose['log_product'][local_idx].clone(),
                        'highest_possible':
                            missing_verbose['highest_possible'][local_idx].clone(),
                        'tie_count':
                            missing_verbose['tie_count'][local_idx].clone(),
                    }
                    if retain_state_results:
                        state_verbose_cache[key] = deterministic
                    _fill_verbose_deterministic(rows, deterministic)

                    state_draws = missing_draws[local_idx]
                    if state_draws.shape != (len(rows), m):
                        raise RuntimeError(
                            "Unexpected highest_sampled draw shape for cache miss: "
                            f"{tuple(state_draws.shape)} vs {(len(rows), m)}"
                        )
                    highest_sampled_out[
                        torch.tensor(rows, dtype=torch.long)
                    ] = state_draws

        # Cached states no longer have a vocabulary sampler in memory.  Re-run
        # only the lightweight diagnostic sampling path, once per unique cached
        # state, and draw independently for every occurrence.
        if verbose and cached_diag_keys:
            cached_draws = _resample_highest_sampled_for_cached_states(
                x=x,
                revealed=revealed,
                representative_rows=cached_diag_representative_rows,
                draw_counts=cached_diag_draw_counts,
                num_unrevealed=num_unrevealed,
            )

            for key, state_draws in zip(cached_diag_keys, cached_draws):
                rows = rows_by_state[key]
                if state_draws.shape != (len(rows), m):
                    raise RuntimeError(
                        "Unexpected highest_sampled draw shape for cache hit: "
                        f"{tuple(state_draws.shape)} vs {(len(rows), m)}"
                    )
                highest_sampled_out[
                    torch.tensor(rows, dtype=torch.long)
                ] = state_draws

        if verbose:
            # Derive log_a_active directly from the cached/full masked-slot output,
            # so it is guaranteed to match the values used for A(S) and q.
            verbose_out['log_a_active'] = torch.gather(
                out,
                dim=1,
                index=active_slots_all,
            ).detach().cpu()

        return out, verbose_out

    # ==================================================================
    # Successful trajectory Monte Carlo
    # ==================================================================

    for batch_start in range(
        0,
        num_samples,
        batch_size,
    ):
        bsz = min(
            batch_size,
            num_samples - batch_start,
        )

        # A state only appears at its corresponding reveal count. All its
        # occurrences in this batch are grouped at that step, so only LATER
        # batches can reuse it. Logits are useful only for verbose cache hits:
        # ordinary cache hits already have the complete log-a vector.
        retain_state_results = batch_start + bsz < num_samples
        state_evaluator.cache_writes_enabled = verbose and retain_state_results

        # Initial state:
        #
        # observed tokens = target z
        # masked positions = M
        x = sequence_tokens.expand(
            bsz,
            -1,
        ).clone()

        x[:, masked_pos_t] = mask_id

        revealed = torch.zeros(
            (bsz, masked_len),
            dtype=torch.bool,
            device=device,
        )

        # Accumulate
        #
        #     log W
        #       =
        #     sum_t log A(S_{t-1})
        #
        # entirely in FP64.
        log_weight = torch.zeros(
            bsz,
            dtype=torch.float64,
            device=device,
        )

        alive = torch.ones(
            bsz,
            dtype=torch.bool,
            device=device,
        )

        batch_verbose: List[Dict[str, object]] = []
        if verbose:
            batch_verbose = [
                {
                    'sample_index': batch_start + row,
                    'sample_log_estimate': None,
                    'reveal_path_indices': [],
                    'tie_count': 0,
                    'steps': [],
                }
                for row in range(bsz)
            ]

        for step in range(masked_len):
            unrevealed = ~revealed
            num_unrevealed = masked_len - step

            # ----------------------------------------------------------
            # log a_i(S)
            # ----------------------------------------------------------

            log_a, verbose_batch = compute_log_a_for_batch(
                x=x,
                revealed=revealed,
                alive=alive,
                num_unrevealed=num_unrevealed,
            )
            # [bsz, masked_len], FP64

            # ----------------------------------------------------------
            # A(S) = sum_i a_i(S)
            # ----------------------------------------------------------

            log_A = torch.logsumexp(
                log_a,
                dim=-1,
            )
            # [bsz], FP64

            if verbose:
                if verbose_batch is None:
                    raise RuntimeError('Verbose step data was not produced.')
                log_A_cpu = log_A.detach().cpu()
                for row in range(bsz):
                    step_record, step_ties = _verbose_step_record(
                        verbose_batch, log_A_cpu, row, step, verbose_compact
                    )
                    batch_verbose[row]['steps'].append(step_record)
                    batch_verbose[row]['tie_count'] += step_ties

            invalid_A = torch.isnan(log_A) | torch.isposinf(log_A)
            invalid_A |= log_A > _LOW_CONFIDENCE_LOG_TOL
            if bool(invalid_A.any().item()):
                row = int(torch.nonzero(invalid_A, as_tuple=False)[0, 0].item())
                _check_low_confidence_log_mass(
                    log_A[row:row + 1], 'A(S)',
                    state_evaluator.context(revealed[row]),
                )
            # Only genuine zero success mass terminates a valid trajectory.
            still_alive = alive & ~torch.isneginf(log_A)

            # Multiply by A(S) in log space.
            log_weight = torch.where(
                still_alive,
                log_weight + log_A,
                log_weight,
            )

            alive = still_alive

            # ----------------------------------------------------------
            # q(i | S) = a_i(S) / A(S)
            #
            # Keep the proposal calculation in FP64.
            #
            # For dead rows, use a uniform dummy distribution over
            # unrevealed slots. This avoids NaNs from softmax(-inf,...)
            # without introducing Python/GPU synchronization.
            # ----------------------------------------------------------

            dummy_log_q = torch.where(
                unrevealed,
                torch.zeros_like(log_a),
                torch.full_like(
                    log_a,
                    float("-inf"),
                ),
            )

            proposal_logits = torch.where(
                alive.unsqueeze(-1),
                log_a,
                dummy_log_q,
            )

            q = torch.softmax(
                proposal_logits,
                dim=-1,
            )
            # [bsz, masked_len], FP64

            # ----------------------------------------------------------
            # Sample next successful reveal index
            # ----------------------------------------------------------

            if sample_on_device:
                next_slots = torch.multinomial(
                    q,
                    num_samples=1,
                    replacement=True,
                    generator=rng,
                ).squeeze(-1)
            else:
                next_slots = torch.multinomial(
                    q.detach().cpu(),
                    num_samples=1,
                    replacement=True,
                    generator=rng,
                ).squeeze(-1).to(device)

            next_abs_positions = masked_pos_t[
                next_slots
            ]
            # [bsz]

            if verbose:
                revealed_positions = (next_abs_positions + 1).detach().cpu().tolist()
                for row, position in enumerate(revealed_positions):
                    batch_verbose[row]['reveal_path_indices'].append(int(position))

            next_target_ids = masked_target_row[
                next_slots
            ].unsqueeze(-1)
            # [bsz, 1]

            # ----------------------------------------------------------
            # Successful trajectory conditioning:
            #
            # once i is selected under q(i | S), the successful transition
            # fixes that position to the target z_i.
            # ----------------------------------------------------------

            x.scatter_(
                dim=1,
                index=next_abs_positions.unsqueeze(-1),
                src=next_target_ids,
            )

            revealed.scatter_(
                dim=1,
                index=next_slots.unsqueeze(-1),
                src=reveal_true_base[:bsz],
            )

        # --------------------------------------------------------------
        # Finished trajectory estimates
        # --------------------------------------------------------------

        batch_log_probs = torch.where(
            alive,
            log_weight,
            torch.full_like(
                log_weight,
                float("-inf"),
            ),
        )

        # Stable sum across this Monte Carlo batch.
        batch_log_sum = torch.logsumexp(
            batch_log_probs,
            dim=0,
        )

        # Stable sum across all Monte Carlo batches.
        running_log_sum = torch.logaddexp(
            running_log_sum,
            batch_log_sum,
        )

        num_accumulated += bsz

        if verbose:
            finished_logs = batch_log_probs.detach().cpu().tolist()
            for row, sample_log_estimate in enumerate(finished_logs):
                batch_verbose[row]['sample_log_estimate'] = float(sample_log_estimate)
            verbose_samples.extend(batch_verbose)

        if return_samples:
            sample_log_probabilities.extend(
                batch_log_probs
                .detach()
                .cpu()
                .tolist()
            )

            batch_probabilities = torch.where(
                torch.isfinite(
                    batch_log_probs
                ),
                torch.exp(
                    batch_log_probs
                ),
                torch.zeros_like(
                    batch_log_probs
                ),
            )

            sample_probabilities.extend(
                batch_probabilities
                .detach()
                .cpu()
                .tolist()
            )

    # ==================================================================
    # Arithmetic Monte Carlo mean
    #
    #   (1/K) sum_r W_r
    #
    # in log-space.
    # ==================================================================

    log_average_probability = (
        running_log_sum
        - math.log(num_accumulated)
    ).item()

    if math.isfinite(
        log_average_probability
    ):
        try:
            average_probability = float(
                math.exp(
                    log_average_probability
                )
            )
        except OverflowError:
            average_probability = float("inf")
    else:
        average_probability = 0.0

    # ==================================================================
    # Preserve original output format exactly
    # ==================================================================

    result: Dict[str, object] = {
        "probability": average_probability,
        "log_probability": log_average_probability,
        "num_samples": num_samples,
        "estimation_method":
            "path_sampling_low_confidence_fast_from_partially_masked",
        "decoding_scheme": "full",
        "temperature": temperature,
        "masked_indexes": [
            int(i)
            for i in masked_indexes
        ],
        "num_masked": masked_len,
        "validated_no_ties": False,
        "tie_breaking": "smallest_index_among_max_confidence",
        "model_forward_dtype": "native",
        "model_forward_batch_size": 1,
        "model_eval_mode": True,
        "estimator_dtype_after_logits": "float64",
        "state_cache_enabled": cache_enabled,
        "state_cache_entries": len(state_log_a_cache),
        "state_cache_requests": cache_requests,
        "state_cache_hits": cache_hits,
        "state_cache_misses": cache_misses,
        "state_cache_forward_rows_saved": cache_forward_rows_saved,
        "state_cache_verbose_entries": len(state_verbose_cache),
        "verbose_diagnostic_forward_rows": verbose_diagnostic_forward_rows,
    }

    if return_samples:
        result["sample_probabilities"] = (
            sample_probabilities
        )
        result["sample_log_probabilities"] = (
            sample_log_probabilities
        )
    else:
        result["sample_probabilities"] = None
        result["sample_log_probabilities"] = None

    result['verbose_samples'] = verbose_samples if verbose else None

    return result





@torch.inference_mode()
def _path_sampling_low_confidence_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    temperature: float,
    batch_size: int = 64,
    validate_no_ties: bool = False,
    return_samples: bool = True,
) -> Dict[str, object]:
    """
    Faster unbiased path-sampling estimator for the full-distribution
    low-confidence remasking decoder.

    Main speedups versus the original:
    - only sorts currently masked suffix positions;
    - vectorizes the F_j(c_i) searchsorted computation over j and i;
    - uses fp32 for full-vocab log_softmax/sort/CDF tensors;
    - keeps only log_weight in fp64;
    - samples on device when possible;
    - optionally avoids storing all per-sample probabilities.

    Assumes the helper functions `_model_device` and `_suffix_attention_mask`
    are available, as in your original implementation.
    """

    device = _model_device(model)

    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    if prompt_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError(
            "prompt_tokens and target_tokens must both have shape [1, length]."
        )

    if prompt_tokens.shape[0] != 1 or target_tokens.shape[0] != 1:
        raise ValueError(
            "This estimator is for one fixed z only. "
            "Expected exactly one prompt/target pair."
        )

    suffix_len = target_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]

    if steps != suffix_len:
        raise ValueError(
            "Low-confidence path estimator assumes exactly one revealed token "
            "per step, so steps must equal suffix_len."
        )

    if temperature <= 0:
        raise ValueError(
            "Low-confidence full-distribution estimator requires temperature > 0."
        )

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    # Prefer on-device sampling. Fall back to CPU for less common devices.
    if device.type in {"cuda", "cpu"}:
        rng_device = device
        sample_on_device = True
    else:
        rng_device = torch.device("cpu")
        sample_on_device = False

    rng = torch.Generator(device=rng_device)
    if seed is not None:
        rng.manual_seed(seed)

    prompt_row = prompt_tokens[0]
    target_row = target_tokens[0]

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []

    running_log_sum = torch.tensor(
        float("-inf"),
        dtype=torch.float64,
        device=device,
    )

    num_accumulated = 0

    def compute_log_a_for_batch(
        suffix: torch.Tensor,
        revealed: torch.Tensor,
        alive: torch.Tensor,
        num_masked: int,
    ) -> torch.Tensor:
        """
        Computes log a_i(S) for every row and every suffix index i.

        Returns:
            log_a_full: [bsz, suffix_len]
                        revealed positions and dead rows are -inf.
        """

        bsz = suffix.shape[0]
        masked = ~revealed

        x = torch.cat(
            [
                prompt_row.unsqueeze(0).expand(bsz, -1),
                suffix,
            ],
            dim=1,
        )

        batched_attn = None
        if attn is not None:
            if attn.shape[0] == bsz:
                batched_attn = attn
            else:
                batched_attn = attn.expand(bsz, *attn.shape[1:])

        logits = model(x, attention_mask=batched_attn).logits
        suffix_logits = logits[:, prompt_len:, :]
        vocab_size = suffix_logits.shape[-1]

        # Every row has exactly num_masked masked positions, since all rows
        # reveal exactly one index per estimator step.
        active_idx = masked.nonzero(as_tuple=False)[:, 1].view(
            bsz,
            num_masked,
        )
        # active_idx: [bsz, m]

        active_logits = torch.gather(
            suffix_logits,
            dim=1,
            index=active_idx.unsqueeze(-1).expand(-1, -1, vocab_size),
        )
        # [bsz, m, vocab]

        active_target_ids = target_row[active_idx]
        # [bsz, m]

        # Full-vocab tensors are the expensive part. fp32 is much faster and
        # normally sufficient here. Keep only final weights in fp64.
        scaled_logits = active_logits.float() / float(temperature)

        log_probs = torch.log_softmax(
            scaled_logits,
            dim=-1,
        )
        # [bsz, m, vocab]

        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=active_target_ids.unsqueeze(-1),
        ).squeeze(-1)
        # [bsz, m]

        sorted_log_probs, _ = torch.sort(
            log_probs,
            dim=-1,
        )
        # [bsz, m, vocab]

        log_cdf = torch.logcumsumexp(
            sorted_log_probs,
            dim=-1,
        ).clamp_max_(0.0)
        # [bsz, m, vocab]

        # thresholds[b, j, i] = log c_i,
        # evaluated against position j's sorted distribution.
        thresholds = target_log_probs.unsqueeze(1).expand(
            -1,
            num_masked,
            -1,
        ).contiguous()
        # [bsz, m, m]

        # Strict CDF:
        # F_j(c_i) = P[log p_j(V_j) < log c_i].
        left_idx = torch.searchsorted(
            sorted_log_probs,
            thresholds,
            right=False,
        )
        # [bsz, m, m]

        gather_idx = (left_idx - 1).clamp_min(0)

        log_F = torch.gather(
            log_cdf,
            dim=-1,
            index=gather_idx,
        )
        # [bsz, m, m], dim 1 is j, dim 2 is i.

        log_F = torch.where(
            left_idx > 0,
            log_F,
            torch.full_like(log_F, float("-inf")),
        )

        eye_m = torch.eye(
            num_masked,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

        if validate_no_ties:
            right_idx = torch.searchsorted(
                sorted_log_probs,
                thresholds,
                right=True,
            )

            positive_tie = (
                (right_idx > left_idx)
                & torch.isfinite(thresholds)
                & (~eye_m)
                & alive.view(bsz, 1, 1)
            )

            if bool(positive_tie.any().item()):
                raise ValueError(
                    "Detected a positive-probability confidence tie. "
                    "The current estimator uses the strict no-ties formula "
                    "F_j(c) = P(confidence < c). To estimate the actual "
                    "decoder under ties, implement the decoder's exact "
                    "tie-breaking rule inside a_i(S)."
                )

        # Exclude the j = i term from prod_{j != i} F_j(c_i).
        log_F = log_F.masked_fill(eye_m, 0.0)

        log_product = log_F.sum(dim=1)
        # [bsz, m]

        log_a_active = target_log_probs + log_product
        # [bsz, m]

        log_a_full = torch.full(
            (bsz, suffix_len),
            float("-inf"),
            dtype=log_a_active.dtype,
            device=device,
        )

        log_a_full.scatter_(
            dim=1,
            index=active_idx,
            src=log_a_active,
        )

        log_a_full = torch.where(
            alive.unsqueeze(-1),
            log_a_full,
            torch.full_like(log_a_full, float("-inf")),
        )

        return log_a_full

    for batch_start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - batch_start)

        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        revealed = torch.zeros(
            (bsz, suffix_len),
            dtype=torch.bool,
            device=device,
        )

        log_weight = torch.zeros(
            bsz,
            dtype=torch.float64,
            device=device,
        )

        alive = torch.ones(
            bsz,
            dtype=torch.bool,
            device=device,
        )

        for step in range(suffix_len):
            masked = ~revealed
            num_masked = suffix_len - step

            log_a = compute_log_a_for_batch(
                suffix=suffix,
                revealed=revealed,
                alive=alive,
                num_masked=num_masked,
            )
            # log_a is fp32.

            log_A = torch.logsumexp(
                log_a,
                dim=-1,
            ).to(torch.float64)
            # [bsz]

            still_alive = alive & torch.isfinite(log_A)

            log_weight = torch.where(
                still_alive,
                log_weight + log_A,
                log_weight,
            )

            alive = still_alive

            q = torch.zeros_like(log_a)

            if alive.any():
                q[alive] = torch.softmax(
                    log_a[alive],
                    dim=-1,
                )

            dead = ~alive
            if dead.any():
                dummy_probs = masked[dead].float()
                dummy_probs = dummy_probs / dummy_probs.sum(
                    dim=-1,
                    keepdim=True,
                )
                q[dead] = dummy_probs

            q = torch.where(
                masked,
                q,
                torch.zeros_like(q),
            )

            q_sum = q.sum(dim=-1, keepdim=True)

            q = q / q_sum.clamp_min(
                torch.finfo(q.dtype).tiny,
            )

            if sample_on_device:
                next_indices = torch.multinomial(
                    q.float(),
                    num_samples=1,
                    replacement=True,
                    generator=rng,
                ).squeeze(-1)
            else:
                next_indices = torch.multinomial(
                    q.detach().cpu().float(),
                    num_samples=1,
                    replacement=True,
                    generator=rng,
                ).squeeze(-1).to(device)

            next_target_ids = torch.gather(
                target_row.unsqueeze(0).expand(bsz, -1),
                dim=1,
                index=next_indices.unsqueeze(-1),
            )

            suffix.scatter_(
                dim=1,
                index=next_indices.unsqueeze(-1),
                src=next_target_ids,
            )

            revealed.scatter_(
                dim=1,
                index=next_indices.unsqueeze(-1),
                src=torch.ones(
                    (bsz, 1),
                    dtype=torch.bool,
                    device=device,
                ),
            )

        batch_log_probs = torch.where(
            alive,
            log_weight,
            torch.full_like(log_weight, float("-inf")),
        )

        running_log_sum = torch.logaddexp(
            running_log_sum,
            torch.logsumexp(batch_log_probs, dim=0),
        )

        num_accumulated += bsz

        if return_samples:
            sample_log_probabilities.extend(
                batch_log_probs.detach().cpu().tolist()
            )

            batch_probabilities = torch.where(
                torch.isfinite(batch_log_probs),
                torch.exp(batch_log_probs),
                torch.zeros_like(batch_log_probs),
            )

            sample_probabilities.extend(
                batch_probabilities.detach().cpu().tolist()
            )

    log_average_probability = (
        running_log_sum - math.log(num_accumulated)
    ).item()

    if math.isfinite(log_average_probability):
        try:
            average_probability = float(math.exp(log_average_probability))
        except OverflowError:
            average_probability = float("inf")
    else:
        average_probability = 0.0

    result = {
        "probability": average_probability,
        "log_probability": log_average_probability,
        "num_samples": num_samples,
        "estimation_method": "path_sampling_low_confidence_fast",
        "decoding_scheme": "full",
        "temperature": temperature,
        "validated_no_ties": validate_no_ties,
    }

    if return_samples:
        result["sample_probabilities"] = sample_probabilities
        result["sample_log_probabilities"] = sample_log_probabilities
    else:
        result["sample_probabilities"] = None
        result["sample_log_probabilities"] = None

    return result


@torch.no_grad()
def compute_diffusion_probabilistic_extraction(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor] = None,
    mask_id: int = 126336,
    remasking: str = 'low-confidence',
    estimation_method: str = 'exact',
    num_samples: int = 20,
    seed: Optional[int] = None,
    model_family: str = 'llada',
    decoding_scheme: str = 'full',
    k: int = 40,
    temperature: float = 0.0,
    masked_indexes: Optional[Sequence[int]] = None,
    verbose: bool = False,
    verbose_compact: bool = False,
    verbose_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
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
        String selector. Supported: 'low-confidence' and 'target-token-confidence'.
    estimation_method:
        'exact' (branching over tie-breaks) or 'monte-carlo'.
    num_samples:
        Number of Monte Carlo samples when estimation_method='monte-carlo'.
    seed:
        RNG seed for Monte Carlo.
    """
    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
        raise ValueError('prompt_tokens must have shape (1, a).')
    if target_tokens.ndim != 2 or target_tokens.shape[0] != 1:
        raise ValueError('target_tokens must have shape (1, j).')

    use_variable_count_low_confidence_masks = (
        model_family.lower() == 'llada'
        and estimation_method in {'exact', 'path_sampling'}
        and remasking == 'low-confidence'
        and masked_indexes is not None
    )
    normalized_masked_indexes = validate_masked_indexes(
        masked_indexes,
        expected_count=None if use_variable_count_low_confidence_masks else 50,
    )
    sequence_tokens = None
    if normalized_masked_indexes is not None:
        sequence_tokens = torch.cat([prompt_tokens, target_tokens], dim=1)
        if sequence_tokens.shape[1] != 100:
            raise ValueError(
                '--masked_indexes is only supported for 100-token sequences; '
                f'got total length {sequence_tokens.shape[1]}.'
            )

    model_family = model_family.lower()
    if model_family != 'llada':
        raise ValueError('compute_diffusion_probabilistic_extraction only supports model_family="llada".')
    normalized_decoding_scheme = decoding_scheme.lower()
    if verbose_compact and not verbose:
        raise ValueError('verbose_compact requires verbose=True.')
    if verbose_callback is not None and not verbose:
        raise ValueError('verbose_callback requires verbose=True.')
    if verbose:
        valid_path_verbose = (
            normalized_masked_indexes is not None
            and remasking == 'low-confidence'
            and estimation_method == 'path_sampling'
            and normalized_decoding_scheme == 'full'
            and math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
        valid_mc_verbose = (
            normalized_masked_indexes is not None
            and remasking == 'low-confidence'
            and estimation_method == 'monte-carlo'
            and normalized_decoding_scheme == 'full'
            and math.isfinite(float(temperature))
            and float(temperature) > 0.0
        )
        if not (valid_path_verbose or valid_mc_verbose):
            raise ValueError(
                'verbose diagnostics require partially masked low-confidence '
                'path sampling at temperature 1 or Monte Carlo sampling at '
                'positive temperature, both with full decoding.'
            )
    if normalized_decoding_scheme not in {'full', 'top_k', 'elbo'}:
        raise ValueError("decoding_scheme must be one of {'full', 'top_k', 'ELBO'} for model_family='llada'.")
    if normalized_decoding_scheme == 'top_k' and k <= 0:
        raise ValueError('k must be > 0 when decoding_scheme="top_k".')
    if normalized_decoding_scheme == 'random' and remasking != 'random':
        raise ValueError('decoding_scheme="random" requires remasking="random".')

    if normalized_decoding_scheme == 'elbo':
        if normalized_masked_indexes is None:
            result = _elbo_probability(
                model=model,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                mask_id=mask_id,
            )
        else:
            result = _elbo_probability_from_partially_masked(
                model=model,
                sequence_tokens=sequence_tokens,
                masked_indexes=normalized_masked_indexes,
                mask_id=mask_id,
            )
        return {
            'method': 'elbo',
            'probability': result['probability'],
            'log_probability': result['log_probability'],
            'remasking': remasking,
            'decoding_scheme': 'ELBO',
        }

    _validate_common_args(remasking=remasking, estimation_method=estimation_method)

    if steps <= 0:
        raise ValueError('steps must be > 0.')
    if normalized_masked_indexes is None and target_tokens.shape[1] < steps:
        raise ValueError('steps must be <= target suffix length for this scheduler.')

    if remasking == 'target-token-confidence':
        if normalized_masked_indexes is not None:
            _unsupported_partially_masked_configuration(
                remasking=remasking,
                estimation_method=estimation_method,
                decoding_scheme=normalized_decoding_scheme,
            )
        if estimation_method != 'exact':
            raise ValueError('remasking="target-token-confidence" only supports estimation_method="exact".')
        if temperature <= 0:
            raise ValueError('temperature must be > 0 for remasking="target-token-confidence".')

        result = _exact_probability_target_token_confidence(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            steps=steps,
            attention_mask=attention_mask,
            mask_id=mask_id,
            temperature=temperature,
            decoding_scheme=decoding_scheme,
            k=k,
        )
        return {
            'method': 'exact',
            'probability': result['probability'],
            'log_probability': result['log_probability'],
            'remasking': 'target-token-confidence',
            'temperature': temperature,
            'decoding_scheme': decoding_scheme,
            'k': k if decoding_scheme == 'top_k' else None,
        }

    if remasking == 'random':
        if normalized_decoding_scheme not in {'full', 'top_k'}:
            raise ValueError('remasking="random" requires decoding_scheme in {"full", "top_k"}.')
        if estimation_method != 'path_sampling':
            raise ValueError('remasking="random" only supports estimation_method="path_sampling".')
        if num_samples <= 0:
            raise ValueError('num_samples must be > 0 when estimation_method="path_sampling".')
        if normalized_masked_indexes is None:
            path_sampling_result = _path_sampling_random_probability(
                model=model,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                num_samples=num_samples,
                seed=seed,
                decoding_scheme=normalized_decoding_scheme,
                k=k,
                temperature=temperature,
            )
        else:
            path_sampling_result = _path_sampling_random_probability_from_partially_masked(
                model=model,
                sequence_tokens=sequence_tokens,
                masked_indexes=normalized_masked_indexes,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                num_samples=num_samples,
                seed=seed,
                decoding_scheme=normalized_decoding_scheme,
                k=k,
                temperature=temperature,
            )
        return {
            **path_sampling_result,
            'method': 'path_sampling',
            'probability': path_sampling_result['probability'],
            'sample_probabilities': path_sampling_result['sample_probabilities'],
            'num_samples': path_sampling_result['num_samples'],
            'remasking': 'random',
            'decoding_scheme': normalized_decoding_scheme,
            'k': k if normalized_decoding_scheme == 'top_k' else None,
        }

    if remasking == 'highest-index':
        if estimation_method != 'exact':
            raise ValueError('remasking="highest-index" only supports estimation_method="exact".')
        if normalized_masked_indexes is None:
            result = highest_index_probability(
                model=model,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                decoding_scheme=decoding_scheme,
                k=k,
                temperature=temperature,
            )
        else:
            result = highest_index_probability_from_partially_masked(
                model=model,
                sequence_tokens=sequence_tokens,
                masked_indexes=normalized_masked_indexes,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                decoding_scheme=decoding_scheme,
                k=k,
                temperature=temperature,
            )
        return {
            'method': 'exact',
            'probability': result['probability'],
            'log_probability': result['log_probability'],
            'remasking': 'highest-index',
            'decoding_scheme': decoding_scheme,
            'k': k if decoding_scheme == 'top_k' else None,
            'temperature': temperature if temperature > 0 else None,
        }

    if estimation_method == 'path_sampling':
        if normalized_masked_indexes is not None:
            if remasking == 'low-confidence':
                if normalized_decoding_scheme != 'full':
                    raise ValueError('remasking="low-confidence" with partially masked indexes only supports decoding_scheme="full".')
                if not math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError('estimation_method="path_sampling" with remasking="low-confidence" requires temperature == 1.')
                if num_samples <= 0:
                    raise ValueError('num_samples must be > 0 when estimation_method="path_sampling".')

                path_sampling_result = _path_sampling_low_confidence_probability_fast_from_partially_masked(
                    model=model,
                    sequence_tokens=sequence_tokens,
                    masked_indexes=normalized_masked_indexes,
                    steps=steps,
                    attention_mask=attention_mask,
                    mask_id=mask_id,
                    num_samples=num_samples,
                    seed=seed,
                    temperature=temperature,
                    verbose=verbose,
                    verbose_compact=verbose_compact,
                )
                return {
                    'method': 'path_sampling',
                    'probability': path_sampling_result['probability'],
                    'log_probability': path_sampling_result['log_probability'],
                    'sample_probabilities': path_sampling_result['sample_probabilities'],
                    'sample_log_probabilities': path_sampling_result['sample_log_probabilities'],
                    'verbose_samples': path_sampling_result['verbose_samples'],
                    'num_samples': path_sampling_result['num_samples'],
                    'remasking': 'low-confidence',
                    'decoding_scheme': normalized_decoding_scheme,
                    'k': None,
                }
            _unsupported_partially_masked_configuration(
                remasking=remasking,
                estimation_method=estimation_method,
                decoding_scheme=normalized_decoding_scheme,
            )
        if normalized_decoding_scheme not in {'full', 'top_k'}:
            raise ValueError('estimation_method="path_sampling" with remasking="low-confidence" requires decoding_scheme in {"full", "top_k"}.')
        if not math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('estimation_method="path_sampling" with remasking="low-confidence" requires temperature == 1.')
        if num_samples <= 0:
            raise ValueError('num_samples must be > 0 when estimation_method="path_sampling".')

        path_sampling_result = _path_sampling_low_confidence_probability(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            steps=steps,
            attention_mask=attention_mask,
            mask_id=mask_id,
            num_samples=num_samples,
            seed=seed,
            temperature=temperature
            )
        return {
            'method': 'path_sampling',
            'probability': path_sampling_result['probability'],
            'sample_probabilities': path_sampling_result['sample_probabilities'],
            'num_samples': path_sampling_result['num_samples'],
            'remasking': 'low-confidence',
            'decoding_scheme': normalized_decoding_scheme,
            'k': k if normalized_decoding_scheme == 'top_k' else None,
        }

    if estimation_method == 'exact':
        if normalized_masked_indexes is not None:
            if normalized_decoding_scheme != 'full':
                raise ValueError(
                    'estimation_method="exact" with partially masked '
                    'low-confidence remasking requires decoding_scheme="full".'
                )
            result = _exact_low_confidence_probability_dp_from_partially_masked(
                model=model,
                sequence_tokens=sequence_tokens,
                masked_indexes=normalized_masked_indexes,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                temperature=temperature,
            )
            return {
                **result,
                'method': 'exact',
                'remasking': 'low-confidence',
                'decoding_scheme': 'full',
                'k': None,
            }
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

    if temperature > 0:

        if normalized_masked_indexes is None:
            mc = _monte_carlo_probability_temperature_fast(
                model=model,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                num_samples=num_samples,
                seed=seed,
                temperature=temperature,
                decoding_scheme=decoding_scheme,
                k=k,
            )
        else:
            mc = _monte_carlo_probability_temperature_fast_from_partially_masked(
                model=model,
                sequence_tokens=sequence_tokens,
                masked_indexes=normalized_masked_indexes,
                steps=steps,
                attention_mask=attention_mask,
                mask_id=mask_id,
                num_samples=num_samples,
                seed=seed,
                temperature=temperature,
                decoding_scheme=decoding_scheme,
                k=k,
                verbose=verbose,
                verbose_compact=verbose_compact,
                verbose_callback=verbose_callback,
            )
        
    else:
        if normalized_masked_indexes is not None:
            _unsupported_partially_masked_configuration(
                remasking=remasking,
                estimation_method=estimation_method,
                decoding_scheme=normalized_decoding_scheme,
            )
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
        'verbose_samples': mc.verbose_samples,
        'decoding_scheme': decoding_scheme,
        'k': k if decoding_scheme == 'top_k' else None,
    }
from generate import add_gumbel_noise
def _add_gumbel_noise_with_generator(
    logits: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """
    Same computation as generate.add_gumbel_noise, with optional local RNG.

    When generator is None, importing/calling add_gumbel_noise directly would
    also be fine.
    """
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)

    if generator is None:
        noise = torch.rand_like(
            logits,
            dtype=torch.float64,
        )
    else:
        noise = torch.rand(
            logits.shape,
            dtype=torch.float64,
            device=logits.device,
            generator=generator,
        )

    gumbel_noise = (-torch.log(noise)) ** temperature

    return logits.exp() / gumbel_noise


def _monte_carlo_verbose_step_record(
    step: int,
    sequence_indices: List[int],
    sampled_log_confidence: List[float],
    highest_possible_confidence_indices: List[int],
    sampled_tie_indices: List[int],
    compact: bool,
) -> Dict[str, object]:
    """Build one JSON-safe Monte Carlo verbose step record."""
    record: Dict[str, object] = {
        'step_index': int(step),
        'highest_possible_confidence_indices': [
            int(index) for index in highest_possible_confidence_indices
        ],
    }
    if compact:
        record['sequence_indices'] = [int(index) for index in sequence_indices]
        record['sampled_log_confidence'] = [
            float(value) for value in sampled_log_confidence
        ]
    else:
        record['candidates'] = [
            {
                'sequence_index': int(index),
                'sampled_log_confidence': float(log_confidence),
            }
            for index, log_confidence in zip(
                sequence_indices, sampled_log_confidence
            )
        ]
    if len(sampled_tie_indices) > 1:
        record['sampled_tie_indices'] = [
            int(index) for index in sampled_tie_indices
        ]
    return record


@torch.inference_mode()
@_low_confidence_eval_mode
def _monte_carlo_probability_temperature_fast_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,             # [1, L] full target sequence z
    masked_indexes: list[int],                 # 1-indexed masked positions
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    temperature: float,
    decoding_scheme: str,
    k: int,
    mc_batch_size: int = 16384,
    model_batch_size: int = 64,
    verbose: bool = False,
    verbose_compact: bool = False,
    verbose_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    use_state_cache: bool = True,
) -> MonteCarloResult:
    """
    Naive Monte Carlo estimator for one-token-per-step low-confidence remasking.

    This version is numerically aligned as closely as possible with the
    successful-trajectory sampler.

    At every successful state S:

      1. Candidate tokens are sampled from

             p_i(v | S) = softmax(logits_i / temperature)_v.

         Rather than using a separately normalized torch.multinomial weight
         calculation, sampling uses the same sorted-logit / log-CDF
         representation used by the successful-trajectory estimator.

      2. Candidate confidence is the UNTEMPERED confidence

             c_i(v | S) = softmax(logits_i)_v.

      3. Ranking compares FP64 log-confidences directly:

             log c_i(V_i)
                 = l_i(V_i) - log Z_i.

         No raw-logit confidence threshold is reconstructed.

      4. If multiple positions tie at the maximum sampled confidence, the
         smallest sequence index is selected deterministically. Because
         masked_pos is sorted and active slots preserve that order, this is
         the first tied active slot.

      5. If the chosen position's sampled token is not the target z_i, the
         trajectory immediately fails. Otherwise the target token is
         permanently revealed.

    The returned estimate is hits / num_samples.

    Model forwards always use batch size one and temporary eval mode; original
    module training flags are restored on return or error. model_batch_size is
    retained and validated for API compatibility, but no longer batches model
    forwards. mc_batch_size controls trajectory batching and defaults to 16384
    so typical 2-15k-sample runs evaluate each successful state only once. The
    candidate tensors scale with remaining_positions * mc_batch_size, not with
    vocabulary size. Native active logits use the same per-call 64 MiB CPU LRU
    cache as STS when multiple trajectory batches are needed. The last batch
    reads existing cache entries but skips writes: its states cannot recur.
    use_state_cache=False disables the cache. Custom stochastic/stateful eval
    forwards are unsupported. Changing mc_batch_size changes RNG consumption,
    but not the singleton forwards, distribution arithmetic, or decoder law.

    Verbose diagnostics are observational only: they add no random draws and
    do not participate in winner selection or state transitions.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device)

    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError(
            "sequence_tokens must have shape [1, L], "
            f"got {tuple(sequence_tokens.shape)}"
        )

    seq_len = int(sequence_tokens.shape[1])

    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))

    if not masked_pos:
        raise ValueError(
            "masked_indexes must contain at least one position."
        )

    if len(masked_pos) != len(masked_indexes):
        raise ValueError(
            "masked_indexes must not contain duplicate positions."
        )

    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError(
            f"masked_indexes must be 1-indexed positions in [1, {seq_len}]"
        )

    masked_len = len(masked_pos)

    if steps != masked_len:
        raise ValueError(
            "This naive low-confidence MC implementation reveals exactly one "
            "masked token per step, so steps must equal "
            f"len(masked_indexes)={masked_len}."
        )

    if (
        not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError(
            "temperature must be finite and > 0."
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if mc_batch_size <= 0:
        raise ValueError(
            "mc_batch_size must be positive."
        )

    if model_batch_size <= 0:
        raise ValueError(
            "model_batch_size must be positive."
        )

    if verbose_compact and not verbose:
        raise ValueError("verbose_compact requires verbose=True.")

    if verbose_callback is not None and not verbose:
        raise ValueError("verbose_callback requires verbose=True.")

    if str(decoding_scheme).lower() != "full":
        raise ValueError(
            "This convergence-check sampler matches the cached estimator's "
            "full-distribution decoding only; decoding_scheme must be 'full'."
        )

    # k is intentionally unused under full-distribution sampling.
    _ = k

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

        if attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )

    tau = float(temperature)

    masked_pos_t = torch.tensor(
        masked_pos,
        dtype=torch.long,
        device=device,
    )

    target_row = sequence_tokens[0]
    masked_target_row = target_row[masked_pos_t]

    # ------------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------------

    if device.type in {"cuda", "cpu"}:
        rng_device = device
        sample_on_device = True
    else:
        rng_device = torch.device("cpu")
        sample_on_device = False

    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device=rng_device)
        rng.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # MC
    # ------------------------------------------------------------------

    state_evaluator = _LowConfidenceStateEvaluator(
        model, attention_mask, masked_pos_t, use_cache=use_state_cache,
    )
    hits = 0
    verbose_samples: List[Dict[str, object]] = []

    slot_grid_base = torch.arange(
        masked_len,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for batch_start in range(
        0,
        num_samples,
        mc_batch_size,
    ):
        bsz = min(
            mc_batch_size,
            num_samples - batch_start,
        )

        # A revealed set determines the step, and all occurrences at that step
        # are already grouped below. It can only recur in a LATER batch.
        # Skip device-to-CPU cache copies when no later batch will use them.
        state_evaluator.cache_writes_enabled = batch_start + bsz < num_samples

        # --------------------------------------------------------------
        # Initial state: z outside M, masks inside M.
        # --------------------------------------------------------------

        x = sequence_tokens.expand(
            bsz,
            -1,
        ).clone()

        x[:, masked_pos_t] = mask_id

        revealed = torch.zeros(
            (bsz, masked_len),
            dtype=torch.bool,
            device=device,
        )

        alive = torch.ones(
            bsz,
            dtype=torch.bool,
            device=device,
        )

        batch_verbose: List[Dict[str, object]] = []
        if verbose:
            batch_verbose = [
                {
                    'sample_index': batch_start + row,
                    'is_hit': False,
                    'reveal_path_indices': [],
                    'tie_count': 0,
                    'steps': [],
                }
                for row in range(bsz)
            ]

        # ==============================================================
        # Sequential low-confidence decoding
        # ==============================================================

        for step in range(masked_len):

            alive_rows_t = torch.nonzero(
                alive,
                as_tuple=False,
            ).squeeze(-1)

            if alive_rows_t.numel() == 0:
                break

            m = masked_len - step

            # ----------------------------------------------------------
            # Group identical successful states S.
            #
            # Preserve FIRST-OCCURRENCE order, matching the state grouping
            # convention used by the successful-trajectory cache.
            #
            # Group order controls RNG consumption, so preserve it independently
            # of the canonical singleton model forwards.
            # ----------------------------------------------------------

            alive_revealed = revealed.index_select(
                0,
                alive_rows_t,
            )

            alive_rows_cpu = (
                alive_rows_t
                .detach()
                .cpu()
                .tolist()
            )


            revealed_cpu = (
                alive_revealed
                .detach()
                .cpu()
                .tolist()
            )

            state_to_index: dict[tuple[bool, ...], int] = {}
            rows_per_state: list[list[int]] = []
            representative_rows: list[int] = []

            for row, state_bits in zip(
                alive_rows_cpu,
                revealed_cpu,
            ):
                key = tuple(bool(v) for v in state_bits)

                state_idx = state_to_index.get(key)

                if state_idx is None:
                    state_idx = len(rows_per_state)
                    state_to_index[key] = state_idx

                    rows_per_state.append([])
                    representative_rows.append(
                        int(row)
                    )

                rows_per_state[state_idx].append(
                    int(row)
                )


            # model_batch_size remains an accepted compatibility argument;
            # all model forwards have batch size one, independent of grouping.
            for global_state_idx, representative_row in enumerate(representative_rows):
                trajectory_rows = rows_per_state[global_state_idx]
                group_size = len(trajectory_rows)
                rows_t = torch.tensor(trajectory_rows, dtype=torch.long, device=device)
                state_active_slots = slot_grid_base[0][~revealed[representative_row]]
                distribution = state_evaluator.distribution(
                    x[representative_row], revealed[representative_row], tau,
                )
                state_active_logits = distribution.logits
                log_Z_conf = distribution.log_Z_conf
                sorted_logits = distribution.sorted_logits
                sorted_token_ids = distribution.sorted_token_ids
                log_cdf = distribution.log_cdf
                vocab_size = sorted_logits.shape[-1]

                # Draw one independent candidate at every masked
                # position for every trajectory occupying this state.
                if sample_on_device:
                    uniform_draws = torch.rand(
                        (m, group_size),
                        dtype=torch.float64,
                        device=device,
                        generator=rng,
                    )
                else:
                    uniform_draws = torch.rand(
                        (m, group_size),
                        dtype=torch.float64,
                        device="cpu",
                        generator=rng,
                    ).to(device)

                sampled_sorted_slots = torch.searchsorted(
                    log_cdf,
                    uniform_draws.log(),
                    right=False,
                ).clamp_max(
                    vocab_size - 1
                )
                # [m, group_size]

                sampled_token_ids = torch.gather(
                    sorted_token_ids,
                    dim=-1,
                    index=sampled_sorted_slots,
                )
                # [m, group_size]

                # Pull sampled raw logits from the same sorted FP64
                # representation used to construct the STS CDF.
                sampled_raw_logits = torch.gather(
                    sorted_logits,
                    dim=-1,
                    index=sampled_sorted_slots,
                )
                # [m, group_size]

                # ==================================================
                # Untempered sampled confidence
                #
                #     log c_i(V_i)
                #       =
                #     l_i(V_i) - log Z_i
                # ==================================================

                sampled_log_confidence = (
                    sampled_raw_logits
                    - log_Z_conf.unsqueeze(-1)
                )
                # [m, group_size]

                # ==================================================
                # Winner selection -- direct FP64 log-confidence ranking
                #
                # Compare
                #
                #     log c_i(V_i)
                #       =
                #     l_i(V_i) - log Z_i
                #
                # directly across positions.  This avoids reconstructing
                # raw-logit thresholds and therefore removes the associated
                # subtraction/addition round trip.
                #
                # Ties are resolved deterministically in favor of the
                # smallest sequence index.  masked_pos is sorted and
                # state_active_slots preserves that order, so the first
                # tied local position is exactly the smallest sequence index.
                # ==================================================

                max_sampled_log_confidence = (
                    sampled_log_confidence.max(
                        dim=0,
                        keepdim=True,
                    ).values
                )
                # [1, group_size]

                is_max_confidence = (
                    sampled_log_confidence
                    == max_sampled_log_confidence
                )
                # [m, group_size]

                chosen_local_positions = (
                    is_max_confidence
                    .to(torch.int64)
                    .argmax(dim=0)
                )
                # [group_size]

                # Preserve the historical RNG-consumption pattern of this
                # function.  Previously every step sampled a winner from the
                # (uniform-on-tied-maxima) winner weights, even when the
                # maximum was unique.  The draw is now semantically ignored
                # because tie-breaking is deterministic, but retaining it
                # keeps all later candidate-token draws aligned with the old
                # implementation whenever the preceding state path is the same.
                winner_weights = (
                    is_max_confidence
                    .transpose(0, 1)
                    .to(torch.float64)
                    .contiguous()
                )
                winner_weights.div_(
                    winner_weights.sum(
                        dim=-1,
                        keepdim=True,
                    )
                )

                if sample_on_device:
                    discarded_tie_draw = torch.multinomial(
                        winner_weights,
                        num_samples=1,
                        replacement=True,
                        generator=rng,
                    )
                else:
                    discarded_tie_draw = torch.multinomial(
                        winner_weights.detach().cpu(),
                        num_samples=1,
                        replacement=True,
                        generator=rng,
                    )

                # ==================================================
                # Permanently revealed position
                # ==================================================

                chosen_slots = state_active_slots[
                    chosen_local_positions
                ]

                chosen_abs_positions = masked_pos_t[
                    chosen_slots
                ]

                sampled_token_ids_by_trajectory = (
                    sampled_token_ids
                    .transpose(0, 1)
                )

                chosen_token_ids = torch.gather(
                    sampled_token_ids_by_trajectory,
                    dim=1,
                    index=chosen_local_positions.unsqueeze(
                        -1
                    ),
                ).squeeze(-1)

                chosen_target_ids = masked_target_row[
                    chosen_slots
                ]

                matched = (
                    chosen_token_ids
                    == chosen_target_ids
                )

                # ==================================================
                # Observational verbose diagnostics
                #
                # These reductions and host copies happen only after
                # candidate sampling and winner selection. They consume no
                # randomness and do not feed back into the estimator.
                # ==================================================

                if verbose:
                    maximum_confidence = (
                        state_active_logits.max(dim=-1).values
                        - log_Z_conf
                    )
                    highest_possible = (
                        maximum_confidence
                        == maximum_confidence.max()
                    )

                    sequence_indices = (
                        masked_pos_t[state_active_slots] + 1
                    ).detach().cpu().tolist()
                    highest_possible_indices = [
                        int(index)
                        for index, selected in zip(
                            sequence_indices,
                            highest_possible.detach().cpu().tolist(),
                        )
                        if selected
                    ]
                    sampled_logs_cpu = (
                        sampled_log_confidence.detach().cpu()
                    )
                    sampled_maxima_cpu = (
                        is_max_confidence.detach().cpu()
                    )
                    chosen_indices = (
                        chosen_abs_positions + 1
                    ).detach().cpu().tolist()

                    for column, row in enumerate(trajectory_rows):
                        sampled_tie_indices = [
                            int(index)
                            for index, selected in zip(
                                sequence_indices,
                                sampled_maxima_cpu[:, column].tolist(),
                            )
                            if selected
                        ]
                        step_record = _monte_carlo_verbose_step_record(
                            step=step,
                            sequence_indices=sequence_indices,
                            sampled_log_confidence=(
                                sampled_logs_cpu[:, column].tolist()
                            ),
                            highest_possible_confidence_indices=(
                                highest_possible_indices
                            ),
                            sampled_tie_indices=sampled_tie_indices,
                            compact=verbose_compact,
                        )
                        batch_verbose[row]['steps'].append(step_record)
                        batch_verbose[row]['reveal_path_indices'].append(
                            int(chosen_indices[column])
                        )
                        if len(sampled_tie_indices) > 1:
                            batch_verbose[row]['tie_count'] += 1

                    del maximum_confidence
                    del highest_possible
                    del sampled_logs_cpu
                    del sampled_maxima_cpu

                # ==================================================
                # Failed trajectories terminate immediately
                # ==================================================

                failed_rows_t = rows_t[
                    ~matched
                ]

                if failed_rows_t.numel() > 0:
                    alive[
                        failed_rows_t
                    ] = False

                # ==================================================
                # Successful trajectories reveal target token
                # ==================================================

                successful_rows_t = rows_t[
                    matched
                ]

                if successful_rows_t.numel() > 0:

                    successful_slots = chosen_slots[
                        matched
                    ]

                    successful_abs_positions = (
                        chosen_abs_positions[
                            matched
                        ]
                    )

                    successful_token_ids = (
                        chosen_token_ids[
                            matched
                        ]
                    )

                    x[
                        successful_rows_t,
                        successful_abs_positions,
                    ] = successful_token_ids

                    revealed[
                        successful_rows_t,
                        successful_slots,
                    ] = True

                # --------------------------------------------------
                # Release per-state temporaries
                # --------------------------------------------------

                del state_active_logits
                del sorted_token_ids
                del sorted_logits
                del log_cdf
                del uniform_draws
                del sampled_sorted_slots
                del sampled_token_ids
                del sampled_raw_logits
                del sampled_log_confidence
                del max_sampled_log_confidence
                del is_max_confidence
                del winner_weights
                del discarded_tie_draw
                del chosen_local_positions
                del sampled_token_ids_by_trajectory
                del distribution



        # --------------------------------------------------------------
        # A surviving trajectory reconstructed all masked target tokens.
        # --------------------------------------------------------------

        hits += int(
            alive.sum().item()
        )

        if verbose:
            hit_flags = alive.detach().cpu().tolist()
            for row, is_hit in enumerate(hit_flags):
                batch_verbose[row]['is_hit'] = bool(is_hit)

            if verbose_callback is None:
                verbose_samples.extend(batch_verbose)
            else:
                verbose_callback(batch_verbose)

    # ------------------------------------------------------------------
    # Bernoulli estimate + uncertainty
    # ------------------------------------------------------------------

    estimate, se, wald, wilson = _safe_wald_and_wilson(
        hits,
        num_samples,
    )

    return MonteCarloResult(
        estimate=estimate,
        standard_error=se,
        wald_ci=wald,
        wilson_ci=wilson,
        hits=hits,
        num_samples=num_samples,
        verbose_samples=(
            verbose_samples
            if verbose and verbose_callback is None
            else None
        ),
    )

@torch.inference_mode()
@_low_confidence_eval_mode
def _duel_low_confidence_probability_fast_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, 100], full target sequence z
    masked_indexes: list[int],              # 1-indexed masked positions M
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    temperature: float,
    verbose: bool = False,
    verbose_compact: bool = False,
) -> Dict[str, object]:
    """
    Deterministic DUEL estimator for low-confidence remasking.

    DUEL constructs one deterministic reveal path and then evaluates the
    target-token probability along that path.

    At state S, every still-masked position i is assigned its highest possible
    UNTEMPERED confidence

        h_i(S) = max_v softmax(logits_i)_v.

    For numerical alignment with the low-confidence MC and successful-
    trajectory implementations, positions are ranked directly in FP64
    log-confidence space:

        log h_i(S)
            = max_v logits_i(v)
              - logsumexp_v logits_i(v).

    The position with largest log h_i(S) is revealed next.  Exact ties are
    resolved by taking the smallest sequence index.  The selected position is
    then forced to its target token z_i and the model is evaluated again at the
    next state.  All other unrevealed positions remain masked.

    If the resulting deterministic reveal path is

        pi = (pi_1, ..., pi_|M|),

    DUEL returns the chain-rule target probability along that path:

        p_DUEL(z_M | z_not_M)
            = prod_k p_{pi_k}(z_{pi_k} | S_{k-1}),

    where target-token sampling probabilities use the requested temperature

        p_i(v | S) = softmax(logits_i / temperature)_v.

    Importantly, DUEL does NOT multiply by the probability of selecting the
    reveal path itself.  The reveal path is treated as deterministic.

    The implementation performs path construction and path scoring in one pass:
    the same model forward that determines pi_k also supplies
    p_{pi_k}(z_{pi_k} | S_{k-1}).  For a deterministic model this is
    mathematically identical to first constructing the complete path and then
    replaying that path to score it, while requiring half as many model forwards.

    NUMERICAL STRATEGY
    ------------------
    To match the other low-confidence estimators as closely as possible:
      - temporary deterministic evaluation mode restores all caller settings;
      - the model forward stays in the model's existing/native dtype;
      - autocast is disabled around the forward so an outer autocast context
        cannot silently change that dtype;
      - logits at active masked positions are immediately promoted to FP64;
      - confidence ranking and target-token probabilities are computed in FP64;
      - multiplicative target probabilities are accumulated in log-space;
      - low-confidence ranking is UNTEMPERED, while target-token probability
        uses the requested sampling temperature;
      - exact cross-position confidence ties choose the smallest sequence index.

    Assumptions aligned with the successful-trajectory implementation:
      - sequence_tokens has shape [1, 100];
      - masked_indexes contains exactly 50 valid 1-indexed positions;
      - steps == len(masked_indexes);
      - attention_mask, if provided, has shape [1, 100].

    Notes
    -----
    DUEL itself uses no random sampling and therefore has no num_samples or seed
    argument. Path construction and scoring use one singleton forward per step,
    without unused vocabulary sorting/CDF work or logits caching. Custom models
    with stochastic or mutable evaluation behavior remain unsupported.
    """

    device = _model_device(model)

    # Keep model parameters/buffers in their existing dtype.  As in the other
    # estimators, only active-position logits are widened to FP64 after forward.
    sequence_tokens = sequence_tokens.to(device)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError(
            f"sequence_tokens must have shape [1, 100], "
            f"got {tuple(sequence_tokens.shape)}"
        )

    seq_len = int(sequence_tokens.shape[1])

    if seq_len != 100:
        raise ValueError(
            f"Expected sequence length 100, got {seq_len}"
        )

    # Convert 1-indexed masked positions to sorted unique 0-indexed positions.
    # Keeping them sorted makes torch.argmax's first-maximum behavior exactly
    # implement smallest-sequence-index tie breaking below.
    masked_pos = sorted(
        set(int(i) - 1 for i in masked_indexes)
    )

    if len(masked_pos) != 50:
        raise ValueError(
            f"Expected exactly 50 masked positions out of 100, "
            f"got {len(masked_pos)}"
        )

    if len(masked_pos) != len(masked_indexes):
        raise ValueError(
            "masked_indexes must not contain duplicate positions."
        )

    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError(
            "masked_indexes must be 1-indexed positions in [1, 100]"
        )

    masked_len = len(masked_pos)

    if steps != masked_len:
        raise ValueError(
            "DUEL reveals exactly one masked token per step, so steps must equal "
            f"len(masked_indexes)={masked_len}."
        )

    if (
        not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError(
            "DUEL full-distribution estimator requires finite temperature > 0."
        )

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

        if attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )

    tau = float(temperature)

    masked_pos_t = torch.tensor(
        masked_pos,
        dtype=torch.long,
        device=device,
    )  # [50]

    full_target_row = sequence_tokens[0]
    masked_target_row = full_target_row[masked_pos_t]
    # [50]

    # ------------------------------------------------------------------
    # Initial state: observed positions = z, masked positions = M
    # ------------------------------------------------------------------

    x = sequence_tokens.clone()
    x[:, masked_pos_t] = mask_id

    revealed = torch.zeros(
        masked_len,
        dtype=torch.bool,
        device=device,
    )

    slot_grid = torch.arange(
        masked_len,
        dtype=torch.long,
        device=device,
    )

    # Accumulate log P = sum_k log p_{pi_k}(z_{pi_k} | S_{k-1}).
    log_probability = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )

    reveal_path_indices: List[int] = []
    verbose_steps: List[Dict[str, object]] = []

    # ==================================================================
    # Deterministic DUEL path construction + chain-rule scoring
    # ==================================================================

    for step in range(masked_len):
        m = masked_len - step

        # Active masked slots remain in ascending masked-slot order, and
        # masked_pos itself is ascending absolute sequence-index order.
        active_slots = slot_grid[~revealed]
        # [m]

        active_abs_positions = masked_pos_t[active_slots]
        # [m]

        context = f"step={step}, revealed_indices={sorted(reveal_path_indices)}"
        active_target_ids = masked_target_row[active_slots]
        target_sample_log_probs, highest_log_confidence = _target_probability_state(
            model, x[0], active_abs_positions, active_target_ids, attention_mask,
            tau, context, need_confidence=True,
        )

        # torch.argmax returns the first exact maximum.  Because active_slots
        # and active_abs_positions are ascending, this is precisely the
        # smallest sequence index among tied maxima.
        chosen_local = torch.argmax(
            highest_log_confidence
        )

        chosen_slot = active_slots[chosen_local]
        chosen_abs_position = active_abs_positions[chosen_local]

        # ==============================================================
        # 2. Target-token probability at the SAME state
        #
        # Sampling probability uses temperature, while DUEL ranking above
        # remains untempered.
        # ==============================================================

        chosen_target_log_probability = target_sample_log_probs[
            chosen_local
        ]

        log_probability = (
            log_probability
            + chosen_target_log_probability
        )

        # ==============================================================
        # 3. Force the selected position to its target token
        # ==============================================================

        chosen_target_id = masked_target_row[chosen_slot]

        x[0, chosen_abs_position] = chosen_target_id
        revealed[chosen_slot] = True

        # Public/API-facing reveal path uses 1-indexed absolute positions.
        chosen_abs_position_1idx = int(
            chosen_abs_position.item()
        ) + 1
        reveal_path_indices.append(
            chosen_abs_position_1idx
        )

        if verbose:
            chosen_local_int = int(chosen_local.item())

            if verbose_compact:
                step_record: Dict[str, object] = {
                    "step": step + 1,
                    "revealed_index": chosen_abs_position_1idx,
                    "target_log_probability": float(
                        chosen_target_log_probability.item()
                    ),
                    "highest_log_confidence": float(
                        highest_log_confidence[chosen_local].item()
                    ),
                }
            else:
                active_indices_1idx = (
                    active_abs_positions + 1
                ).detach().cpu().tolist()

                step_record = {
                    "step": step + 1,
                    "revealed_index": chosen_abs_position_1idx,
                    "revealed_masked_slot": int(chosen_slot.item()),
                    "target_token_id": int(chosen_target_id.item()),
                    "target_log_probability": float(
                        chosen_target_log_probability.item()
                    ),
                    "target_probability": float(
                        torch.exp(chosen_target_log_probability).item()
                    ),
                    "highest_log_confidence": float(
                        highest_log_confidence[chosen_local].item()
                    ),
                    "highest_confidence": float(
                        torch.exp(
                            highest_log_confidence[chosen_local]
                        ).item()
                    ),
                    "active_indices": [
                        int(v) for v in active_indices_1idx
                    ],
                    "active_highest_log_confidences": (
                        highest_log_confidence
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "active_target_log_probabilities": (
                        target_sample_log_probs
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "chosen_local_index": chosen_local_int,
                }

            verbose_steps.append(step_record)

        del highest_log_confidence
        del active_target_ids
        del target_sample_log_probs
        del chosen_target_log_probability

    # ==================================================================
    # Final probability
    # ==================================================================

    log_probability_value = float(
        log_probability.item()
    )

    _check_low_confidence_log_mass(
        log_probability, "DUEL path probability", "completed deterministic path",
    )
    probability = 0.0 if log_probability_value == -math.inf else math.exp(log_probability_value)

    # ==================================================================
    # Output -- dictionary style aligned with the successful-trajectory API
    # ==================================================================

    result: Dict[str, object] = {
        "probability": probability,
        "log_probability": log_probability_value,
        "estimation_method": "duel_low_confidence_fast_from_partially_masked",
        "decoding_scheme": "full",
        "temperature": temperature,
        "masked_indexes": [
            int(i) for i in masked_indexes
        ],
        "num_masked": masked_len,
        "reveal_path_indices": reveal_path_indices,
        "tie_breaking": "smallest_index_among_max_confidence",
        "path_construction": "max_possible_untempered_confidence",
        "path_probability": "target_token_chain_rule_only",
        "model_forward_dtype": "native",
        "model_forward_calls": masked_len,
        "model_forward_batch_size": 1,
        "model_eval_mode": True,
        "estimator_dtype_after_logits": "float64",
    }

    result["verbose_steps"] = (
        verbose_steps if verbose else None
    )

    return result


def _logaddexp_scalar(a: float, b: float) -> float:
    """Stable scalar log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a

    hi = max(a, b)
    return hi + math.log1p(math.exp(-abs(a - b)))


def _exact_low_conf_log_a(
    distribution: _LowConfidenceDistribution,  # one canonical state
    active_target_ids: torch.Tensor,      # [1, r]
    temperature: float,
) -> torch.Tensor:
    """
    Compute exact log a_i(S) for every currently masked position i.

    Matches the supplied low-confidence STS / Monte-Carlo semantics:

        candidate:
            V_i ~ softmax(logits_i / temperature)

        ranking confidence:
            c_i(v) = softmax(logits_i)_v

        winner:
            largest sampled confidence

        tie:
            smallest absolute sequence index

    active positions must be ordered by increasing absolute sequence index.
    """
    # Consume the same singleton distribution as STS. Do not recompute
    # normalizers or CDFs in a batched tensor with a different reduction shape.
    active_logits = distribution.logits.unsqueeze(0)
    device = active_logits.device
    B, r, V = active_logits.shape
    tau = float(temperature)
    log_Z_conf = distribution.log_Z_conf.unsqueeze(0)
    log_Z_sample = distribution.log_Z_sample.unsqueeze(0)

    # Raw logit of target token z_i.
    target_raw_logits = torch.gather(
        active_logits,
        dim=-1,
        index=active_target_ids.unsqueeze(-1),
    ).squeeze(-1)

    # log p_i(z_i | S), where sampling uses temperature.
    target_sample_log_probs = (
        target_raw_logits / tau
        - log_Z_sample
    )

    # log c_i(z_i | S), where confidence is UNTEMPERED.
    target_conf_log_probs = (
        target_raw_logits
        - log_Z_conf
    )

    # No competitors on the final step.
    if r == 1:
        return target_sample_log_probs

    sorted_logits = distribution.sorted_logits.unsqueeze(0)
    log_cdf = distribution.log_cdf.unsqueeze(0)
    del active_logits

    # --------------------------------------------------------------
    # Compare directly in FP64 log-confidence space.
    # --------------------------------------------------------------
    sorted_log_confidence = (
        sorted_logits
        - log_Z_conf.unsqueeze(-1)
    )

    # Shape convention:
    #
    #   axis 1 = competitor j
    #   axis 2 = proposed winner i
    #
    # [B, competitor_j, proposed_i]
    target_conf_values = (
        target_conf_log_probs
        .unsqueeze(1)
        .expand(-1, r, -1)
        .contiguous()
    )

    # first competitor token with:
    #
    #   confidence >= c_i*
    #   confidence >  c_i*
    #
    # respectively.
    left_idx = torch.searchsorted(
        sorted_log_confidence,
        target_conf_values,
        right=False,
    )

    right_idx = torch.searchsorted(
        sorted_log_confidence,
        target_conf_values,
        right=True,
    )

    # --------------------------------------------------------------
    # L_{j,i}
    #
    # P[c_j(V_j) < c_i*]
    # --------------------------------------------------------------
    left_gather_idx = (
        left_idx - 1
    ).clamp(
        min=0,
        max=V - 1,
    )

    log_L = torch.gather(
        log_cdf,
        dim=-1,
        index=left_gather_idx,
    )

    log_L.masked_fill_(
        left_idx == 0,
        -math.inf,
    )

    log_L.masked_fill_(
        left_idx == V,
        0.0,
    )

    # --------------------------------------------------------------
    # LE_{j,i}
    #
    # P[c_j(V_j) <= c_i*]
    # --------------------------------------------------------------
    right_gather_idx = (
        right_idx - 1
    ).clamp(
        min=0,
        max=V - 1,
    )

    log_LE = torch.gather(
        log_cdf,
        dim=-1,
        index=right_gather_idx,
    )

    log_LE.masked_fill_(
        right_idx == 0,
        -math.inf,
    )

    log_LE.masked_fill_(
        right_idx == V,
        0.0,
    )

    # --------------------------------------------------------------
    # Exact deterministic smallest-index tie rule.
    #
    # active positions are ordered by increasing absolute sequence
    # index.
    #
    # For proposed winner i:
    #
    #   competitor j < i:
    #       j would win an exact tie,
    #       therefore j MUST be strictly below i.
    #
    #   competitor j > i:
    #       i wins an exact tie,
    #       therefore j may be <= i.
    #
    # This is exactly the STS tie correction.
    # --------------------------------------------------------------
    proposed_i = torch.arange(
        r,
        dtype=torch.long,
        device=device,
    ).view(1, r)

    log_win_mass = torch.zeros(
        (B, r),
        dtype=torch.float64,
        device=device,
    )

    # Select all strict/non-strict factors together, avoiding several small
    # GPU kernels per competitor. Keep additions in exactly STS's order.
    competitor_j = proposed_i.transpose(0, 1)
    competitor_factors = torch.where(
        competitor_j < proposed_i,
        log_L,
        torch.where(competitor_j > proposed_i, log_LE, 0.0),
    )
    for competitor_j in range(r):
        log_win_mass = log_win_mass + competitor_factors[:, competitor_j, :]

    # --------------------------------------------------------------
    # a_i(S)
    #
    # = probability i samples target
    #   * probability i wins confidence competition.
    # --------------------------------------------------------------
    return (
        target_sample_log_probs
        + log_win_mass
    )


@torch.inference_mode()
@_low_confidence_eval_mode
def _exact_low_confidence_probability_dp_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, L], complete target z
    masked_indexes: list[int],              # 1-indexed masked positions M
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    temperature: float,
    state_batch_size: int = 64,
    max_masked: int = MAX_EXACT_LOW_CONFIDENCE_MASKED,
    evaluate_all_states: bool = False,
) -> Dict[str, object]:
    """
    Exact extraction probability under one-token-per-step
    low-confidence remasking.

    Uses dynamic programming over subsets S of correctly revealed
    target positions.

    Forward recurrence:

        DP[empty] = 1

        DP[S U {i}]
            +=
        DP[S] * a_i(S)

    where a_i(S) is the exact probability that, from S,

      1. position i samples target token z_i, and
      2. position i wins the sampled-confidence competition.

    The answer is DP[M].

    Semantics match the supplied low-confidence MC and STS:

      * sampling:
            softmax(logits / temperature)

      * ranking:
            untempered softmax(logits)

      * one reveal per step

      * maximum sampled confidence wins

      * exact confidence ties:
            smallest absolute sequence index wins

      * singleton model forwards through the shared STS state evaluator

      * temporary deterministic evaluation mode, with autocast disabled

      * native model-forward dtype

      * FP64 probability calculations after logits

      * probability products/sums accumulated in log-space

    Complexity for m masked positions:

        states = 2^m

    and every reachable nonterminal state is evaluated exactly once. Only
    states with exactly zero incoming probability are skipped; no small positive
    probability is discarded. evaluate_all_states=True also evaluates unreachable
    states for auditing. state_batch_size only chunks state scheduling; model
    forward batch size is always one.
    The caller's module modes and deterministic backend settings are restored.

    log_probability is the authoritative natural-log result. FP64 and log-space
    accumulation support probabilities down to 1e-100 without floors or pruning.

    For m=12:
        at most 4095 model-state evaluations

    instead of:
        12! = 479,001,600 reveal orders.
    """
    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device)

    # ==============================================================
    # Validation
    # ==============================================================

    if (
        sequence_tokens.ndim != 2
        or sequence_tokens.shape[0] != 1
    ):
        raise ValueError(
            "sequence_tokens must have shape [1, L]."
        )

    seq_len = int(
        sequence_tokens.shape[1]
    )

    raw_masked_pos = [
        int(i) - 1
        for i in masked_indexes
    ]

    if not raw_masked_pos:
        raise ValueError(
            "masked_indexes must contain at least one position."
        )

    if (
        len(raw_masked_pos)
        != len(set(raw_masked_pos))
    ):
        raise ValueError(
            "masked_indexes must not contain duplicate positions."
        )

    # This matches MC / STS and is important for the tie rule:
    # local active-slot order == absolute sequence-index order.
    masked_pos = sorted(
        raw_masked_pos
    )

    if any(
        pos < 0 or pos >= seq_len
        for pos in masked_pos
    ):
        raise ValueError(
            "masked_indexes must be 1-indexed positions "
            f"in [1, {seq_len}]."
        )

    masked_len = len(
        masked_pos
    )

    if steps != masked_len:
        raise ValueError(
            "Exact low-confidence DP reveals exactly one "
            "masked token per step, so steps must equal "
            f"len(masked_indexes)={masked_len}."
        )

    if masked_len > max_masked:
        raise ValueError(
            f"masked_len={masked_len} exceeds "
            f"max_masked={max_masked}; exact DP is exponential."
        )

    if (
        not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError(
            "temperature must be finite and > 0."
        )

    if state_batch_size <= 0:
        raise ValueError(
            "state_batch_size must be positive."
        )

    if attention_mask is not None:
        attention_mask = attention_mask.to(
            device
        )

        if attention_mask.shape != (
            1,
            seq_len,
        ):
            raise ValueError(
                "attention_mask must have shape "
                f"[1, {seq_len}]."
            )

    tau = float(
        temperature
    )

    masked_pos_t = torch.tensor(
        masked_pos,
        dtype=torch.long,
        device=device,
    )

    masked_target_row = sequence_tokens[
        0,
        masked_pos_t,
    ]

    # ==============================================================
    # Subset representation
    # ==============================================================

    # Bit i == target position i in M has already been revealed.
    num_states = (
        1 << masked_len
    )

    full_state = (
        num_states - 1
    )

    bit_values = torch.bitwise_left_shift(
        torch.ones(
            masked_len,
            dtype=torch.int64,
            device=device,
        ),
        torch.arange(
            masked_len,
            dtype=torch.int64,
            device=device,
        ),
    )

    # Every state is evaluated once, so caching native logits has no benefit.
    state_evaluator = _LowConfidenceStateEvaluator(
        model, attention_mask, masked_pos_t, use_cache=False,
    )

    # Evaluate states on demand in numeric topological order. This preserves
    # the original CPU logaddexp accumulation order, while avoiding forwards
    # for states that cannot be reached by any successful reveal path.
    log_dp = [-math.inf] * num_states
    log_dp[0] = 0.0
    for batch_start in range(0, full_state, state_batch_size):
        for state in range(batch_start, min(batch_start + state_batch_size, full_state)):
            base = log_dp[state]
            if base == -math.inf and not evaluate_all_states:
                continue

            revealed = torch.bitwise_and(state, bit_values) != 0
            x_row = sequence_tokens[0].clone()
            x_row[masked_pos_t] = torch.where(
                revealed, masked_target_row,
                torch.full_like(masked_target_row, int(mask_id)),
            )
            distribution = state_evaluator.distribution(x_row, revealed, tau)
            log_a = _exact_low_conf_log_a(
                distribution=distribution,
                active_target_ids=masked_target_row[~revealed].unsqueeze(0),
                temperature=tau,
            )[0]
            # Construct diagnostic context from the CPU subset instead of
            # copying the revealed-position tensor back from the GPU again.
            positions = [masked_pos[i] + 1 for i in range(masked_len) if state & (1 << i)]
            context = f"step={len(positions)}, revealed_indices={positions}"
            _check_low_confidence_log_mass(
                log_a, "successful transition masses", context,
            )
            # Match STS's full masked-slot reduction, including revealed -inf.
            log_a_full = torch.full(
                (masked_len,), -math.inf, dtype=torch.float64, device=device,
            )
            log_a_full[~revealed] = log_a
            _check_low_confidence_log_mass(
                torch.logsumexp(log_a_full, dim=-1), "A(S)", context,
            )
            # The row is tiny (at most max_masked doubles). Moving it to CPU
            # enables exact reachability decisions and avoids tiny GPU DP kernels.
            log_a_cpu = log_a_full.detach().cpu().tolist()
            del distribution, log_a, log_a_full, x_row, revealed

            if base == -math.inf:
                continue  # Audit mode evaluated this state, but it has no mass.
            remaining = full_state ^ state
            while remaining:
                bit = remaining & -remaining
                slot = bit.bit_length() - 1
                log_transition = log_a_cpu[slot]
                if log_transition != -math.inf:
                    next_state = state | bit
                    log_dp[next_state] = _logaddexp_scalar(
                        log_dp[next_state], base + log_transition,
                    )
                remaining ^= bit

    log_probability = float(
        log_dp[
            full_state
        ]
    )

    _check_low_confidence_log_mass(
        torch.tensor(log_probability, dtype=torch.float64),
        "final DP probability", "full revealed subset",
    )
    # Only -inf denotes a genuine zero. NaN/+inf must never masquerade as zero.
    probability = 0.0 if log_probability == -math.inf else math.exp(log_probability)

    return {
        "probability":
            float(probability),

        "log_probability":
            log_probability,

        "estimation_method":
            "exact_low_confidence_subset_dp",

        "decoding_scheme":
            "full",

        "temperature":
            tau,

        "masked_indexes": [
            int(i)
            for i in masked_indexes
        ],

        "num_masked":
            masked_len,

        "num_dp_states":
            num_states,

        "num_nonterminal_states":
            full_state,

        "state_batch_size":
            state_batch_size,

        "model_forward_calls":
            state_evaluator.forward_rows,

        "num_evaluated_states":
            state_evaluator.forward_rows,

        "num_skipped_unreachable_states":
            full_state - state_evaluator.forward_rows,

        "evaluate_all_states":
            bool(evaluate_all_states),

        "model_forward_batch_size":
            1,

        "model_eval_mode":
            True,

        "tie_breaking":
            "smallest_index_among_max_confidence",

        "model_forward_dtype":
            "native",

        "probability_dtype_after_logits":
            "float64",
    }
@torch.no_grad()
def compute_autoregressive_probabilistic_extraction(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    model_family: str = 'llama',
    decoding_scheme: str = 'top_k',
    k: int = 40,
    temperature: float = 0.0,
    return_token_details: bool = False,
):
    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
        raise ValueError('prompt_tokens must have shape (1, a).')
    if target_tokens.ndim != 2 or target_tokens.shape[0] != 1:
        raise ValueError('target_tokens must have shape (1, j).')

    model_family = model_family.lower()
    if model_family not in AUTOREGRESSIVE_MODEL_FAMILIES:
        raise ValueError('compute_autoregressive_probabilistic_extraction only supports model_family in {"llama", "llama2", "olmo", "mistral"}.')

    result = _autoregressive_probability(
        model=model,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        attention_mask=attention_mask,
        decoding_scheme=decoding_scheme,
        k=k,
        temperature=temperature,
        return_token_details=return_token_details,
    )
    result['model_family'] = model_family
    return result


@torch.no_grad()
def compute_probabilistic_extraction(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor] = None,
    mask_id: int = 126336,
    remasking: str = 'low-confidence',
    estimation_method: str = 'exact',
    num_samples: int = 20,
    seed: Optional[int] = None,
    model_family: str = 'llada',
    decoding_scheme: str = 'auto',
    k: int = 40,
    temperature: float = 0.0,
    return_token_details: bool = False,
    masked_indexes: Optional[Sequence[int]] = None,
):
    model_family = model_family.lower()
    if model_family in AUTOREGRESSIVE_MODEL_FAMILIES:
        if masked_indexes is not None:
            raise ValueError('--masked_indexes is only supported when model_family="llada".')
        ar_decoding_scheme = 'top_k' if decoding_scheme == 'auto' else decoding_scheme
        return compute_autoregressive_probabilistic_extraction(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            attention_mask=attention_mask,
            model_family=model_family,
            decoding_scheme=ar_decoding_scheme,
            k=k,
            temperature=temperature,
            return_token_details=return_token_details,
        )
    if model_family == 'llada':
        diffusion_decoding_scheme = 'full' if decoding_scheme == 'auto' else decoding_scheme
        return compute_diffusion_probabilistic_extraction(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            steps=steps,
            attention_mask=attention_mask,
            mask_id=mask_id,
            remasking=remasking,
            estimation_method=estimation_method,
            num_samples=num_samples,
            seed=seed,
            model_family=model_family,
            decoding_scheme=diffusion_decoding_scheme,
            k=k,
            temperature=temperature,
            masked_indexes=masked_indexes,
        )
    raise ValueError("model_family must be one of {'llada', 'llama', 'llama2', 'olmo', 'mistral'}")
