import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from get_log_likelihood import get_log_likelihood

AUTOREGRESSIVE_MODEL_FAMILIES = {'llama', 'llama2'}


@dataclass
class MonteCarloResult:
    estimate: float
    standard_error: float
    wald_ci: Tuple[float, float]
    wilson_ci: Tuple[float, float]
    hits: int
    num_samples: int


def _validate_common_args(remasking: str, estimation_method: str) -> None:
    allowed_remasking = {'low-confidence', 'target-token-confidence', 'random'}
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
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    decoding_scheme: str,
    k: int,
    sample_batch_size: int = 256,
) -> MonteCarloResult:
    """
    Tie-free Rao-Blackwellized estimator for p_z under:
      - LLaDA low-confidence remasking
      - temperature = 1
      - one token revealed per step

    The estimator samples reveal orders only. It analytically integrates out
    the token proposals of losing positions.

    Supports:
      decoding_scheme == "top_k"   (recommended)
      decoding_scheme == "full"    (exact tie-free full-vocab version, but slow)

    Assumptions:
      - prompt_tokens shape: [1, P]
      - target_tokens shape: [1, S]
      - steps == S
      - custom helpers/classes already exist and work:
          _model_device
          _suffix_attention_mask
          MonteCarloResult
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if decoding_scheme not in {"top_k", "full"}:
        raise ValueError("decoding_scheme must be 'top_k' or 'full'")

    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device, non_blocking=True)
    target_tokens = target_tokens.to(device, non_blocking=True)

    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
        raise ValueError("prompt_tokens must have shape [1, P]")
    if target_tokens.ndim != 2 or target_tokens.shape[0] != 1:
        raise ValueError("target_tokens must have shape [1, S]")

    prefix_len = prompt_tokens.shape[1]
    suffix_len = target_tokens.shape[1]

    if steps != suffix_len:
        raise ValueError(
            f"This implementation assumes one token revealed per step, "
            f"so steps must equal suffix_len. Got steps={steps}, suffix_len={suffix_len}."
        )

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)
    target_suffix = target_tokens[0]  # [S]

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    all_weights = []

    for start in range(0, num_samples, sample_batch_size):
        bsz = min(sample_batch_size, num_samples - start)

        # Current partially revealed suffix for each trajectory.
        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # Importance weight of each trajectory.
        weights = torch.ones(bsz, dtype=torch.float64, device=device)

        prompt_batch = prompt_tokens.expand(bsz, -1)

        for _step_idx in range(steps):
            alive = weights > 0
            if not alive.any():
                break

            alive_idx = alive.nonzero(as_tuple=False).squeeze(-1)
            cur_bsz = alive_idx.numel()

            cur_suffix = suffix.index_select(0, alive_idx)  # [B,S]

            x = torch.empty(
                (cur_bsz, prefix_len + suffix_len),
                dtype=prompt_tokens.dtype,
                device=device,
            )
            x[:, :prefix_len] = prompt_batch[:cur_bsz]
            x[:, prefix_len:] = cur_suffix

            logits = model(x, attention_mask=attn).logits[:, prefix_len:, :]  # [B,S,V]
            full_probs = logits.softmax(dim=-1)                                # [B,S,V]
            masked = (cur_suffix == mask_id)                                   # [B,S]

            # Threshold t_i = p_i(z_i), the confidence if position i samples the correct token.
            target_prob = full_probs.gather(
                dim=-1,
                index=target_suffix.view(1, suffix_len, 1).expand(cur_bsz, -1, -1),
            ).squeeze(-1)  # [B,S]

            # Compute q_i under the tie-free approximation.
            if decoding_scheme == "top_k":
                q = _compute_q_topk_tie_free(
                    logits=logits,
                    full_probs=full_probs,
                    target_suffix=target_suffix,
                    target_prob=target_prob,
                    masked=masked,
                    k=k,
                )  # [B,S], zero on invalid/non-masked positions
            else:
                q = _compute_q_full_tie_free(
                    full_probs=full_probs,
                    target_suffix=target_suffix,
                    target_prob=target_prob,
                    masked=masked,
                )  # [B,S], exact tie-free full-vocab version

            step_survival = q.sum(dim=-1)  # [B]

            # Update importance weights: multiply by probability of surviving this reveal step.
            new_weights = weights.index_select(0, alive_idx)
            new_weights = new_weights * step_survival.to(torch.float64)
            weights[alive_idx] = new_weights

            # If step_survival == 0, those trajectories are dead.
            survive = step_survival > 0
            if not survive.any():
                break

            surv_idx_local = survive.nonzero(as_tuple=False).squeeze(-1)
            surv_idx_global = alive_idx.index_select(0, surv_idx_local)

            q_surv = q.index_select(0, surv_idx_local)
            step_survival_surv = step_survival.index_select(0, surv_idx_local)

            # Sample next revealed index from q / sum(q).
            reveal_dist = q_surv / step_survival_surv.unsqueeze(-1)  # [B,S]
            next_pos = torch.multinomial(reveal_dist, num_samples=1, generator=rng).squeeze(-1)  # [B]

            # Reveal the correct target token at that sampled position.
            row_idx = torch.arange(next_pos.shape[0], device=device)
            suffix[surv_idx_global[row_idx], next_pos] = target_suffix[next_pos]

            # Zero out dead trajectories explicitly.
            dead_idx_local = (~survive).nonzero(as_tuple=False).squeeze(-1)
            if dead_idx_local.numel() > 0:
                dead_idx_global = alive_idx.index_select(0, dead_idx_local)
                weights[dead_idx_global] = 0.0

        all_weights.append(weights)

    all_weights = torch.cat(all_weights, dim=0)  # [num_samples]
    estimate = all_weights.mean().item()

    if num_samples > 1:
        se = all_weights.std(unbiased=True).item() / math.sqrt(num_samples)
    else:
        se = 0.0

    z = 1.96
    wald_lo = max(0.0, estimate - z * se)
    wald_hi = min(1.0, estimate + z * se)

    # Wilson is not really the right interval for weighted estimators.
    # Reuse the Wald interval slot for compatibility.
    return MonteCarloResult(
        estimate=estimate,
        standard_error=se,
        wald_ci=(wald_lo, wald_hi),
        wilson_ci=(wald_lo, wald_hi),
        hits=-1,
        num_samples=num_samples,
    )


def _compute_q_topk_tie_free(
    logits: torch.Tensor,          # [B,S,V]
    full_probs: torch.Tensor,      # [B,S,V]
    target_suffix: torch.Tensor,   # [S]
    target_prob: torch.Tensor,     # [B,S], where target_prob[:,i] = p_i(z_i)
    masked: torch.Tensor,          # [B,S]
    k: int,
) -> torch.Tensor:
    """
    Tie-free top-k formula:

      q_i = \tilde p_i(z_i) * prod_{j != i} F_j(t_i)

    where
      t_i = p_i(z_i),
      \tilde p_j is the top-k truncated/renormalized proposal distribution,
      F_j(t) = sum_{v in K_j : p_j(v) < t} \tilde p_j(v)

    Returns q of shape [B,S].
    """
    B, S, V = logits.shape
    top_k = min(k, V)

    top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)   # [B,S,K], [B,S,K]
    proposal_probs = top_vals.softmax(dim=-1)                 # [B,S,K]

    # Full-softmax confidence values of tokens in top-k support.
    support_conf = full_probs.gather(dim=-1, index=top_idx)   # [B,S,K]

    # Whether the target token is in the top-k support.
    target_expanded = target_suffix.view(1, S, 1).expand(B, -1, -1)  # [B,S,1]
    target_in_support = (top_idx == target_expanded)                  # [B,S,K]
    has_target = target_in_support.any(dim=-1)                        # [B,S]

    # Proposal probability of the correct token under top-k sampling.
    proposal_p_correct = torch.where(
        has_target,
        (proposal_probs * target_in_support.to(proposal_probs.dtype)).sum(dim=-1),
        torch.zeros_like(target_prob),
    )  # [B,S]

    # F[j, i] = probability that competitor position j samples a token with confidence < t_i
    # Build as tensor over candidate i and competitor j:
    #
    # thresholds:      [B, i, 1, 1]
    # support_conf:    [B, 1, j, K]
    # proposal_probs:  [B, 1, j, K]
    thresholds = target_prob.unsqueeze(2).unsqueeze(3)                 # [B,S,1,1]
    support_conf_exp = support_conf.unsqueeze(1)                       # [B,1,S,K]
    proposal_probs_exp = proposal_probs.unsqueeze(1)                   # [B,1,S,K]

    competitor_cdf = (
        proposal_probs_exp * (support_conf_exp < thresholds).to(proposal_probs.dtype)
    ).sum(dim=-1)  # [B, candidate_i, competitor_j]

    # Exclude unmasked competitors from the product: factor = 1
    competitor_mask = masked.unsqueeze(1)                              # [B,1,S]
    competitor_cdf = torch.where(
        competitor_mask,
        competitor_cdf,
        torch.ones_like(competitor_cdf),
    )

    # Exclude j == i from the product: factor = 1
    eye = torch.eye(S, dtype=torch.bool, device=logits.device).unsqueeze(0)  # [1,S,S]
    competitor_cdf = torch.where(
        eye,
        torch.ones_like(competitor_cdf),
        competitor_cdf,
    )

    product_term = competitor_cdf.prod(dim=-1)  # [B,S]

    q = proposal_p_correct * product_term

    # Candidate i itself must be masked.
    q = torch.where(masked, q, torch.zeros_like(q))
    return q


def _compute_q_full_tie_free(
    full_probs: torch.Tensor,      # [B,S,V]
    target_suffix: torch.Tensor,   # [S]
    target_prob: torch.Tensor,     # [B,S]
    masked: torch.Tensor,          # [B,S]
) -> torch.Tensor:
    """
    Exact tie-free full-vocab formula:

      q_i = p_i(z_i) * prod_{j != i} F_j(t_i)

    where
      t_i = p_i(z_i),
      F_j(t) = sum_{v : p_j(v) < t} p_j(v)

    This is exact for the tie-free full-vocab case, but usually much slower
    and more memory-intensive than top-k.
    """
    B, S, V = full_probs.shape

    proposal_p_correct = target_prob  # [B,S], because proposal law == full softmax at temperature 1

    thresholds = target_prob.unsqueeze(2).unsqueeze(3)                 # [B,S,1,1]
    probs_exp = full_probs.unsqueeze(1)                                # [B,1,S,V]

    competitor_cdf = (
        probs_exp * (probs_exp < thresholds).to(full_probs.dtype)
    ).sum(dim=-1)  # [B, candidate_i, competitor_j]

    competitor_mask = masked.unsqueeze(1)                              # [B,1,S]
    competitor_cdf = torch.where(
        competitor_mask,
        competitor_cdf,
        torch.ones_like(competitor_cdf),
    )

    eye = torch.eye(S, dtype=torch.bool, device=full_probs.device).unsqueeze(0)
    competitor_cdf = torch.where(
        eye,
        torch.ones_like(competitor_cdf),
        competitor_cdf,
    )

    product_term = competitor_cdf.prod(dim=-1)  # [B,S]
    q = proposal_p_correct * product_term
    q = torch.where(masked, q, torch.zeros_like(q))
    return q

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
    """
    Fast Monte Carlo estimator for LLaDA low-confidence remasking with:
      - temperature > 0
      - one token revealed per step
      - optional top-k token sampling

    Assumes custom helpers/classes already exist and work:
      - _model_device
      - _suffix_attention_mask
      - _safe_wald_and_wilson
      - MonteCarloResult

    This implementation is mathematically equivalent to the original for the
    stated setting, but vectorized across Monte Carlo samples and positions.
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

    # Hoist constants once.
    target_suffix = target_tokens[0]                            # [S]
    hits = 0
    vocab_arange_cache = None

    # Process independent trajectories in GPU batches.
    for start in range(0, num_samples, mc_batch_size):
        bsz = min(mc_batch_size, num_samples - start)

        # suffix: [B, S]
        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # alive[b] == True means trajectory b is still consistent with target.
        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        # Pre-expand prefix once per batch.
        # prompt_batch: [B, P]
        prompt_batch = prompt_tokens.expand(bsz, -1)

        # Build model input buffer once and update suffix in-place each step.
        # x: [B, P+S]
        x = torch.empty(
            (bsz, prefix_len + suffix_len),
            dtype=prompt_tokens.dtype,
            device=device,
        )
        x[:, :prefix_len] = prompt_batch

        for _step_idx in range(steps):
            # If all trajectories are dead, stop early.
            if not alive.any():
                break

            # Update suffix portion in-place.
            x[:, prefix_len:] = suffix

            # Forward one big batch.
            logits = model(x, attention_mask=attn).logits[:, prefix_len:, :]   # [B, S, V]
            # Unscaled probs are used for low-confidence ranking in the original code.
            probs = logits.softmax(dim=-1)                                      # [B, S, V]

            # Current masked positions.
            masked = (suffix == mask_id) & alive[:, None]                       # [B, S]

            # Sample a token proposal for every (alive, masked) position.
            scaled_logits = logits / temperature                                # [B, S, V]

            if decoding_scheme == "top_k":
                top_k = min(k, scaled_logits.shape[-1])
                top_vals, top_idx = torch.topk(scaled_logits, k=top_k, dim=-1)  # [B,S,K], [B,S,K]
                top_probs = top_vals.softmax(dim=-1)                             # [B,S,K]

                sampled_local = torch.multinomial(
                    top_probs.reshape(-1, top_k),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, suffix_len)                                       # [B,S]

                sampled_tokens = top_idx.gather(
                    dim=-1,
                    index=sampled_local.unsqueeze(-1),
                ).squeeze(-1)                                                    # [B,S]
            else:
                vocab_size = scaled_logits.shape[-1]
                sampled_tokens = torch.multinomial(
                    scaled_logits.softmax(dim=-1).reshape(-1, vocab_size),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, suffix_len)                                       # [B,S]

            # Low-confidence remasking score:
            # confidence = p_model(sampled_token | current masked context)
            chosen_prob = probs.gather(
                dim=-1,
                index=sampled_tokens.unsqueeze(-1),
            ).squeeze(-1)                                                        # [B,S]

            # Only masked positions are eligible; everything else gets -inf.
            neg_inf = torch.full_like(chosen_prob, float("-inf"))
            confidence = torch.where(masked, chosen_prob, neg_inf)               # [B,S]

            # One token revealed per step => select the single most confident masked position.
            selected_pos = confidence.argmax(dim=-1)                             # [B]

            batch_idx = torch.arange(bsz, device=device)
            selected_token = sampled_tokens[batch_idx, selected_pos]             # [B]
            selected_target = target_suffix[selected_pos]                        # [B]

            # Only alive trajectories participate.
            active = alive

            # Commit the chosen token for active trajectories.
            suffix[batch_idx[active], selected_pos[active]] = selected_token[active]

            # Kill trajectories whose committed token mismatches the target.
            mismatch = active & (selected_token != selected_target)
            alive = alive & (~mismatch)

        # Surviving trajectories must exactly equal the target suffix.
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
    """
    Faster batched version of the original estimator.

    Same high-level behavior:
      - samples a random reveal permutation for each sample
      - uses the same fixed schedule across steps
      - computes the path probability of obtaining target_tokens
      - supports 'top_k' and full-softmax decoding
      - returns the same output structure

    Assumptions kept from the original code:
      - prompt_tokens has shape [1, prompt_len]
      - target_tokens has shape [1, suffix_len]
      - attention mask produced by _suffix_attention_mask is valid for the full sequence
    """
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
        raise ValueError('For model_family="llama", prompt_tokens must contain at least one token.')
    if decoding_scheme not in {'top_k', 'full', 'greedy'}:
        raise ValueError("decoding_scheme must be one of {'top_k', 'full', 'greedy'} for model_family='llama'.")
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

    model_family = model_family.lower()
    if model_family != 'llada':
        raise ValueError('compute_diffusion_probabilistic_extraction only supports model_family="llada".')
    normalized_decoding_scheme = decoding_scheme.lower()
    if normalized_decoding_scheme not in {'full', 'top_k', 'elbo'}:
        raise ValueError("decoding_scheme must be one of {'full', 'top_k', 'ELBO'} for model_family='llada'.")
    if normalized_decoding_scheme == 'top_k' and k <= 0:
        raise ValueError('k must be > 0 when decoding_scheme="top_k".')
    if normalized_decoding_scheme == 'random' and remasking != 'random':
        raise ValueError('decoding_scheme="random" requires remasking="random".')

    if normalized_decoding_scheme == 'elbo':
        result = _elbo_probability(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
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
    if target_tokens.shape[1] < steps:
        raise ValueError('steps must be <= target suffix length for this scheduler.')

    if remasking == 'target-token-confidence':
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
        return {
            'method': 'path_sampling',
            'probability': path_sampling_result['probability'],
            'sample_probabilities': path_sampling_result['sample_probabilities'],
            'num_samples': path_sampling_result['num_samples'],
            'remasking': 'random',
            'decoding_scheme': normalized_decoding_scheme,
            'k': k if normalized_decoding_scheme == 'top_k' else None,
        }

    if estimation_method == 'path_sampling':
        if normalized_decoding_scheme not in {'full', 'top_k'}:
            raise ValueError('estimation_method="path_sampling" with remasking="low-confidence" requires decoding_scheme in {"full", "top_k"}.')
        if not math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('estimation_method="path_sampling" with remasking="low-confidence" requires temperature == 1.')
        if num_samples <= 0:
            raise ValueError('num_samples must be > 0 when estimation_method="path_sampling".')

        path_sampling = _path_sampling_probability_temperature1_tie_free(
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
        )
        return {
            'method': 'path_sampling',
            'probability': path_sampling.estimate,
            'estimate': path_sampling.estimate,
            'standard_error': path_sampling.standard_error,
            'wald_ci': path_sampling.wald_ci,
            'wilson_ci': path_sampling.wilson_ci,
            'hits': path_sampling.hits,
            'num_samples': path_sampling.num_samples,
            'remasking': 'low-confidence',
            'temperature': temperature,
            'decoding_scheme': normalized_decoding_scheme,
            'k': k if normalized_decoding_scheme == 'top_k' else None,
        }

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

    if temperature > 0:

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
        'decoding_scheme': decoding_scheme,
        'k': k if decoding_scheme == 'top_k' else None,
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
        raise ValueError('compute_autoregressive_probabilistic_extraction only supports model_family in {"llama", "llama2"}.')

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
):
    model_family = model_family.lower()
    if model_family in AUTOREGRESSIVE_MODEL_FAMILIES:
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
        )
    raise ValueError("model_family must be one of {'llada', 'llama', 'llama2'}")
