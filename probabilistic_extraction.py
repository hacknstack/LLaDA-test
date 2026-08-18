import math
import secrets
from dataclasses import dataclass
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
        '_elbo_probability, _path_sampling_random_probability, or '
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

@torch.inference_mode()
def _path_sampling_random_probability_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, 100]
    masked_indexes: list[int],              # 1-indexed masked positions
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
    Batched random-remasking trajectory estimator for partially masked
    conditioning.

    Estimates

        p_{theta, phi, M}(z_M | z_not_M)

    by sampling random reveal trajectories and evaluating

        hat p =
            prod_{t=1}^T
            prod_{i in B_t}
                p_i(z_i | S_{t-1}),

    where the ordered reveal batches B_1, ..., B_T are induced by a
    uniformly random permutation of the masked set M and the deterministic
    transfer schedule.

    For steps == len(masked_indexes), this is the one-token-per-step
    specialization.

    Precision strategy for A100/H100:
      - model forward: model's native dtype (typically BF16/FP16/FP32)
      - selected vocabulary logits: FP32
      - softmax/top-k normalization: FP32
      - sums of token log-probabilities: FP64
      - path log-probabilities: FP64
      - Monte Carlo averaging: FP64/log-space

    Supported decoding schemes:
      - "full": full temperature-scaled softmax
      - "top_k": temperature-scaled softmax restricted to top-k logits

    Temperature must be finite and strictly positive.

    Assumptions:
      - sequence_tokens has shape [1, 100]
      - exactly 50 positions are masked
      - 1 <= steps <= 50
      - attention_mask, if provided, has shape [1, 100]
    """

    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device)

    # ------------------------------------------------------------------
    # Validate inputs
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

    # Convert 1-indexed -> sorted unique 0-indexed positions.
    masked_pos = sorted(
        set(int(i) - 1 for i in masked_indexes)
    )

    if len(masked_pos) != 50:
        raise ValueError(
            f"Expected exactly 50 masked positions out of 100, "
            f"got {len(masked_pos)}"
        )

    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError(
            "masked_indexes must be 1-indexed positions in [1, 100]"
        )

    masked_len = len(masked_pos)

    if steps < 1 or steps > masked_len:
        raise ValueError(
            f"steps must be in [1, {masked_len}], got {steps}"
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if (
        not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError(
            "temperature must be finite and strictly positive."
        )

    if decoding_scheme not in {"full", "top_k"}:
        raise ValueError(
            "decoding_scheme must be either 'full' or 'top_k', "
            f"got {decoding_scheme!r}"
        )

    if decoding_scheme == "top_k" and k <= 0:
        raise ValueError(
            "k must be positive when decoding_scheme='top_k'."
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
    )
    # [50]

    full_target_row = sequence_tokens[0]
    masked_target_row = full_target_row[masked_pos_t]
    # [50]

    # ------------------------------------------------------------------
    # Deterministic transfer schedule
    #
    # b_t = floor(|M| / T) + indicator(t <= |M| mod T)
    # ------------------------------------------------------------------

    base = masked_len // steps
    rem = masked_len % steps

    schedule = [
        base + (1 if i < rem else 0)
        for i in range(steps)
    ]

    assert sum(schedule) == masked_len
    assert all(step_size >= 1 for step_size in schedule)

    # ------------------------------------------------------------------
    # RNG
    #
    # seed=None uses PyTorch's ordinary global RNG.
    #
    # We generate random permutation keys in FP64. For 50 positions,
    # this makes accidental equal random keys negligibly likely while
    # remaining trivial in cost compared with a model forward.
    # ------------------------------------------------------------------

    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Outputs / stable global Monte Carlo accumulator
    # ------------------------------------------------------------------

    sample_probabilities: List[float] = []

    running_log_sum = torch.tensor(
        float("-inf"),
        dtype=torch.float64,
        device=device,
    )

    num_accumulated = 0

    # ==================================================================
    # Monte Carlo batches
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

        # --------------------------------------------------------------
        # Initial sequence state:
        #
        # observed positions remain z;
        # masked positions begin at mask_id.
        # --------------------------------------------------------------

        x = sequence_tokens.expand(
            bsz,
            -1,
        ).clone()

        x[:, masked_pos_t] = mask_id

        # Path probability is accumulated entirely in log-space / FP64.
        log_path_probability = torch.zeros(
            bsz,
            dtype=torch.float64,
            device=device,
        )

        alive = torch.ones(
            bsz,
            dtype=torch.bool,
            device=device,
        )

        # --------------------------------------------------------------
        # Uniform random reveal permutation.
        #
        # Sorting IID continuous random keys produces a uniform random
        # permutation. FP64 makes finite-precision key collisions
        # negligible here.
        #
        # CPU work is tiny: only [batch_size, 50].
        # --------------------------------------------------------------

        perm_scores = torch.rand(
            (bsz, masked_len),
            generator=rng,
            device="cpu",
            dtype=torch.float64,
        )

        permutation = torch.argsort(
            perm_scores,
            dim=-1,
        ).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        # [bsz, 50]
        #
        # Values index masked_pos_t / masked_target_row.

        # Expand attention mask once per Monte Carlo batch rather than once
        # per diffusion step.
        batched_attn = None

        if attention_mask is not None:
            batched_attn = attention_mask.expand(
                bsz,
                -1,
            )

        start = 0

        # ==============================================================
        # Decode trajectory
        # ==============================================================

        for step_size in schedule:
            reveal_slots = permutation[
                :,
                start:start + step_size,
            ]
            # [bsz, step_size]

            start += step_size

            # ----------------------------------------------------------
            # Model forward.
            #
            # All positions in this batch must be evaluated from the SAME
            # pre-reveal state x.
            # ----------------------------------------------------------

            outputs = model(
                x,
                attention_mask=batched_attn,
            )

            logits = outputs.logits
            # [bsz, 100, vocab]

            vocab_size = logits.shape[-1]

            # ----------------------------------------------------------
            # Map masked-slot indices -> absolute sequence positions.
            # ----------------------------------------------------------

            reveal_abs_positions = masked_pos_t[
                reveal_slots
            ]
            # [bsz, step_size]

            gather_index = (
                reveal_abs_positions
                .unsqueeze(-1)
                .expand(
                    -1,
                    -1,
                    vocab_size,
                )
            )

            # ----------------------------------------------------------
            # Only selected positions are promoted to FP32.
            #
            # This is important when the model itself runs BF16/FP16.
            # Doing the vocabulary normalization directly in BF16 would
            # lose accuracy before the FP64 path accumulator ever sees it.
            # ----------------------------------------------------------

            step_logits = torch.gather(
                logits,
                dim=1,
                index=gather_index,
            ).float()
            # [bsz, step_size, vocab], FP32

            # Release references to the much larger full output ASAP.
            del outputs
            del logits

            target_ids = torch.gather(
                masked_target_row
                .unsqueeze(0)
                .expand(bsz, -1),
                dim=1,
                index=reveal_slots,
            )
            # [bsz, step_size]

            # ==========================================================
            # Token log probabilities
            # ==========================================================

            if decoding_scheme == "top_k":
                # ------------------------------------------------------
                # Temperature scaling does NOT alter top-k membership
                # for tau > 0, so perform top-k on the raw FP32 logits.
                #
                # This avoids dividing the entire [B, step, V] tensor.
                # ------------------------------------------------------

                top_k = min(
                    int(k),
                    vocab_size,
                )

                topk_vals, topk_idx = torch.topk(
                    step_logits,
                    k=top_k,
                    dim=-1,
                )
                # [bsz, step_size, top_k]

                in_topk = (
                    topk_idx
                    == target_ids.unsqueeze(-1)
                ).any(dim=-1)
                # [bsz, step_size]

                target_raw_logits = torch.gather(
                    step_logits,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)
                # [bsz, step_size]

                if tau == 1.0:
                    target_scaled_logits = target_raw_logits
                    topk_scaled_vals = topk_vals
                else:
                    target_scaled_logits = (
                        target_raw_logits / tau
                    )

                    topk_scaled_vals = (
                        topk_vals / tau
                    )

                topk_log_normalizer = torch.logsumexp(
                    topk_scaled_vals,
                    dim=-1,
                )
                # [bsz, step_size]

                # +inf normalization indicates a numerical pathology.
                if bool(
                    torch.isposinf(
                        topk_log_normalizer
                    ).any().item()
                ):
                    raise FloatingPointError(
                        "Encountered +inf top-k log-normalizer. "
                        "This may indicate non-finite model logits or "
                        "an excessively small temperature."
                    )

                token_log_probs = (
                    target_scaled_logits
                    - topk_log_normalizer
                )
                # [bsz, step_size]

                token_log_probs = torch.where(
                    in_topk,
                    token_log_probs,
                    torch.full_like(
                        token_log_probs,
                        float("-inf"),
                    ),
                )

            else:
                # ------------------------------------------------------
                # Full-distribution decoder:
                #
                # p(v) = softmax(logits / tau)_v
                #
                # To avoid allocating a second huge scaled tensor, scale
                # the selected-position FP32 logits in-place.
                # ------------------------------------------------------

                if tau != 1.0:
                    step_logits.div_(tau)

                target_scaled_logits = torch.gather(
                    step_logits,
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)
                # [bsz, step_size]

                log_normalizer = torch.logsumexp(
                    step_logits,
                    dim=-1,
                )
                # [bsz, step_size]

                if bool(
                    torch.isposinf(
                        log_normalizer
                    ).any().item()
                ):
                    raise FloatingPointError(
                        "Encountered +inf log-normalizer. "
                        "This may indicate non-finite model logits or "
                        "an excessively small temperature."
                    )

                token_log_probs = (
                    target_scaled_logits
                    - log_normalizer
                )
                # [bsz, step_size]

            # ----------------------------------------------------------
            # Numerical validation.
            #
            # -inf is legitimate: it means the target token has zero
            # probability under this decoder.
            #
            # NaN or +inf are not legitimate probabilities.
            # ----------------------------------------------------------

            invalid = (
                torch.isnan(token_log_probs)
                | torch.isposinf(token_log_probs)
            )

            if bool(invalid.any().item()):
                raise FloatingPointError(
                    "Encountered NaN or +inf token log-probability "
                    "during random-remasking trajectory estimation."
                )

            # Due only to floating-point roundoff, target-logsumexp can
            # occasionally become an extremely small positive number.
            # A probability cannot exceed 1, so enforce log p <= 0.
            token_log_probs.clamp_max_(0.0)

            # ----------------------------------------------------------
            # If any simultaneously revealed target token has probability
            # zero, this complete trajectory has probability zero.
            # ----------------------------------------------------------

            step_has_zero = torch.isneginf(
                token_log_probs
            ).any(dim=-1)
            # [bsz]

            # Replace -inf by zero ONLY for the summation itself.
            # step_has_zero separately records that the trajectory is dead.
            safe_token_log_probs = torch.where(
                torch.isneginf(token_log_probs),
                torch.zeros_like(token_log_probs),
                token_log_probs,
            )

            # ----------------------------------------------------------
            # Promote before summing.
            #
            # This is the important FP32 -> FP64 precision boundary.
            # ----------------------------------------------------------

            step_log_prob = (
                safe_token_log_probs
                .to(torch.float64)
                .sum(dim=-1)
            )
            # [bsz], FP64

            was_alive = alive

            still_alive = (
                was_alive
                & (~step_has_zero)
            )

            log_path_probability = torch.where(
                still_alive,
                log_path_probability + step_log_prob,
                log_path_probability,
            )

            alive = still_alive

            # ----------------------------------------------------------
            # Successful path state update.
            #
            # The probability of generating these target tokens has just
            # been included in the path weight, so the next successful
            # state contains the targets visibly.
            #
            # Updating dead rows as well is harmless and avoids expensive
            # dynamic batch compaction / synchronization in the usual
            # full-distribution case.
            # ----------------------------------------------------------

            x.scatter_(
                dim=1,
                index=reveal_abs_positions,
                src=target_ids,
            )

        # ==============================================================
        # Complete path probabilities
        # ==============================================================

        batch_log_probs = torch.where(
            alive,
            log_path_probability,
            torch.full_like(
                log_path_probability,
                float("-inf"),
            ),
        )
        # [bsz], FP64

        # --------------------------------------------------------------
        # Arithmetic Monte Carlo mean accumulated stably:
        #
        #   log sum_r exp(log W_r)
        # --------------------------------------------------------------

        batch_log_sum = torch.logsumexp(
            batch_log_probs,
            dim=0,
        )

        running_log_sum = torch.logaddexp(
            running_log_sum,
            batch_log_sum,
        )

        num_accumulated += bsz

        # --------------------------------------------------------------
        # Preserve original per-sample output format.
        #
        # Extremely tiny probabilities can underflow when converted from
        # log-space to an ordinary float. The estimator/mean itself stays
        # in log-space until the final conversion.
        # --------------------------------------------------------------

        batch_probabilities = torch.where(
            torch.isfinite(batch_log_probs),
            torch.exp(batch_log_probs),
            torch.zeros_like(batch_log_probs),
        )

        sample_probabilities.extend(
            batch_probabilities
            .detach()
            .cpu()
            .tolist()
        )

    # ==================================================================
    # Final arithmetic mean
    #
    #   (1 / K) sum_r W_r
    #
    # ==================================================================

    log_average_probability = (
        running_log_sum
        - math.log(num_accumulated)
    ).item()

    if math.isfinite(log_average_probability):
        # A true probability cannot exceed one. This only protects against
        # microscopic positive roundoff in the accumulated log probability.
        log_average_probability = min(
            log_average_probability,
            0.0,
        )

        average_probability = float(
            math.exp(log_average_probability)
        )
    else:
        average_probability = 0.0

    # ------------------------------------------------------------------
    # Preserve original output format
    # ------------------------------------------------------------------

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
    batch_size: int = 64,
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
    When use_state_cache=True, log a(S) is memoized by the 50-bit revealed-set
    state S.  Identical states are deduplicated within a Monte Carlo batch and
    reused across later batches, so the expensive estimator computation is
    performed only once per unique state encountered.

    In verbose mode, deterministic diagnostic subvalues are cached alongside
    log a(S): target log-probabilities, the tie-adjusted competitor win mass,
    highest-possible-position flags, and tie counts.  highest_sampled is NOT
    cached: it is freshly resampled for every trajectory occurrence.  Cache
    misses draw those samples while the full-vocabulary CDF is already present;
    cache hits perform only a lightweight diagnostic resampling forward instead
    of recomputing A(S).

    Caching is disabled only for model.training=True, because stochastic
    training-mode forwards would make the state values non-deterministic.

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
      - masked_indexes contains exactly 50 valid 1-indexed positions;
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

    # Convert 1-indexed masked positions to sorted unique 0-indexed slots.
    masked_pos = sorted(
        set(int(i) - 1 for i in masked_indexes)
    )

    if len(masked_pos) != 50:
        raise ValueError(
            f"Expected exactly 50 masked positions out of 100, "
            f"got {len(masked_pos)}"
        )

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
    )  # [50]

    full_target_row = sequence_tokens[0]
    masked_target_row = full_target_row[masked_pos_t]
    # [50]

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
    batch_ids_base = torch.arange(
        max_bsz, dtype=torch.long, device=device
    ).unsqueeze(1)
    eye_base = torch.eye(
        masked_len, dtype=torch.bool, device=device
    )
    reveal_true_base = torch.ones(
        (max_bsz, 1), dtype=torch.bool, device=device
    )

    # Each revealed-set state S fits in 50 bits.  Keeping a compact state id
    # lets us memoize log a(S) without transferring the full [50] boolean mask
    # to the CPU for every trajectory and step.
    state_bit_values = torch.bitwise_left_shift(
        torch.ones(masked_len, dtype=torch.int64, device=device),
        torch.arange(masked_len, dtype=torch.int64, device=device),
    )

    # The estimator cache stays on the model device: each entry is only a
    # 50-element FP64 log-a vector.  Verbose deterministic diagnostics are tiny
    # and are cached on CPU so verbose caching adds negligible VRAM.
    #
    # highest_sampled is intentionally never cached.  It is a stochastic
    # diagnostic and is resampled independently for every trajectory occurrence.
    cache_enabled = bool(
        use_state_cache
        and not bool(getattr(model, "training", False))
    )
    state_log_a_cache: Dict[int, torch.Tensor] = {}
    state_verbose_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    cache_requests = 0
    cache_hits = 0
    cache_misses = 0
    cache_forward_rows_saved = 0
    verbose_diagnostic_forward_rows = 0

    # ==================================================================
    # Compute log a_i(S)
    # ==================================================================

    def _compute_log_a_for_batch_uncached(
        x: torch.Tensor,              # [bsz, 100]
        revealed: torch.Tensor,       # [bsz, 50]
        alive: torch.Tensor,          # [bsz]
        num_unrevealed: int,
        verbose_draw_counts: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, object]]]:
        """
        Returns

            log_a_full : [bsz, 50], float64

        where revealed positions and dead trajectories are -inf.

        Full-vocab work is vectorized across all currently masked positions.
        """

        bsz = x.shape[0]
        m = num_unrevealed

        unrevealed = ~revealed

        # --------------------------------------------------------------
        # Attention mask
        # --------------------------------------------------------------

        batched_attn = None

        if attention_mask is not None:
            batched_attn = attention_mask.expand(
                bsz,
                -1,
            )

        # --------------------------------------------------------------
        # Model forward
        # --------------------------------------------------------------

        # Keep the forward in the model's existing dtype.  Disabling autocast
        # prevents an outer autocast context from silently changing it.
        if device.type in {"cuda", "cpu"}:
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model(
                    x,
                    attention_mask=batched_attn,
                )
        else:
            outputs = model(
                x,
                attention_mask=batched_attn,
            )

        logits = outputs.logits
        # [bsz, 100, vocab], model-native dtype

        vocab_size = logits.shape[-1]

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

        batch_ids = batch_ids_base[:bsz]

        # Keep the active-position copy in the model's native dtype first.
        # A widening cast to FP64 preserves the exact value ordering and exact
        # ties of FP16/BF16/FP32 logits, so the expensive vocabulary sort can
        # later be done in the native dtype without changing confidence order.
        active_logits_native = logits[
            batch_ids,
            active_abs_positions,
            :,
        ]
        # [bsz, m, vocab], model-native dtype

        del outputs
        del logits

        # Probability calculations remain FP64.
        active_logits = active_logits_native.to(dtype=torch.float64)
        # [bsz, m, vocab], FP64

        # --------------------------------------------------------------
        # Normalizers for confidence and sampling distributions
        # --------------------------------------------------------------
        #
        # confidence:
        #
        #   c_j(v) = exp(l_j(v)) / Z_conf_j
        #
        # sampling:
        #
        #   p_j(v) = exp(l_j(v) / tau) / Z_sample_j
        #
        # --------------------------------------------------------------

        log_Z_conf = torch.logsumexp(
            active_logits,
            dim=-1,
        )
        # [bsz, m], FP64

        if tau == 1.0:
            # Exact same distribution; avoid a second vocab reduction.
            log_Z_sample = log_Z_conf
        else:
            log_Z_sample = torch.logsumexp(
                active_logits / tau,
                dim=-1,
            )
            # [bsz, m], FP64

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
        # Final-step fast path.  With one masked position there are no
        # competitors, so the successful-transition probability is simply
        # p_i(z_i | S).  Sorting/CDF/tie machinery would be pure overhead.
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
            del active_logits_native
            return log_a_full, verbose_batch

        highest_possible_active = None
        highest_sampled_active = None
        if verbose:
            maximum_confidence = active_logits.max(dim=-1).values - log_Z_conf
            highest_possible_active = maximum_confidence == maximum_confidence.max(
                dim=-1, keepdim=True
            ).values
            del maximum_confidence

        # --------------------------------------------------------------
        # Sort by CONFIDENCE.
        #
        # Within a fixed position j,
        #
        #     c_j(v) = softmax(l_j)_v
        #
        # is strictly monotone in l_j(v), so sorting raw logits is
        # equivalent to sorting confidence.
        #
        # Crucially this ordering is also independent of temperature.
        # --------------------------------------------------------------

        # The normalizers/target probabilities above needed FP64, but the sort
        # only needs the ordering.  Sorting before widening is substantially
        # cheaper on GPU and is behavior-preserving because conversion from the
        # model dtype to FP64 is exact and monotone.
        del active_logits

        sorted_logits_native = torch.sort(
            active_logits_native,
            dim=-1,
        ).values
        del active_logits_native

        sorted_logits = sorted_logits_native.to(dtype=torch.float64)
        del sorted_logits_native
        # [bsz, m, vocab], FP64

        # --------------------------------------------------------------
        # Build the CDF under the SAMPLING distribution.
        #
        # Because sorted_logits are already in increasing confidence order:
        #
        #   log CDF_j[k]
        #     =
        #   log sum_{r <= k}
        #       p_j(v_r).
        #
        # No giant sample_log_probs tensor is necessary.
        # --------------------------------------------------------------

        if tau == 1.0:
            log_cdf = torch.logcumsumexp(
                sorted_logits,
                dim=-1,
            )
        else:
            log_cdf = torch.logcumsumexp(
                sorted_logits / tau,
                dim=-1,
            )

        log_cdf.sub_(
            log_Z_sample.unsqueeze(-1)
        )

        # Numerically CDF <= 1, so log CDF <= 0.
        log_cdf.clamp_max_(0.0)
        # [bsz, m, vocab], FP64

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
        # Scatter back into the fixed 50-slot representation.
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

    def _resample_highest_sampled_for_cached_states(
        x: torch.Tensor,
        revealed: torch.Tensor,
        representative_rows: List[int],
        draw_counts: List[int],
        num_unrevealed: int,
    ) -> List[torch.Tensor]:
        """Freshly resample highest_sampled for cached states.

        The cached estimator values do not retain vocabulary-sized tensors.
        Therefore a cache hit performs only a diagnostic model forward and
        categorical sampling; it does NOT recompute sorting, CDF comparisons,
        a_i(S), A(S), or q(i|S).

        Returns one CPU bool tensor [draw_count, m] per representative state.
        """
        nonlocal verbose_diagnostic_forward_rows

        if not representative_rows:
            return []

        m = num_unrevealed

        # Final step is deterministic and requires no model forward/RNG.
        if m == 1:
            return [
                torch.ones((int(count), 1), dtype=torch.bool)
                for count in draw_counts
            ]

        rep_idx = torch.tensor(
            representative_rows, dtype=torch.long, device=device
        )
        diag_x = x.index_select(0, rep_idx)
        diag_revealed = revealed.index_select(0, rep_idx)
        n_states = len(representative_rows)

        batched_attn = None
        if attention_mask is not None:
            batched_attn = attention_mask.expand(n_states, -1)

        if device.type in {"cuda", "cpu"}:
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model(
                    diag_x,
                    attention_mask=batched_attn,
                )
        else:
            outputs = model(
                diag_x,
                attention_mask=batched_attn,
            )

        logits = outputs.logits
        verbose_diagnostic_forward_rows += n_states

        local_slot_grid = slot_grid_base.expand(n_states, -1)
        active_slots = local_slot_grid[
            ~diag_revealed
        ].view(n_states, m)
        active_abs_positions = masked_pos_t[active_slots]

        local_batch_ids = torch.arange(
            n_states, dtype=torch.long, device=device
        ).unsqueeze(1)

        active_logits_native = logits[
            local_batch_ids,
            active_abs_positions,
            :,
        ]
        del outputs
        del logits

        draws: List[torch.Tensor] = []

        # Process each unique cached state independently after the shared model
        # forward.  This limits temporary FP64 vocabulary storage to [m, vocab]
        # instead of [num_states, m, vocab].
        for state_row, count_raw in enumerate(draw_counts):
            count = int(count_raw)
            if count <= 0:
                raise ValueError("Diagnostic draw counts must be positive.")

            active_logits = active_logits_native[state_row].to(torch.float64)
            log_Z_conf = torch.logsumexp(active_logits, dim=-1)

            # torch.multinomial accepts unnormalized non-negative weights.
            # Shifting by each position's maximum avoids overflow and preserves
            # exactly the softmax(logits / tau) categorical distribution.
            scaled = active_logits / tau
            scaled = scaled - scaled.max(dim=-1, keepdim=True).values
            weights = torch.exp(scaled)

            if sample_on_device:
                sampled_token_ids = torch.multinomial(
                    weights,
                    num_samples=count,
                    replacement=True,
                    generator=diagnostic_rng,
                )
            else:
                sampled_token_ids = torch.multinomial(
                    weights.detach().cpu(),
                    num_samples=count,
                    replacement=True,
                    generator=diagnostic_rng,
                ).to(device)
            # [m, count]

            sampled_raw_logits = torch.gather(
                active_logits,
                dim=-1,
                index=sampled_token_ids,
            )
            sampled_confidence = (
                sampled_raw_logits
                - log_Z_conf.unsqueeze(-1)
            )
            highest = (
                sampled_confidence
                == sampled_confidence.max(dim=0, keepdim=True).values
            ).transpose(0, 1)
            # [count, m]

            draws.append(highest.detach().cpu())

            del active_logits
            del scaled
            del weights
            del sampled_token_ids
            del sampled_raw_logits
            del sampled_confidence

        del active_logits_native
        return draws

    def compute_log_a_for_batch(
        x: torch.Tensor,
        revealed: torch.Tensor,
        alive: torch.Tensor,
        num_unrevealed: int,
        state_ids: torch.Tensor,
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

        # One compact synchronization per step/batch: only state ids for live
        # trajectories are transferred to CPU.
        live_rows = live_rows_t.detach().cpu().tolist()
        live_state_ids = state_ids[live_rows_t].detach().cpu().tolist()
        cache_requests += num_live

        # Group rows by state while preserving first-occurrence order.
        rows_by_state: Dict[int, List[int]] = {}
        for row, state_id in zip(live_rows, live_state_ids):
            key = int(state_id)
            rows_by_state.setdefault(key, []).append(int(row))

        missing_keys: List[int] = []
        missing_representative_rows: List[int] = []
        missing_draw_counts: List[int] = []

        cached_diag_keys: List[int] = []
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
                value = missing_log_a[local_idx].clone()
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
            # Derive log_a_active directly from the cached/full 50-slot output,
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

        # Compact 50-bit representation of S for memoization.  Bit r is set iff
        # masked slot r has already been successfully revealed.
        state_ids = torch.zeros(
            bsz, dtype=torch.int64, device=device
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
                state_ids=state_ids,
            )
            # [bsz, 50], FP64

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

            still_alive = (
                alive
                & torch.isfinite(log_A)
            )

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
            # [bsz, 50], FP64

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

            state_ids.bitwise_or_(
                state_bit_values[next_slots]
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

    use_exact_low_confidence_dp = (
        model_family.lower() == 'llada'
        and estimation_method == 'exact'
        and remasking == 'low-confidence'
        and masked_indexes is not None
    )
    normalized_masked_indexes = validate_masked_indexes(
        masked_indexes,
        expected_count=None if use_exact_low_confidence_dp else 50,
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
    mc_batch_size: int = 512,
    model_batch_size: int = 64,
    verbose: bool = False,
    verbose_compact: bool = False,
    verbose_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
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

    hits = 0
    verbose_samples: List[Dict[str, object]] = []

    # As in the successful-trajectory implementation, state sharing is valid
    # only when the model itself is deterministic for a fixed input state.
    deduplicate_states = not bool(
        getattr(model, "training", False)
    )

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
            # torch.unique(dim=0) can reorder states, which can change the
            # composition/order of model-forward batches and thereby create
            # tiny GPU numerical differences.
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

            if deduplicate_states:

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

            else:
                # Training-mode fallback: every alive trajectory gets its own
                # model forward so stochastic model draws are not shared.
                rows_per_state = [
                    [int(row)]
                    for row in alive_rows_cpu
                ]

                representative_rows = [
                    int(row)
                    for row in alive_rows_cpu
                ]

            num_states = len(
                representative_rows
            )

            # ----------------------------------------------------------
            # Evaluate unique states in VRAM-bounded chunks.
            # ----------------------------------------------------------

            for state_chunk_start in range(
                0,
                num_states,
                model_batch_size,
            ):
                state_chunk_end = min(
                    state_chunk_start + model_batch_size,
                    num_states,
                )

                chunk_state_indices = list(
                    range(
                        state_chunk_start,
                        state_chunk_end,
                    )
                )

                rep_rows_chunk = [
                    representative_rows[s]
                    for s in chunk_state_indices
                ]

                rep_rows_t = torch.tensor(
                    rep_rows_chunk,
                    dtype=torch.long,
                    device=device,
                )

                state_x = x.index_select(
                    0,
                    rep_rows_t,
                )

                state_revealed = revealed.index_select(
                    0,
                    rep_rows_t,
                )

                chunk_n = len(
                    rep_rows_chunk
                )

                batched_attn = None

                if attention_mask is not None:
                    batched_attn = attention_mask.expand(
                        chunk_n,
                        -1,
                    )

                # ------------------------------------------------------
                # Model forward.
                #
                # Identical policy to STS:
                #   - native model dtype
                #   - outer autocast disabled
                # ------------------------------------------------------

                if device.type in {"cuda", "cpu"}:
                    with torch.autocast(
                        device_type=device.type,
                        enabled=False,
                    ):
                        outputs = model(
                            state_x,
                            attention_mask=batched_attn,
                        )
                else:
                    outputs = model(
                        state_x,
                        attention_mask=batched_attn,
                    )

                logits = outputs.logits
                del outputs

                # ------------------------------------------------------
                # Active masked slots.
                # ------------------------------------------------------

                slot_grid = slot_grid_base.expand(
                    chunk_n,
                    -1,
                )

                active_slots = slot_grid[
                    ~state_revealed
                ].view(
                    chunk_n,
                    m,
                )

                active_abs_positions = masked_pos_t[
                    active_slots
                ]

                local_batch_ids = torch.arange(
                    chunk_n,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(1)

                active_logits_native = logits[
                    local_batch_ids,
                    active_abs_positions,
                    :,
                ]

                del logits

                # ======================================================
                # One unique state at a time
                # ======================================================

                for (
                    local_state_idx,
                    global_state_idx,
                ) in enumerate(chunk_state_indices):

                    trajectory_rows = rows_per_state[
                        global_state_idx
                    ]

                    group_size = len(
                        trajectory_rows
                    )

                    rows_t = torch.tensor(
                        trajectory_rows,
                        dtype=torch.long,
                        device=device,
                    )

                    state_active_slots = active_slots[
                        local_state_idx
                    ]
                    # [m]

                    state_active_logits_native = (
                        active_logits_native[
                            local_state_idx
                        ]
                    )
                    # [m, vocab], native dtype

                    # STS widens active logits to FP64 before probability
                    # calculations.
                    state_active_logits = (
                        state_active_logits_native
                        .to(torch.float64)
                    )
                    # [m, vocab]

                    vocab_size = int(
                        state_active_logits.shape[-1]
                    )

                    # ==================================================
                    # Normalizers -- same arithmetic as STS
                    # ==================================================

                    log_Z_conf = torch.logsumexp(
                        state_active_logits,
                        dim=-1,
                    )
                    # [m]

                    if tau == 1.0:
                        # In the requested temp=1 case this is literally
                        # the same tensor/value as the confidence normalizer.
                        log_Z_sample = log_Z_conf
                    else:
                        log_Z_sample = torch.logsumexp(
                            state_active_logits / tau,
                            dim=-1,
                        )
                    # [m]

                    # ==================================================
                    # Candidate sampling -- STS-aligned log-CDF method
                    #
                    # STS sorts in the model/native dtype and only then
                    # widens to FP64. Do exactly the same here.
                    # ==================================================

                    (
                        sorted_logits_native,
                        sorted_token_ids,
                    ) = torch.sort(
                        state_active_logits_native,
                        dim=-1,
                    )

                    sorted_logits = (
                        sorted_logits_native
                        .to(torch.float64)
                    )

                    del sorted_logits_native

                    if tau == 1.0:
                        log_cdf = torch.logcumsumexp(
                            sorted_logits,
                            dim=-1,
                        )
                    else:
                        log_cdf = torch.logcumsumexp(
                            sorted_logits / tau,
                            dim=-1,
                        )

                    log_cdf.sub_(
                        log_Z_sample.unsqueeze(-1)
                    )

                    # Same numerical guard as STS.
                    log_cdf.clamp_max_(0.0)

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

                del active_logits_native

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
    argument.  Determinism additionally assumes the model is deterministic for a
    fixed input state (normally model.eval()).
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

        batched_attn = attention_mask

        # --------------------------------------------------------------
        # Model forward -- same dtype policy as MC / STS
        # --------------------------------------------------------------

        if device.type in {"cuda", "cpu"}:
            with torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                outputs = model(
                    x,
                    attention_mask=batched_attn,
                )
        else:
            outputs = model(
                x,
                attention_mask=batched_attn,
            )

        logits = outputs.logits
        del outputs

        # Gather logits only for still-masked positions, then widen to FP64.
        active_logits_native = logits[
            0,
            active_abs_positions,
            :,
        ]
        del logits

        active_logits = active_logits_native.to(torch.float64)
        del active_logits_native
        # [m, vocab]

        # ==============================================================
        # 1. DUEL ranking: highest POSSIBLE untempered confidence
        #
        #    log h_i = max_v l_i(v) - logsumexp_v l_i(v)
        #
        # Compare these FP64 log-confidences directly.
        # ==============================================================

        log_Z_conf = torch.logsumexp(
            active_logits,
            dim=-1,
        )
        # [m]

        max_raw_logits = active_logits.max(
            dim=-1,
        ).values
        # [m]

        highest_log_confidence = (
            max_raw_logits
            - log_Z_conf
        )
        # [m]

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

        if tau == 1.0:
            log_Z_sample = log_Z_conf
        else:
            log_Z_sample = torch.logsumexp(
                active_logits / tau,
                dim=-1,
            )

        active_target_ids = masked_target_row[active_slots]
        target_raw_logits = torch.gather(
            active_logits,
            dim=-1,
            index=active_target_ids.unsqueeze(-1),
        ).squeeze(-1)
        # [m]

        target_sample_log_probs = (
            target_raw_logits / tau
            - log_Z_sample
        )
        # [m]

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

        del active_logits
        del log_Z_conf
        del max_raw_logits
        del highest_log_confidence
        del log_Z_sample
        del active_target_ids
        del target_raw_logits
        del target_sample_log_probs
        del chosen_target_log_probability

    # ==================================================================
    # Final probability
    # ==================================================================

    log_probability_value = float(
        log_probability.item()
    )

    if math.isfinite(log_probability_value):
        try:
            probability = float(
                math.exp(log_probability_value)
            )
        except OverflowError:
            probability = float("inf")
    else:
        probability = 0.0

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
    active_logits_native: torch.Tensor,   # [B, r, V]
    active_target_ids: torch.Tensor,      # [B, r]
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
    device = active_logits_native.device
    B, r, V = active_logits_native.shape
    tau = float(temperature)

    # --------------------------------------------------------------
    # Same numerical convention as STS / MC:
    #
    # model output remains native dtype;
    # all probability calculations after that are FP64.
    # --------------------------------------------------------------
    active_logits = active_logits_native.to(torch.float64)

    log_Z_conf = torch.logsumexp(
        active_logits,
        dim=-1,
    )

    if tau == 1.0:
        log_Z_sample = log_Z_conf
    else:
        log_Z_sample = torch.logsumexp(
            active_logits / tau,
            dim=-1,
        )

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

    # --------------------------------------------------------------
    # Same strategy as STS:
    #
    # sorting only needs the ordering, so sort in native dtype and
    # widen to FP64 afterward.
    # --------------------------------------------------------------
    del active_logits

    sorted_logits_native = torch.sort(
        active_logits_native,
        dim=-1,
    ).values

    sorted_logits = sorted_logits_native.to(torch.float64)
    del sorted_logits_native

    # --------------------------------------------------------------
    # CDF under the SAMPLING distribution.
    #
    # Sorting by raw logit is equivalent to sorting by confidence,
    # because softmax is monotone within a position.
    # --------------------------------------------------------------
    if tau == 1.0:
        log_cdf = torch.logcumsumexp(
            sorted_logits,
            dim=-1,
        )
    else:
        log_cdf = torch.logcumsumexp(
            sorted_logits / tau,
            dim=-1,
        )

    log_cdf.sub_(
        log_Z_sample.unsqueeze(-1)
    )

    # Numerical guard identical in spirit to STS.
    log_cdf.clamp_max_(0.0)

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

    # Deliberately preserve STS's competitor accumulation order.
    for competitor_j in range(r):

        strict_for_smaller = (
            competitor_j < proposed_i
        )

        nonstrict_for_larger = (
            competitor_j > proposed_i
        )

        competitor_factor = torch.where(
            strict_for_smaller,
            log_L[:, competitor_j, :],
            torch.where(
                nonstrict_for_larger,
                log_LE[:, competitor_j, :],
                torch.zeros(
                    (),
                    dtype=torch.float64,
                    device=device,
                ),
            ),
        )

        log_win_mass = (
            log_win_mass
            + competitor_factor
        )

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

      * native model-forward dtype

      * FP64 probability calculations after logits

      * probability products/sums accumulated in log-space

    Complexity for m masked positions:

        states = 2^m

    and every nonterminal state is evaluated exactly once.

    For m=12:
        4095 model-state evaluations

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

    # Exact DP assumes one deterministic set of logits for each x(S).
    #
    # The supplied MC explicitly stops sharing states in training mode,
    # because dropout/etc. would make a fixed state stochastic.
    # Integrating that additional randomness would no longer be this
    # finite exact DP.
    if bool(
        getattr(model, "training", False)
    ):
        raise ValueError(
            "Exact DP requires deterministic logits for each state. "
            "Call model.eval() first."
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

    slot_grid_base = torch.arange(
        masked_len,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # log_a_table[S, i] = log a_i(S).
    #
    # Revealed / invalid transitions remain -inf.
    #
    # For m=12 this is only:
    #
    #   4096 * 12 * 8 ~= 384 KiB.
    log_a_table = torch.full(
        (
            num_states,
            masked_len,
        ),
        -math.inf,
        dtype=torch.float64,
        device=device,
    )

    # ==============================================================
    # Evaluate every nonterminal model state exactly once.
    #
    # Popcount sorting has two advantages:
    #
    # 1. equal active-set sizes remain contiguous for postprocessing;
    # 2. we still chunk the FLATTENED state list, so a forward batch
    #    can span two DP layers.
    #
    # Consequently the model-forward count is exactly
    #
    #   ceil((2^m - 1) / state_batch_size),
    #
    # instead of sum_k ceil(C(m,k)/batch).
    # ==============================================================

    state_order = sorted(
        range(full_state),
        key=int.bit_count,
    )

    for batch_start in range(
        0,
        full_state,
        state_batch_size,
    ):
        batch_states_py = state_order[
            batch_start:
            batch_start + state_batch_size
        ]

        batch_size = len(
            batch_states_py
        )

        state_ids = torch.tensor(
            batch_states_py,
            dtype=torch.int64,
            device=device,
        )

        revealed = (
            torch.bitwise_and(
                state_ids.unsqueeze(1),
                bit_values.unsqueeze(0),
            )
            != 0
        )
        # [B, m]

        # ----------------------------------------------------------
        # Construct x(S):
        #
        # outside M:
        #     target token
        #
        # revealed inside M:
        #     target token
        #
        # unrevealed inside M:
        #     MASK
        # ----------------------------------------------------------

        x = sequence_tokens.expand(
            batch_size,
            -1,
        ).clone()

        x[
            :,
            masked_pos_t,
        ] = torch.where(
            revealed,
            masked_target_row.unsqueeze(0),
            torch.full(
                (
                    batch_size,
                    masked_len,
                ),
                int(mask_id),
                dtype=sequence_tokens.dtype,
                device=device,
            ),
        )

        batched_attn = None

        if attention_mask is not None:
            batched_attn = (
                attention_mask.expand(
                    batch_size,
                    -1,
                )
            )

        # ----------------------------------------------------------
        # Model forward:
        #
        # identical dtype policy to supplied MC / STS.
        # ----------------------------------------------------------

        if device.type in {
            "cuda",
            "cpu",
        }:
            with torch.autocast(
                device_type=device.type,
                enabled=False,
            ):
                outputs = model(
                    x,
                    attention_mask=batched_attn,
                )
        else:
            outputs = model(
                x,
                attention_mask=batched_attn,
            )

        # Retain only positions in M before releasing the much larger
        # [B, sequence_length, vocab] output.
        masked_logits_native = (
            outputs.logits[
                :,
                masked_pos_t,
                :,
            ]
        )

        del outputs
        del x

        # ----------------------------------------------------------
        # Rows with the same popcount have the same number r of
        # still-masked positions, so process those together.
        # ----------------------------------------------------------

        popcounts = [
            state.bit_count()
            for state in batch_states_py
        ]

        group_start = 0

        while group_start < batch_size:

            num_revealed = (
                popcounts[group_start]
            )

            group_end = (
                group_start + 1
            )

            while (
                group_end < batch_size
                and popcounts[group_end]
                == num_revealed
            ):
                group_end += 1

            group_size = (
                group_end
                - group_start
            )

            num_unrevealed = (
                masked_len
                - num_revealed
            )

            group_revealed = revealed[
                group_start:
                group_end
            ]

            active_slots = (
                slot_grid_base
                .expand(
                    group_size,
                    -1,
                )[
                    ~group_revealed
                ]
                .view(
                    group_size,
                    num_unrevealed,
                )
            )

            # Because masked_pos was sorted, active_slots are also in
            # increasing ABSOLUTE sequence-index order.
            #
            # This is what makes local j < i exactly equivalent to
            # the MC decoder's smallest-sequence-index tie rule.

            local_rows = torch.arange(
                group_start,
                group_end,
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)

            active_logits_native = (
                masked_logits_native[
                    local_rows,
                    active_slots,
                    :,
                ]
            )

            active_target_ids = (
                masked_target_row[
                    active_slots
                ]
            )

            group_log_a = (
                _exact_low_conf_log_a(
                    active_logits_native=(
                        active_logits_native
                    ),
                    active_target_ids=(
                        active_target_ids
                    ),
                    temperature=tau,
                )
            )

            group_state_ids = state_ids[
                group_start:
                group_end
            ]

            log_a_table[
                group_state_ids.unsqueeze(1),
                active_slots,
            ] = group_log_a

            group_start = (
                group_end
            )

        del masked_logits_native
        del revealed
        del state_ids

    # ==============================================================
    # Exact subset DP.
    #
    # Let D[S] be the total probability of reaching S while every
    # revealed token has matched the target.
    #
    # D[empty] = 1
    #
    # D[S U {i}] += D[S] a_i(S)
    #
    # Every possible successful reveal order corresponds to exactly
    # one path from empty -> full, so D[full] is the desired
    # extraction probability.
    # ==============================================================

    # Tiny table; moving it to CPU avoids thousands of tiny GPU
    # kernels during the combinatorial DP.
    log_a_cpu = (
        log_a_table
        .detach()
        .cpu()
        .tolist()
    )

    log_dp = [
        -math.inf
    ] * num_states

    log_dp[0] = 0.0

    # Numeric state order is already topological:
    #
    #   S | (1 << i) > S.
    for state in range(
        full_state
    ):
        base = log_dp[
            state
        ]

        if base == -math.inf:
            continue

        remaining = (
            full_state ^ state
        )

        while remaining:

            bit = (
                remaining
                & -remaining
            )

            slot = (
                bit.bit_length()
                - 1
            )

            log_transition = (
                log_a_cpu[
                    state
                ][
                    slot
                ]
            )

            if (
                log_transition
                != -math.inf
            ):
                next_state = (
                    state | bit
                )

                candidate = (
                    base
                    + log_transition
                )

                log_dp[
                    next_state
                ] = _logaddexp_scalar(
                    log_dp[
                        next_state
                    ],
                    candidate,
                )

            remaining ^= bit

    log_probability = float(
        log_dp[
            full_state
        ]
    )

    if math.isfinite(
        log_probability
    ):
        probability = math.exp(
            log_probability
        )
    else:
        probability = 0.0

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
            math.ceil(
                full_state
                / state_batch_size
            ),

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
