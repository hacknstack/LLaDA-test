import math
import secrets
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Any

import torch
import torch.nn.functional as F
from get_log_likelihood import get_log_likelihood, get_log_likelihood_from_partially_masked

AUTOREGRESSIVE_MODEL_FAMILIES = {'llama', 'llama2', 'olmo', 'mistral'}


@dataclass
class MonteCarloResult:
    estimate: float
    standard_error: float
    wald_ci: Tuple[float, float]
    wilson_ci: Tuple[float, float]
    hits: int
    num_samples: int


def validate_masked_indexes(masked_indexes: Optional[Sequence[int]]) -> Optional[List[int]]:
    if masked_indexes is None:
        return None

    normalized = [int(index) for index in masked_indexes]
    if len(normalized) != 50:
        raise ValueError('--masked_indexes must contain exactly 50 integers.')
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
        '_monte_carlo_probability_temperature_fast. '
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
    validate_no_ties: bool = False,  # retained for API compatibility; ties are handled uniformly
    return_samples: bool = True,
    verbose: bool = False,
    verbose_compact: bool = False,
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

    where ties at the maximum confidence are broken uniformly.  For a
    proposed successful winner i, define

        L_{j,i}(S) = P[c_j(V_j | S) <  c_i^*(S)]
        E_{j,i}(S) = P[c_j(V_j | S) == c_i^*(S)].

    If exactly k competitors tie i and every other competitor is below i,
    the decoder selects i with probability 1/(k+1).  Thus a_i(S) is the
    target-token sampling probability p_i(z_i | S) times the total probability
    that i wins after this uniform tie-break.  The implementation computes that
    total exactly with a log-space dynamic program over the number of tied
    competitors.

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
        confidence thresholds, successful-transition probabilities, proposal
        probabilities, trajectory log weights, and Monte Carlo averaging are FP64;
      - all multiplicative probabilities are accumulated in log-space.

    TIE BREAKING
    ------------
    If multiple sampled positions share the maximum confidence, the decoder is
    assumed to choose uniformly among those tied maxima.  The estimator
    marginalizes this uniform tie-breaking exactly inside a_i(S); it does not
    need to sample an additional tie-break RNG draw.

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
    tie_denominators_log_base = torch.arange(
        1, masked_len + 1, dtype=torch.float64, device=device
    ).log()
    reveal_true_base = torch.ones(
        (max_bsz, 1), dtype=torch.bool, device=device
    )

    # ==================================================================
    # Compute log a_i(S)
    # ==================================================================

    def compute_log_a_for_batch(
        x: torch.Tensor,              # [bsz, 100]
        revealed: torch.Tensor,       # [bsz, 50]
        alive: torch.Tensor,          # [bsz]
        num_unrevealed: int,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
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

        if verbose:
            if sample_on_device:
                diagnostic_uniform = torch.rand(
                    (bsz, m), device=device, dtype=torch.float64, generator=diagnostic_rng
                )
            else:
                diagnostic_uniform = torch.rand(
                    (bsz, m), device='cpu', dtype=torch.float64, generator=diagnostic_rng
                ).to(device)
            sampled_sorted_slots = torch.searchsorted(
                log_cdf, diagnostic_uniform.log().unsqueeze(-1), right=False
            ).squeeze(-1).clamp_max(vocab_size - 1)
            sampled_logits = torch.gather(
                sorted_logits, dim=-1, index=sampled_sorted_slots.unsqueeze(-1)
            ).squeeze(-1)
            sampled_confidence = sampled_logits - log_Z_conf
            highest_sampled_active = sampled_confidence == sampled_confidence.max(
                dim=-1, keepdim=True
            ).values
            del diagnostic_uniform
            del sampled_sorted_slots
            del sampled_logits
            del sampled_confidence

        # --------------------------------------------------------------
        # Construct confidence thresholds.
        #
        # For competitor position j and proposed successful winner i, let
        #
        #   c* = c_i(z_i | S).
        #
        # We need the competitor probabilities
        #
        #   L_{j,i} = P[c_j(V_j) <  c*]
        #   E_{j,i} = P[c_j(V_j) == c*]
        #
        # under the sampling distribution p_j.  Any competitor with confidence
        # above c* makes i unable to win.  If K competitors tie exactly at c*,
        # uniform tie-breaking selects i with probability 1/(K+1).
        #
        # In logit coordinates, c_j(v) compared with c* is equivalent to
        # comparing l_j(v) against
        #
        #   threshold_{j,i} = log Z_conf,j + log c_i*.
        # --------------------------------------------------------------

        thresholds = (
            log_Z_conf.unsqueeze(2)
            + target_conf_log_probs.unsqueeze(1)
        ).contiguous()
        # [bsz, competitor j, proposed winner i], FP64

        # left_idx: first competitor token with confidence >= c*.
        # right_idx: first competitor token with confidence >  c*.
        left_idx = torch.searchsorted(
            sorted_logits,
            thresholds,
            right=False,
        )
        right_idx = torch.searchsorted(
            sorted_logits,
            thresholds,
            right=True,
        )
        # [bsz, m, m]

        # --------------------------------------------------------------
        # L_{j,i} = P(confidence < c*) in log-space.
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

        # Exact CDF endpoints.
        log_L.masked_fill_(left_idx == 0, float("-inf"))
        log_L.masked_fill_(left_idx == vocab_size, 0.0)
        # [bsz, competitor j, proposed winner i], FP64

        # --------------------------------------------------------------
        # E_{j,i} = P(confidence == c*) in log-space.
        #
        # All vocabulary entries in [left_idx, right_idx) have the same raw
        # logit, so their sampling probabilities are equal.  Computing the tie
        # mass as count * p(one tied token) avoids subtracting nearly equal CDFs.
        # --------------------------------------------------------------

        equal_count = right_idx - left_idx
        has_equal = equal_count > 0

        # --------------------------------------------------------------
        # Exclude j == i from the competitor set.  The neutral factor is
        # L=1.  Cross-position exact ties are usually absent, so detect them
        # before constructing E or the O(m^3) tie-count dynamic program.
        # --------------------------------------------------------------

        eye_m = eye_base[:m, :m].unsqueeze(0)
        log_L.masked_fill_(eye_m, 0.0)

        cross_tie_mask = has_equal & (~eye_m)
        has_cross_tie = bool(cross_tie_mask.any().item())

        tie_count = torch.zeros(bsz, dtype=torch.long, device=device)
        if verbose and has_cross_tie:
            tie_count = (
                cross_tie_mask & alive.view(bsz, 1, 1)
            ).sum(dim=(1, 2))

        # validate_no_ties is deliberately ignored.  It remains in the public
        # signature only so existing callers do not break; ties are handled by
        # exact uniform tie-breaking.
        _ = validate_no_ties

        if not has_cross_tie:
            # Fast path: with no cross-position tie mass, uniform tie-breaking
            # reduces exactly to the original strict-order product.  Accumulate
            # in competitor order to preserve the same FP64 operation ordering
            # as the k=0 branch of the full dynamic program, but only over a
            # [bsz, m] tensor instead of [bsz, m, m].
            log_uniform_win_mass = torch.zeros(
                (bsz, m), dtype=torch.float64, device=device
            )
            for competitor_j in range(m):
                log_uniform_win_mass = (
                    log_uniform_win_mass + log_L[:, competitor_j, :]
                )

            log_E = None
            log_dp = None
            log_L_by_i = None
            log_E_by_i = None

        else:
            # ----------------------------------------------------------
            # E_{j,i} = P(confidence == c*) in log-space.
            #
            # All vocabulary entries in [left_idx, right_idx) have the same raw
            # logit, so their sampling probabilities are equal.  Computing the
            # tie mass as count * p(one tied token) avoids subtracting nearly
            # equal CDFs.
            # ----------------------------------------------------------

            equal_gather_idx = left_idx.clamp(
                min=0,
                max=vocab_size - 1,
            )
            equal_raw_logit = torch.gather(
                sorted_logits,
                dim=-1,
                index=equal_gather_idx,
            )

            log_E = (
                equal_count.clamp_min(1).to(torch.float64).log()
                + equal_raw_logit / tau
                - log_Z_sample.unsqueeze(2)
            )
            log_E.masked_fill_(~has_equal, float("-inf"))
            log_E.masked_fill_(eye_m, float("-inf"))

            # ----------------------------------------------------------
            # Exact uniform tie-breaking probability.
            #
            # For proposed winner i, let K be the number of competitors whose
            # sampled confidence equals c_i*.  Conditional on no competitor
            # being above c_i*, i wins with probability 1/(K+1).
            # ----------------------------------------------------------

            log_dp = torch.full(
                (bsz, m, m),
                float("-inf"),
                dtype=torch.float64,
                device=device,
            )
            log_dp[:, :, 0] = 0.0

            # transpose() is only a view; contiguous copies are unnecessary.
            log_L_by_i = log_L.transpose(1, 2)
            log_E_by_i = log_E.transpose(1, 2)

            for competitor_j in range(m):
                log_l = log_L_by_i[:, :, competitor_j].unsqueeze(-1)
                log_e = log_E_by_i[:, :, competitor_j].unsqueeze(-1)

                stay_below = log_dp + log_l

                become_tie = torch.full_like(log_dp, float("-inf"))
                become_tie[:, :, 1:] = log_dp[:, :, :-1] + log_e

                log_dp = torch.logaddexp(
                    stay_below,
                    become_tie,
                )

            log_uniform_win_mass = torch.logsumexp(
                log_dp
                - tie_denominators_log_base[:m].view(1, 1, m),
                dim=-1,
            )
            # [bsz, proposed winner i], FP64

        log_a_active = (
            target_sample_log_probs
            + log_uniform_win_mass
        )
        # [bsz, m], FP64

        # Retain a diagnostic quantity with the old name/shape expectation:
        # this is now the log probability that the competitor field permits i
        # to win after exact uniform tie-breaking, rather than simply sum log F.
        log_product = log_uniform_win_mass
        target_sample_log_probs_64 = target_sample_log_probs

        # No longer needed before returning.
        del sorted_logits
        del log_cdf
        del log_L
        if has_cross_tie:
            del log_E
            del log_dp
            del log_L_by_i
            del log_E_by_i

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
            }

        return log_a_full, verbose_batch

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
        "tie_breaking": "uniform_among_max_confidence",
        "model_forward_dtype": "native",
        "estimator_dtype_after_logits": "float64",
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

    normalized_masked_indexes = validate_masked_indexes(masked_indexes)
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
    if verbose:
        valid_verbose = (
            normalized_masked_indexes is not None
            and remasking == 'low-confidence'
            and estimation_method == 'path_sampling'
            and normalized_decoding_scheme == 'full'
            and math.isclose(float(temperature), 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
        if not valid_verbose:
            raise ValueError(
                'verbose diagnostics require partially masked low-confidence '
                'path sampling with full decoding and temperature 1.'
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
            _unsupported_partially_masked_configuration(
                remasking=remasking,
                estimation_method=estimation_method,
                decoding_scheme=normalized_decoding_scheme,
            )
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
        'decoding_scheme': decoding_scheme,
        'k': k if decoding_scheme == 'top_k' else None,
    }

@torch.inference_mode()
def _monte_carlo_probability_temperature_fast_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,               # [1, L] full target sequence z
    masked_indexes: list[int],                   # 1-indexed masked positions
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
    Naive Monte Carlo estimate of the probability that low-confidence
    remasking exactly regenerates the masked part of `sequence_tokens`,
    conditioned on the unmasked part.

    Specialization:
      - exactly one token is permanently transferred per step;
      - therefore steps == len(masked_indexes);
      - all currently masked positions independently sample a candidate;
      - the position whose sampled candidate has the largest confidence
        is permanently transferred.

    IMPORTANT: sampling and confidence are different distributions.

    Candidate sampling:
        p_i(v | S) = softmax(logits_i / temperature)_v

    Low-confidence ranking:
        c_i(v | S) = softmax(logits_i)_v

    Thus temperature affects which token is sampled, but confidence is
    always the standard untempered softmax confidence.

    For decoding_scheme == "top_k":
      - candidate sampling is restricted to the top-k raw-logit tokens and
        uses the temperature-scaled distribution within that support;
      - confidence is STILL the full-vocabulary untempered softmax
        confidence c_i(v | S).

    GPU / numerical strategy:
      - model forward stays in the model's native dtype;
      - only logits at currently masked positions are gathered;
      - gathered logits are promoted to FP32;
      - categorical sampling uses Gumbel-max, avoiding explicit softmax
        probability tensors and torch.multinomial over the full vocabulary;
      - confidence is evaluated stably in log-space;
      - failed trajectories are immediately removed from subsequent model
        forwards;
      - only currently unrevealed masked positions are processed;
      - no FP64 vocabulary-sized tensors are created.

    The returned estimate remains an ordinary Bernoulli Monte Carlo
    estimate based on exact reconstruction hits.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if (
        not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError(
            "temperature must be finite and > 0"
        )

    if steps <= 0:
        raise ValueError(
            "steps must be positive"
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive"
        )

    if mc_batch_size <= 0:
        raise ValueError(
            "mc_batch_size must be positive"
        )

    if decoding_scheme not in {"top_k", "full"}:
        raise ValueError(
            "decoding_scheme must be 'top_k' or 'full'"
        )

    if decoding_scheme == "top_k" and k <= 0:
        raise ValueError(
            "k must be >= 1 when decoding_scheme == 'top_k'"
        )

    device = _model_device(model)

    sequence_tokens = sequence_tokens.to(
        device,
        non_blocking=True,
    )

    if (
        sequence_tokens.ndim != 2
        or sequence_tokens.shape[0] != 1
    ):
        raise ValueError(
            "sequence_tokens must have shape [1, L], "
            f"got {tuple(sequence_tokens.shape)}"
        )

    seq_len = sequence_tokens.shape[1]

    # Convert 1-indexed -> sorted unique 0-indexed positions.
    masked_pos = sorted(
        set(int(i) - 1 for i in masked_indexes)
    )

    if len(masked_pos) == 0:
        raise ValueError(
            "masked_indexes must be non-empty"
        )

    if any(
        pos < 0 or pos >= seq_len
        for pos in masked_pos
    ):
        raise ValueError(
            "masked_indexes must be 1-indexed positions "
            f"in [1, {seq_len}]"
        )

    num_masked = len(masked_pos)

    if steps != num_masked:
        raise ValueError(
            "This specialization assumes one token transferred "
            "per step, so steps must equal the number of masked "
            "positions. "
            f"Got steps={steps}, num_masked={num_masked}."
        )

    masked_pos_t = torch.tensor(
        masked_pos,
        dtype=torch.long,
        device=device,
    )
    # [M]

    target_masked_tokens = sequence_tokens[
        0,
        masked_pos_t,
    ]
    # [M]

    # ------------------------------------------------------------------
    # Attention mask
    # ------------------------------------------------------------------

    base_attn = None

    if attention_mask is not None:
        attention_mask = attention_mask.to(
            device,
            non_blocking=True,
        )

        if (
            attention_mask.ndim != 2
            or attention_mask.shape != (1, seq_len)
        ):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )

        base_attn = attention_mask

    # ------------------------------------------------------------------
    # RNG
    #
    # seed=None uses PyTorch's global device RNG.
    # ------------------------------------------------------------------

    if seed is None:
        rng = None
    else:
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))

    tau = float(temperature)

    hits = 0

    # ==================================================================
    # Monte Carlo batches
    # ==================================================================

    for start in range(
        0,
        num_samples,
        mc_batch_size,
    ):
        initial_bsz = min(
            mc_batch_size,
            num_samples - start,
        )

        # --------------------------------------------------------------
        # Initial state
        # --------------------------------------------------------------

        x = sequence_tokens.expand(
            initial_bsz,
            -1,
        ).clone()

        x[:, masked_pos_t] = mask_id

        # Instead of maintaining an [B,M] revealed mask and repeatedly
        # scanning it, directly maintain the masked slots still eligible
        # for transfer.
        #
        # remaining_slots[b] contains indices into:
        #
        #     masked_pos_t
        #     target_masked_tokens
        #
        remaining_slots = torch.arange(
            num_masked,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(
            initial_bsz,
            -1,
        ).clone()
        # [B,M]

        active_bsz = initial_bsz

        # ==============================================================
        # One-token-per-step low-confidence decoding
        # ==============================================================

        for step_idx in range(steps):
            if active_bsz == 0:
                break

            num_remaining = num_masked - step_idx

            # ----------------------------------------------------------
            # Attention mask for surviving rollouts only.
            # ----------------------------------------------------------

            attn_batch = None

            if base_attn is not None:
                attn_batch = base_attn.expand(
                    active_bsz,
                    -1,
                )

            # ----------------------------------------------------------
            # Model forward.
            #
            # Dead rollouts have already been removed, so every row here
            # can still produce an exact extraction hit.
            # ----------------------------------------------------------

            outputs = model(
                x,
                attention_mask=attn_batch,
            )

            logits = outputs.logits
            # [B_active, L, V]

            vocab_size = logits.shape[-1]

            # ----------------------------------------------------------
            # Absolute sequence positions of currently masked candidates.
            # ----------------------------------------------------------

            remaining_abs_positions = masked_pos_t[
                remaining_slots
            ]
            # [B_active, m]

            batch_idx = torch.arange(
                active_bsz,
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)

            # ----------------------------------------------------------
            # Gather ONLY currently eligible positions.
            #
            # Promote after the model forward to FP32. This prevents
            # BF16/FP16 softmax-confidence errors without forcing the
            # expensive transformer forward to FP32.
            # ----------------------------------------------------------

            active_logits = logits[
                batch_idx,
                remaining_abs_positions,
                :,
            ].float()
            # [B_active, m, V], FP32

            del outputs
            del logits

            # ----------------------------------------------------------
            # Untempered confidence normalizer.
            #
            # c_i(v) = softmax(raw_logits_i)_v
            #
            # Work in log-confidence:
            #
            # log c_i(v) =
            #     raw_logit_i(v) - logsumexp(raw_logits_i)
            #
            # Argmax of confidence == argmax of log confidence.
            # ----------------------------------------------------------

            log_Z_conf = torch.logsumexp(
                active_logits,
                dim=-1,
            )
            # [B_active, m], FP32

            # Catch NaN/+inf/all--inf model outputs without scanning a
            # second full [B,m,V] boolean tensor.
            if not bool(
                torch.isfinite(log_Z_conf).all().item()
            ):
                raise FloatingPointError(
                    "Encountered non-finite confidence normalizer "
                    "during low-confidence Monte Carlo sampling."
                )

            # ==========================================================
            # Sample one candidate token independently at every
            # currently masked position.
            #
            # Gumbel-max:
            #
            #   argmax_v [l_v / tau + G_v]
            #
            # is equivalent, for tau > 0, to
            #
            #   argmax_v [l_v + tau * G_v].
            #
            # The latter is preferable numerically because very small
            # temperatures do not require dividing logits by tiny tau.
            #
            # If E ~ Exp(1), then G = -log(E) is standard Gumbel.
            # ==========================================================

            if decoding_scheme == "top_k":
                top_k = min(
                    int(k),
                    vocab_size,
                )

                # Temperature > 0 does not change top-k membership, so
                # find support from raw logits before temperature enters.
                top_vals, top_idx = torch.topk(
                    active_logits,
                    k=top_k,
                    dim=-1,
                )
                # [B_active,m,K]

                if top_k == 1:
                    # No random sampling required.
                    sampled_tokens = top_idx[
                        ...,
                        0,
                    ]
                    # [B_active,m]

                else:
                    # Generate Gumbel noise in-place from Exp(1).
                    gumbel_scores = torch.empty_like(
                        top_vals
                    )

                    gumbel_scores.exponential_(
                        1.0,
                        generator=rng,
                    )

                    # Protect log() against a finite-precision zero draw.
                    gumbel_scores.clamp_min_(
                        torch.finfo(
                            gumbel_scores.dtype
                        ).tiny
                    )

                    # E -> -tau log(E) + raw_logit
                    gumbel_scores.log_()
                    gumbel_scores.mul_(-tau)
                    gumbel_scores.add_(top_vals)

                    sampled_local = torch.argmax(
                        gumbel_scores,
                        dim=-1,
                    )
                    # [B_active,m]

                    sampled_tokens = torch.gather(
                        top_idx,
                        dim=-1,
                        index=sampled_local.unsqueeze(-1),
                    ).squeeze(-1)
                    # [B_active,m]

                    del gumbel_scores
                    del sampled_local

                del top_vals
                del top_idx

            else:
                # ------------------------------------------------------
                # Full-vocabulary sampling.
                #
                # Avoid explicitly constructing:
                #
                #     softmax(logits / tau)
                #
                # and then feeding a [B*m,V] probability tensor into
                # torch.multinomial.
                #
                # Gumbel-max samples exactly from the same categorical
                # distribution while requiring only one temporary tensor.
                # ------------------------------------------------------

                gumbel_scores = torch.empty_like(
                    active_logits
                )

                gumbel_scores.exponential_(
                    1.0,
                    generator=rng,
                )

                gumbel_scores.clamp_min_(
                    torch.finfo(
                        gumbel_scores.dtype
                    ).tiny
                )

                gumbel_scores.log_()
                gumbel_scores.mul_(-tau)
                gumbel_scores.add_(active_logits)

                sampled_tokens = torch.argmax(
                    gumbel_scores,
                    dim=-1,
                )
                # [B_active,m]

                del gumbel_scores

            # ==========================================================
            # Compute confidence of the sampled candidates.
            #
            # IMPORTANT:
            #
            # sampled_tokens came from p_i(.; tau),
            # but confidence comes from UNTEMPERED c_i(.).
            # ==========================================================

            sampled_raw_logits = torch.gather(
                active_logits,
                dim=-1,
                index=sampled_tokens.unsqueeze(-1),
            ).squeeze(-1)
            # [B_active,m], FP32

            sampled_log_confidence = (
                sampled_raw_logits
                - log_Z_conf
            )
            # [B_active,m], FP32

            del active_logits
            del sampled_raw_logits
            del log_Z_conf

            # ----------------------------------------------------------
            # Low-confidence remasking:
            #
            # permanently reveal the position whose sampled candidate has
            # the HIGHEST standard-softmax confidence.
            #
            # Using log confidence avoids unnecessary exponentiation.
            # ----------------------------------------------------------

            selected_local = torch.argmax(
                sampled_log_confidence,
                dim=-1,
            )
            # [B_active]
            #
            # In the paper's no-ties regime this is unambiguous.
            # Exact ties follow torch.argmax's deterministic convention.

            row_idx = torch.arange(
                active_bsz,
                dtype=torch.long,
                device=device,
            )

            selected_slot = remaining_slots[
                row_idx,
                selected_local,
            ]
            # [B_active]
            #
            # Index into masked_pos_t / target_masked_tokens.

            selected_abs_pos = masked_pos_t[
                selected_slot
            ]
            # [B_active]

            selected_token = sampled_tokens[
                row_idx,
                selected_local,
            ]
            # [B_active]

            selected_target = target_masked_tokens[
                selected_slot
            ]
            # [B_active]

            del sampled_tokens
            del sampled_log_confidence

            # ----------------------------------------------------------
            # Once a permanently transferred token is wrong, exact
            # reconstruction is impossible. Such trajectories can be
            # discarded immediately without changing the Bernoulli MC
            # estimator.
            # ----------------------------------------------------------

            success = (
                selected_token
                == selected_target
            )
            # [B_active]

            survivor_idx = torch.nonzero(
                success,
                as_tuple=False,
            ).squeeze(-1)
            # [B_survivors]

            new_bsz = survivor_idx.numel()

            if new_bsz == 0:
                active_bsz = 0
                break

            # ----------------------------------------------------------
            # Keep only surviving decoder states.
            #
            # Update selected position with its target token. Since these
            # rows survived, selected_token == selected_target.
            # ----------------------------------------------------------

            x = x.index_select(
                0,
                survivor_idx,
            )

            survivor_selected_abs = selected_abs_pos[
                survivor_idx
            ]

            survivor_selected_target = selected_target[
                survivor_idx
            ]

            survivor_rows = torch.arange(
                new_bsz,
                dtype=torch.long,
                device=device,
            )

            x[
                survivor_rows,
                survivor_selected_abs,
            ] = survivor_selected_target

            # ----------------------------------------------------------
            # Remove the just-revealed slot from each survivor's compact
            # list of remaining masked positions.
            #
            # Before: [B_survivors, m]
            # After:  [B_survivors, m-1]
            # ----------------------------------------------------------

            survivor_remaining = remaining_slots.index_select(
                0,
                survivor_idx,
            )
            # [B_survivors,m]

            survivor_selected_local = selected_local[
                survivor_idx
            ]
            # [B_survivors]

            if num_remaining > 1:
                col_idx = torch.arange(
                    num_remaining,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                keep_mask = (
                    col_idx
                    != survivor_selected_local.unsqueeze(1)
                )
                # [B_survivors,m]

                remaining_slots = survivor_remaining[
                    keep_mask
                ].view(
                    new_bsz,
                    num_remaining - 1,
                )

            else:
                # Final token has been revealed.
                remaining_slots = survivor_remaining[
                    :,
                    :0,
                ]

            active_bsz = new_bsz

        # ==============================================================
        # Every trajectory still present after all M reveals is a hit.
        #
        # Every permanent transfer was checked against its corresponding
        # target, and all M positions have now been transferred.
        # ==============================================================

        hits += int(active_bsz)

    # ==================================================================
    # Bernoulli Monte Carlo estimate / intervals
    # ==================================================================

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
    )
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
