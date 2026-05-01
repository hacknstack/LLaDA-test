import math
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

@torch.inference_mode()
def _path_sampling_random_probability_from_partially_masked(
    model,
    sequence_tokens: torch.Tensor,          # [1, 100]
    masked_indexes: list[int],              # 1-indexed masked positions in sequence_tokens
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
    Batched path-sampling estimator for partially masked conditioning.

    High-level behavior:
      - sequence_tokens is the full target sequence z, shape [1, 100]
      - masked_indexes specifies the 50 positions to regenerate
      - the other 50 positions are observed / conditioning tokens
      - samples a random reveal permutation over the 50 masked positions
      - uses the same fixed schedule across steps
      - computes the path probability of obtaining the target tokens at those masked positions
      - supports 'top_k' and full-softmax decoding
      - returns the same output structure as the suffix-only version

    Assumptions:
      - sequence_tokens has shape [1, 100]
      - exactly 50 positions are masked
      - if steps == 50, this is the one-token-per-step specialization
      - attention_mask, if provided, has shape [1, 100]
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

    # Convert 1-indexed -> 0-indexed, deduplicate, sort, validate.
    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))
    if len(masked_pos) != 50:
        raise ValueError(
            f"Expected exactly 50 masked positions out of 100, got {len(masked_pos)}"
        )
    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError("masked_indexes must be 1-indexed positions in [1, 100]")

    masked_len = len(masked_pos)
    masked_pos_t = torch.tensor(masked_pos, dtype=torch.long, device=device)   # [50]

    if steps <= 0:
        raise ValueError("steps must be positive")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], got {tuple(attention_mask.shape)}"
            )

    # Keep CPU RNG for seeded reproducibility style close to the original.
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    base = masked_len // steps
    rem = masked_len % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]

    full_target_row = sequence_tokens[0]                 # [100]
    masked_target_row = full_target_row[masked_pos_t]    # [50]

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []

    for batch_start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - batch_start)

        # Current sequence states for all samples in this batch.
        x = sequence_tokens.expand(bsz, -1).clone()      # [bsz, 100]
        x[:, masked_pos_t] = mask_id

        # Accumulated log-probability for each sample.
        log_path_probability = torch.zeros(bsz, dtype=torch.float64, device=device)

        # Whether the sample is still alive (not zero-probability yet).
        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        # Random reveal permutations over the 50 masked slots, one per sample.
        # Generated on CPU using the seeded CPU RNG, then moved to device.
        perm_scores = torch.rand((bsz, masked_len), generator=rng, device="cpu")
        permutation = perm_scores.argsort(dim=-1).to(device)  # [bsz, 50]
        # permutation indexes into masked_pos_t / masked_target_row, not absolute positions.

        start = 0
        for step_size in schedule:
            reveal_slots = permutation[:, start:start + step_size]  # [bsz, step_size], values in [0, 49]
            start += step_size

            # Repeat attention mask across batch if needed.
            batched_attn = None
            if attention_mask is not None:
                batched_attn = attention_mask.expand(bsz, -1)

            logits = model(x, attention_mask=batched_attn).logits   # [bsz, 100, vocab]
            vocab_size = logits.shape[-1]

            # Map reveal slots -> absolute sequence positions.
            reveal_abs_positions = masked_pos_t[reveal_slots]       # [bsz, step_size]

            gather_index = reveal_abs_positions.unsqueeze(-1).expand(-1, -1, vocab_size)
            step_logits = torch.gather(logits, dim=1, index=gather_index)  # [bsz, step_size, vocab]

            target_ids = torch.gather(
                masked_target_row.unsqueeze(0).expand(bsz, -1),
                dim=1,
                index=reveal_slots,
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

            # Update x with the revealed target tokens for all samples.
            x.scatter_(dim=1, index=reveal_abs_positions, src=target_ids)

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

@torch.inference_mode()
def _path_sampling_low_confidence_full_topk_probability(
    model,
    prompt_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    steps: int,
    attention_mask: Optional[torch.Tensor],
    mask_id: int,
    num_samples: int,
    seed: Optional[int],
    k: int,
    temperature: float,
    batch_size: int = 64,
) -> Dict[str, object]:
    """
    Approximate path-sampling estimator for the third estimator in the note:

        low-confidence remasking,
        full-distribution token sampling,
        temperature = temperature,
        top-k retained successful transitions.

    This estimates p_z^{(k)}, not the exact full-distribution p_z unless
    k covers the full vocabulary.

    Estimator:
        \hat p_z^{(k), full} = prod_t A^{(k)}(S_{t-1})

    where

        A^{(k)}(S) = sum_i a_i^{(k)}(S)

    and

        a_i^{(k)}(S)
        =
        1{target_i in top-k at position i}
        p_i(target_i)
        prod_{j != i, j masked} F_j(p_i(target_i)).

    Here F_j(c) is the full-distribution confidence CDF:

        F_j(c) = sum_{v : p_j(v) < c} p_j(v).

    Important:
      - This low-confidence decoder reveals exactly one position per step.
      - Therefore this implementation requires steps == suffix_len.
      - The full CDF computation sorts the full vocabulary distribution, so
        this is more expensive than the random-remasking estimator.
    """
    device = _model_device(model)
    prompt_tokens = prompt_tokens.to(device)
    target_tokens = target_tokens.to(device)

    suffix_len = target_tokens.shape[1]
    prompt_len = prompt_tokens.shape[1]

    if steps != suffix_len:
        raise ValueError(
            "The low-confidence estimator reveals exactly one suffix token per step, "
            f"so steps must equal suffix_len. Got steps={steps}, suffix_len={suffix_len}."
        )

    if k <= 0:
        return {
            "probability": 0.0,
            "sample_probabilities": [0.0 for _ in range(num_samples)],
            "num_samples": num_samples,
            "estimation_method": "path_sampling_low_confidence_full_topk",
        }

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    # Keep CPU RNG style close to the random-remasking estimator.
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    prompt_row = prompt_tokens[0]   # [prompt_len]
    target_row = target_tokens[0]   # [suffix_len]

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []

    for batch_start in range(0, num_samples, batch_size):
        bsz = min(batch_size, num_samples - batch_start)

        suffix = torch.full(
            (bsz, suffix_len),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        # True while this estimator sample has not hit A^{(k)}(S) = 0.
        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        # Accumulates log prod_t A^{(k)}(S_{t-1}).
        log_estimate = torch.zeros(bsz, dtype=torch.float64, device=device)

        for _ in range(suffix_len):
            if not alive.any():
                break

            masked = suffix.eq(mask_id)  # [bsz, suffix_len]

            x = torch.cat(
                [prompt_row.unsqueeze(0).expand(bsz, -1), suffix],
                dim=1,
            )  # [bsz, prompt_len + suffix_len]

            batched_attn = None
            if attn is not None:
                if attn.shape[0] == bsz:
                    batched_attn = attn
                else:
                    batched_attn = attn.expand(bsz, *attn.shape[1:])

            logits = model(x, attention_mask=batched_attn).logits
            suffix_logits = logits[:, prompt_len:, :]  # [bsz, suffix_len, vocab]

            if temperature <= 0:
                scaled_logits = suffix_logits.float()
            else:
                scaled_logits = suffix_logits.float() / float(temperature)

            bsz_cur, L, vocab_size = scaled_logits.shape
            top_k = min(k, vocab_size)

            # Full-distribution probabilities p_j(v).
            probs = torch.softmax(scaled_logits, dim=-1).to(torch.float64)
            # [bsz, suffix_len, vocab]

            target_ids = target_row.unsqueeze(0).expand(bsz, -1)
            target_probs = torch.gather(
                probs,
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)  # [bsz, suffix_len]

            # Retention condition: target token must be in the top-k under
            # the full distribution at its position.
            _, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
            target_in_topk = (topk_idx == target_ids.unsqueeze(-1)).any(dim=-1)
            # [bsz, suffix_len]

            # Compute full-distribution confidence CDFs:
            #
            #   F_j(c) = sum_{v : p_j(v) < c} p_j(v).
            #
            # We need F_j(target_probs_i) for every pair (j, i).
            # Shape convention below:
            #   j = row/source distribution position
            #   i = candidate revealed position
            sorted_probs = torch.sort(probs, dim=-1).values  # [bsz, L, vocab]
            sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)

            zero_pad = torch.zeros(
                (bsz, L, 1),
                dtype=torch.float64,
                device=device,
            )
            cdf_pad = torch.cat([zero_pad, sorted_cumsum], dim=-1)
            # [bsz, L, vocab + 1]

            # thresholds[b, j, i] = target_probs[b, i].
            thresholds = target_probs.unsqueeze(1).expand(-1, L, -1)
            # [bsz, L, L]

            # Number of entries strictly less than each threshold.
            # right=False gives first index where sorted_probs >= threshold.
            cdf_indices = torch.searchsorted(
                sorted_probs.contiguous(),
                thresholds.contiguous(),
                right=False,
            )
            # [bsz, L, L]

            F_j_i = torch.gather(cdf_pad, dim=-1, index=cdf_indices)
            # [bsz, j, i]

            # log product over j != i, j masked:
            #
            #   sum_{j in M(S), j != i} log F_j(target_prob_i)
            eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
            pair_mask = (
                masked.unsqueeze(2)      # j is masked
                & masked.unsqueeze(1)    # i is masked
                & (~eye)                 # j != i
            )
            # [bsz, j, i]

            log_F_j_i = torch.log(F_j_i)
            log_F_j_i = torch.where(
                pair_mask,
                log_F_j_i,
                torch.zeros_like(log_F_j_i),
            )

            log_confidence_product = log_F_j_i.sum(dim=1)
            # [bsz, i]

            candidate_mask = (
                alive.unsqueeze(-1)
                & masked
                & target_in_topk
                & (target_probs > 0)
            )

            log_target_probs = torch.log(target_probs)

            log_a = log_target_probs + log_confidence_product
            log_a = torch.where(
                candidate_mask,
                log_a,
                torch.full_like(log_a, float("-inf")),
            )
            # [bsz, suffix_len]

            # A^{(k)}(S) = sum_i a_i^{(k)}(S).
            log_A = torch.logsumexp(log_a, dim=-1)  # [bsz]

            has_success_transition = torch.isfinite(log_A)
            active = alive & has_success_transition

            # For active samples, multiply running weight by A^{(k)}(S).
            log_estimate = torch.where(
                active,
                log_estimate + log_A,
                log_estimate,
            )

            # Samples with A^{(k)}(S) = 0 become zero-probability.
            alive = alive & has_success_transition

            if not active.any():
                break

            # Proposal:
            #
            #   q^{(k)}(i | S) = a_i^{(k)}(S) / A^{(k)}(S).
            #
            # Sample on CPU so the provided CPU generator can be used.
            log_q_active = log_a[active] - log_A[active].unsqueeze(-1)
            q_active = torch.exp(log_q_active).detach().cpu()

            sampled_pos_cpu = torch.multinomial(
                q_active,
                num_samples=1,
                replacement=True,
                generator=rng,
            ).squeeze(-1)

            sampled_pos = sampled_pos_cpu.to(device)

            active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)

            # Reveal the sampled successful position by writing the target token.
            sampled_target_tokens = target_row[sampled_pos]
            suffix[active_indices, sampled_pos] = sampled_target_tokens

        batch_log_probs = torch.where(
            alive,
            log_estimate,
            torch.full_like(log_estimate, float("-inf")),
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
        "estimation_method": "path_sampling_low_confidence_full_topk",
    }

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

    if estimation_method == 'path_sampling':
        if normalized_masked_indexes is not None:
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

        path_sampling_result = _path_sampling_low_confidence_full_topk_probability(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            steps=steps,
            attention_mask=attention_mask,
            mask_id=mask_id,
            num_samples=num_samples,
            seed=seed,
            k=k,
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
    Monte Carlo estimate of the probability that the sampler exactly regenerates
    the masked part of `sequence_tokens`, conditioned on the unmasked part.

    Specialization:
      - one token is transferred/kept per step
      - therefore steps must equal the number of masked positions
      - selection rule is: among currently masked positions, keep the sampled
        token with highest confidence under the actual decoding distribution
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if mc_batch_size <= 0:
        raise ValueError("mc_batch_size must be positive")
    if decoding_scheme not in {"top_k", "full"}:
        raise ValueError("decoding_scheme must be 'top_k' or 'full'")
    if decoding_scheme == "top_k" and k <= 0:
        raise ValueError("k must be >= 1 when decoding_scheme == 'top_k'")

    device = _model_device(model)
    sequence_tokens = sequence_tokens.to(device, non_blocking=True)

    if sequence_tokens.ndim != 2 or sequence_tokens.shape[0] != 1:
        raise ValueError(
            f"sequence_tokens must have shape [1, L], got {tuple(sequence_tokens.shape)}"
        )

    seq_len = sequence_tokens.shape[1]

    # Convert 1-indexed -> 0-indexed, deduplicate, sort, validate.
    masked_pos = sorted(set(int(i) - 1 for i in masked_indexes))
    if len(masked_pos) == 0:
        raise ValueError("masked_indexes must be non-empty")
    if any(pos < 0 or pos >= seq_len for pos in masked_pos):
        raise ValueError(
            f"masked_indexes must be 1-indexed positions in [1, {seq_len}]"
        )

    num_masked = len(masked_pos)
    if steps != num_masked:
        raise ValueError(
            "This specialization assumes one token transferred per step, "
            f"so steps must equal the number of masked positions. "
            f"Got steps={steps}, num_masked={num_masked}."
        )

    masked_pos_t = torch.tensor(masked_pos, dtype=torch.long, device=device)   # [M]
    target_masked_tokens = sequence_tokens[0, masked_pos_t]                    # [M]

    # Validate / store base attention mask
    base_attn = None
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)
        if attention_mask.ndim != 2 or attention_mask.shape != (1, seq_len):
            raise ValueError(
                f"attention_mask must have shape [1, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )
        base_attn = attention_mask

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    hits = 0

    for start in range(0, num_samples, mc_batch_size):
        bsz = min(mc_batch_size, num_samples - start)

        # Start from the observed sequence, with the chosen subset masked out.
        x = sequence_tokens.expand(bsz, -1).clone()                            # [B,L]
        x[:, masked_pos_t] = mask_id

        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        attn_batch = None
        if base_attn is not None:
            attn_batch = base_attn.expand(bsz, -1)

        for _step_idx in range(steps):
            if not alive.any():
                break

            logits = model(x, attention_mask=attn_batch).logits                # [B,L,V]

            # Restrict to candidate positions only.
            masked_logits = logits[:, masked_pos_t, :]                         # [B,M,V]

            # Which candidate positions are still masked right now?
            still_masked = (x[:, masked_pos_t] == mask_id)                     # [B,M]

            # Decode from the actual temperature-scaled distribution.
            scaled_masked_logits = masked_logits / temperature                 # [B,M,V]

            if decoding_scheme == "top_k":
                top_k = min(k, scaled_masked_logits.shape[-1])

                top_vals, top_idx = torch.topk(
                    scaled_masked_logits, k=top_k, dim=-1
                )                                                              # [B,M,K]

                top_probs = top_vals.softmax(dim=-1)                           # [B,M,K]

                sampled_local = torch.multinomial(
                    top_probs.reshape(-1, top_k),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, num_masked)                                     # [B,M]

                sampled_tokens = top_idx.gather(
                    dim=-1,
                    index=sampled_local.unsqueeze(-1),
                ).squeeze(-1)                                                  # [B,M]

                # Confidence must match the actual sampling distribution.
                chosen_confidence = top_probs.gather(
                    dim=-1,
                    index=sampled_local.unsqueeze(-1),
                ).squeeze(-1)                                                  # [B,M]

            else:  # full-vocab sampling
                vocab_size = scaled_masked_logits.shape[-1]
                scaled_probs = scaled_masked_logits.softmax(dim=-1)            # [B,M,V]

                sampled_tokens = torch.multinomial(
                    scaled_probs.reshape(-1, vocab_size),
                    num_samples=1,
                    generator=rng,
                ).reshape(bsz, num_masked)                                     # [B,M]

                chosen_confidence = scaled_probs.gather(
                    dim=-1,
                    index=sampled_tokens.unsqueeze(-1),
                ).squeeze(-1)                                                  # [B,M]

            # Only currently masked positions are eligible for transfer.
            neg_inf = torch.full_like(chosen_confidence, float("-inf"))
            confidence = torch.where(still_masked, chosen_confidence, neg_inf) # [B,M]

            selected_slot = confidence.argmax(dim=-1)                          # [B]
            batch_idx = torch.arange(bsz, device=device)

            selected_abs_pos = masked_pos_t[selected_slot]                     # [B]
            selected_token = sampled_tokens[batch_idx, selected_slot]          # [B]
            selected_target = sequence_tokens[0, selected_abs_pos]             # [B]

            # Transfer exactly one token for each active rollout.
            active_idx = batch_idx[alive]
            active_pos = selected_abs_pos[alive]
            active_tok = selected_token[alive]

            x[active_idx, active_pos] = active_tok

            # If the transferred token is wrong, this rollout can no longer
            # exactly match the target sequence.
            mismatch = alive & (selected_token != selected_target)
            alive = alive & (~mismatch)

        if alive.any():
            final_match = (x[:, masked_pos_t] == target_masked_tokens.unsqueeze(0)).all(dim=-1)
            hits += (alive & final_match).sum().item()

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
