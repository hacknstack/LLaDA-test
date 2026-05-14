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
    batch_size: int = 10,
    validate_no_ties: bool = True,
    exact_tail_steps: int = 6,
    proposal: str = "lookahead",  # "local" or "lookahead"
    lookahead_power: float = 1.0,
    topk_exact_children: int = 0,
    rb_tail_depth: int = 0,
) -> Dict[str, object]:
    """
    Unbiased path-sampling estimator for the full-distribution low-confidence
    remasking decoder.

    Implements the estimator for

        p_z = P_{theta, phi}(z_suffix | z_prefix)

    under the full-distribution low-confidence remasking decoder.

    Improvements over the basic implementation:
    1. Computes F_j(c) in log-space using log_softmax + logcumsumexp.
    2. Uses exact-tail Rao-Blackwellization:
           if remaining <= exact_tail_steps,
           compute p_z(S) exactly by dynamic programming over remaining states.
    3. Supports a lookahead proposal:
           q(i | S) proportional to a_i(S) * h_i(S),
       with h_i(S) approximately log A(S union {i}), or exact tail probability
       when the child state is inside the exact tail.
    4. Supports optional top-k Rao-Blackwellization near the tail via
       topk_exact_children and rb_tail_depth.

    Unbiasedness:
    - For proposal="local", this reduces to the original proposal
          q(i | S) = a_i(S) / A(S).
    - For proposal="lookahead", the estimator uses the exact importance ratio
          a_i(S) / q(i | S),
      so it remains unbiased.
    - Exact-tail and top-k Rao-Blackwellization also preserve unbiasedness.

    Practical defaults:
    - exact_tail_steps=6 is usually a good first setting.
    - proposal="lookahead" improves per-sample precision but costs extra model
      calls.
    - topk_exact_children=0 disables top-k branching. Try 1 or 2 with
      rb_tail_depth=1 or 2 if you can afford more computation.
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

    if exact_tail_steps < 0 or exact_tail_steps > suffix_len:
        raise ValueError("exact_tail_steps must satisfy 0 <= exact_tail_steps <= suffix_len.")

    if proposal not in {"local", "lookahead"}:
        raise ValueError('proposal must be either "local" or "lookahead".')

    if lookahead_power < 0:
        raise ValueError("lookahead_power must be nonnegative.")

    if topk_exact_children < 0:
        raise ValueError("topk_exact_children must be nonnegative.")

    if rb_tail_depth < 0:
        raise ValueError("rb_tail_depth must be nonnegative.")

    attn = _suffix_attention_mask(attention_mask, suffix_len, device)

    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    prompt_row = prompt_tokens[0]  # [prompt_len]
    target_row = target_tokens[0]  # [suffix_len]

    eye = torch.eye(suffix_len, dtype=torch.bool, device=device)
    full_revealed_bits = (1 << suffix_len) - 1

    def _logsumexp_floats(values: List[float]) -> float:
        values = [v for v in values if math.isfinite(v)]
        if not values:
            return float("-inf")
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values))

    def _mask_bits_from_revealed_row(revealed_row: torch.Tensor) -> int:
        idxs = torch.nonzero(
            revealed_row.detach().cpu(),
            as_tuple=False,
        ).flatten().tolist()

        bits = 0
        for idx in idxs:
            bits |= 1 << int(idx)
        return bits

    def _state_from_mask_bits(mask_bits: int):
        revealed_vec = torch.tensor(
            [(mask_bits >> i) & 1 for i in range(suffix_len)],
            dtype=torch.bool,
            device=device,
        )

        suffix_vec = torch.full(
            (suffix_len,),
            mask_id,
            dtype=torch.long,
            device=device,
        )

        if revealed_vec.any():
            suffix_vec[revealed_vec] = target_row[revealed_vec]

        return suffix_vec.unsqueeze(0), revealed_vec.unsqueeze(0)

    def _batched_attention(bsz: int):
        if attn is None:
            return None
        if attn.shape[0] == bsz:
            return attn
        return attn.expand(bsz, *attn.shape[1:])

    def _compute_log_a_for_batch(
        suffix: torch.Tensor,
        revealed: torch.Tensor,
        active_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns log_a with shape [bsz, suffix_len].

        log_a[:, i] =
            log p_theta(z_i | x_S, i)
            + sum_{j in M(S), j != i} log F_j(c_i^*(S))

        Already revealed positions receive -inf.
        """

        bsz = suffix.shape[0]
        masked = ~revealed

        if active_rows is None:
            active_rows = torch.ones(bsz, dtype=torch.bool, device=device)

        x = torch.cat(
            [
                prompt_row.unsqueeze(0).expand(bsz, -1),
                suffix,
            ],
            dim=1,
        )

        logits = model(x, attention_mask=_batched_attention(bsz)).logits
        suffix_logits = logits[:, prompt_len:, :]  # [bsz, suffix_len, vocab]

        scaled_logits = suffix_logits.to(torch.float64) / float(temperature)

        # Log-domain distribution.
        log_probs = torch.log_softmax(
            scaled_logits,
            dim=-1,
        )  # [bsz, suffix_len, vocab]

        target_ids = target_row.unsqueeze(0).expand(bsz, -1)

        target_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)  # [bsz, suffix_len]

        sorted_log_probs, _ = torch.sort(log_probs, dim=-1)
        log_cdf = torch.logcumsumexp(sorted_log_probs, dim=-1)

        log_F = torch.empty(
            (bsz, suffix_len, suffix_len),
            dtype=torch.float64,
            device=device,
        )

        tie_flags = None
        if validate_no_ties:
            tie_flags = torch.zeros(
                (bsz, suffix_len, suffix_len),
                dtype=torch.bool,
                device=device,
            )

        for j in range(suffix_len):
            sorted_j = sorted_log_probs[:, j, :].contiguous()
            log_cdf_j = log_cdf[:, j, :]

            # Strict CDF:
            # F_j(c_i) = P_{V_j}[log p_j(V_j) < log c_i]
            left_idx = torch.searchsorted(
                sorted_j,
                target_log_probs,
                right=False,
            )  # [bsz, suffix_len]

            left_gather_idx = (left_idx - 1).clamp_min(0)

            log_F_less = torch.gather(
                log_cdf_j,
                dim=-1,
                index=left_gather_idx,
            )

            log_F_less = torch.where(
                left_idx > 0,
                log_F_less,
                torch.full_like(log_F_less, float("-inf")),
            )

            log_F[:, j, :] = log_F_less

            if validate_no_ties:
                # A positive-probability tie exists when some token at j has
                # exactly the same confidence as c_i^*, and c_i^* > 0.
                right_idx = torch.searchsorted(
                    sorted_j,
                    target_log_probs,
                    right=True,
                )

                positive_confidence = torch.isfinite(target_log_probs)
                tie_flags[:, j, :] = (right_idx > left_idx) & positive_confidence

        if validate_no_ties:
            tie_competitor_mask = (
                masked.unsqueeze(-1)      # j is masked
                & masked.unsqueeze(1)     # i is masked
                & (~eye.unsqueeze(0))     # j != i
                & active_rows.view(bsz, 1, 1)
            )

            detected_tie = (tie_competitor_mask & tie_flags).any()

            if bool(detected_tie.item()):
                raise ValueError(
                    "Detected a positive-probability confidence tie. "
                    "The strict no-ties formula uses "
                    "F_j(c) = P(confidence < c). To estimate the decoder under "
                    "ties, implement the decoder's exact tie-breaking rule "
                    "inside a_i(S)."
                )

        include_j = masked.unsqueeze(-1) & (~eye.unsqueeze(0))

        log_product = torch.where(
            include_j,
            log_F,
            torch.zeros_like(log_F),
        ).sum(dim=1)  # [bsz, suffix_len]

        log_a = target_log_probs + log_product

        log_a = torch.where(
            masked,
            log_a,
            torch.full_like(log_a, float("-inf")),
        )

        return log_a

    exact_cache: Dict[int, float] = {}

    def _exact_log_p_from_mask_bits(mask_bits: int) -> float:
        """
        Exact dynamic-programming computation of

            p_z(S) = sum_i a_i(S) p_z(S union {i})

        for a small number of remaining masked positions.
        """

        if mask_bits == full_revealed_bits:
            return 0.0

        cached = exact_cache.get(mask_bits)
        if cached is not None:
            return cached

        suffix, revealed = _state_from_mask_bits(mask_bits)
        log_a = _compute_log_a_for_batch(suffix, revealed)[0]

        terms: List[float] = []

        for i in range(suffix_len):
            if (mask_bits >> i) & 1:
                continue

            lai = float(log_a[i].detach().cpu().item())
            if not math.isfinite(lai):
                continue

            child_bits = mask_bits | (1 << i)
            child_log_p = _exact_log_p_from_mask_bits(child_bits)

            if math.isfinite(child_log_p):
                terms.append(lai + child_log_p)

        out = _logsumexp_floats(terms)
        exact_cache[mask_bits] = out
        return out

    def _compute_log_h_lookahead_for_batch(
        suffix: torch.Tensor,
        revealed: torch.Tensor,
        log_a: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate future-success score h_i(S).

        If child state is inside the exact tail, use exact log p_z(S union {i}).
        Otherwise use log A(S union {i}).
        """

        bsz = suffix.shape[0]
        masked = ~revealed

        log_h = torch.full_like(log_a, float("-inf"))

        for i in range(suffix_len):
            rows = active_rows & masked[:, i] & torch.isfinite(log_a[:, i])

            if not bool(rows.any().item()):
                continue

            child_suffix = suffix[rows].clone()
            child_revealed = revealed[rows].clone()

            child_suffix[:, i] = target_row[i]
            child_revealed[:, i] = True

            child_remaining = int((~child_revealed[0]).sum().detach().cpu().item())

            if child_remaining <= exact_tail_steps:
                vals: List[float] = []

                for r in range(child_revealed.shape[0]):
                    child_bits = _mask_bits_from_revealed_row(child_revealed[r])
                    vals.append(_exact_log_p_from_mask_bits(child_bits))

                log_h[rows, i] = torch.tensor(
                    vals,
                    dtype=torch.float64,
                    device=device,
                )
            else:
                child_log_a = _compute_log_a_for_batch(child_suffix, child_revealed)
                child_log_A = torch.logsumexp(child_log_a, dim=-1)
                log_h[rows, i] = child_log_A

        return log_h

    def _make_log_proposal_for_batch(
        suffix: torch.Tensor,
        revealed: torch.Tensor,
        log_a: torch.Tensor,
        active_rows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns log q(i | S) with shape [bsz, suffix_len].

        For proposal="local":
            q(i | S) proportional to a_i(S).

        For proposal="lookahead":
            q(i | S) proportional to a_i(S) * h_i(S)^lookahead_power.
        """

        log_q = torch.full_like(log_a, float("-inf"))
        log_A = torch.logsumexp(log_a, dim=-1)

        live = active_rows & torch.isfinite(log_A)

        if not bool(live.any().item()):
            return log_q

        if proposal == "local" or lookahead_power == 0:
            log_q[live] = log_a[live] - log_A[live].unsqueeze(-1)
            return log_q

        log_h = _compute_log_h_lookahead_for_batch(
            suffix=suffix,
            revealed=revealed,
            log_a=log_a,
            active_rows=live,
        )

        log_unnorm = log_a + float(lookahead_power) * log_h
        log_Z = torch.logsumexp(log_unnorm, dim=-1)

        # If the heuristic kills all candidates on a row, fall back to local q.
        fallback = live & (~torch.isfinite(log_Z))

        if bool(fallback.any().item()):
            log_unnorm[fallback] = log_a[fallback]
            log_Z[fallback] = log_A[fallback]

        live = live & torch.isfinite(log_Z)

        if bool(live.any().item()):
            log_q[live] = log_unnorm[live] - log_Z[live].unsqueeze(-1)

        return log_q

    def _compute_log_a_single(mask_bits: int) -> torch.Tensor:
        suffix, revealed = _state_from_mask_bits(mask_bits)
        return _compute_log_a_for_batch(suffix, revealed)[0]

    def _single_log_proposal(
        mask_bits: int,
        log_a: torch.Tensor,
        allowed_indices: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        allowed_set = None
        if allowed_indices is not None:
            allowed_set = set(int(i) for i in allowed_indices)

        valid: List[int] = []
        for i in range(suffix_len):
            if (mask_bits >> i) & 1:
                continue
            if allowed_set is not None and i not in allowed_set:
                continue
            lai = float(log_a[i].detach().cpu().item())
            if math.isfinite(lai):
                valid.append(i)

        if not valid:
            return {}

        if proposal == "local" or lookahead_power == 0:
            scores = {
                i: float(log_a[i].detach().cpu().item())
                for i in valid
            }
        else:
            scores = {}

            for i in valid:
                child_bits = mask_bits | (1 << i)
                child_remaining = suffix_len - child_bits.bit_count()

                if child_remaining <= exact_tail_steps:
                    log_h_i = _exact_log_p_from_mask_bits(child_bits)
                else:
                    child_log_a = _compute_log_a_single(child_bits)
                    log_h_i = float(
                        torch.logsumexp(child_log_a, dim=-1)
                        .detach()
                        .cpu()
                        .item()
                    )

                lai = float(log_a[i].detach().cpu().item())

                if math.isfinite(log_h_i):
                    scores[i] = lai + float(lookahead_power) * log_h_i
                else:
                    scores[i] = float("-inf")

            if not any(math.isfinite(v) for v in scores.values()):
                scores = {
                    i: float(log_a[i].detach().cpu().item())
                    for i in valid
                }

        log_Z = _logsumexp_floats(list(scores.values()))

        if not math.isfinite(log_Z):
            return {}

        return {
            i: scores[i] - log_Z
            for i in valid
            if math.isfinite(scores[i])
        }

    def _sample_from_log_q(log_q_dict: Dict[int, float]) -> int:
        if not log_q_dict:
            raise RuntimeError("Cannot sample from an empty proposal.")

        indices = list(log_q_dict.keys())
        logps = torch.tensor(
            [log_q_dict[i] for i in indices],
            dtype=torch.float64,
            device="cpu",
        )

        probs = torch.exp(logps)
        probs = probs / probs.sum().clamp_min(torch.finfo(probs.dtype).tiny)

        chosen_pos = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
            generator=rng,
        ).item()

        return int(indices[chosen_pos])

    def _single_sample_log_estimate_from_mask_bits(mask_bits: int) -> float:
        """
        Single unbiased continuation estimate from a state S.
        Used inside optional top-k Rao-Blackwellization.
        """

        log_w = 0.0
        current_bits = mask_bits

        while current_bits != full_revealed_bits:
            remaining = suffix_len - current_bits.bit_count()

            if remaining <= exact_tail_steps:
                tail = _exact_log_p_from_mask_bits(current_bits)
                return log_w + tail

            log_a = _compute_log_a_single(current_bits)
            log_q_dict = _single_log_proposal(current_bits, log_a)

            if not log_q_dict:
                return float("-inf")

            j = _sample_from_log_q(log_q_dict)

            lai = float(log_a[j].detach().cpu().item())
            lqj = log_q_dict[j]

            if not math.isfinite(lai) or not math.isfinite(lqj):
                return float("-inf")

            log_w += lai - lqj
            current_bits |= 1 << j

        return log_w

    def _rb_log_estimate_from_mask_bits(mask_bits: int, depth: int) -> float:
        """
        Optional top-k Rao-Blackwellized continuation estimator.

        At each state:
        - exactly branch over topk_exact_children high-a_i children;
        - sample one child from the complement using the chosen proposal;
        - recurse for at most `depth` top-k levels;
        - fall back to single-sample continuation afterward.

        This remains unbiased but can become expensive if depth and top-k are large.
        """

        if mask_bits == full_revealed_bits:
            return 0.0

        remaining = suffix_len - mask_bits.bit_count()

        if remaining <= exact_tail_steps:
            return _exact_log_p_from_mask_bits(mask_bits)

        if depth <= 0 or topk_exact_children <= 0:
            return _single_sample_log_estimate_from_mask_bits(mask_bits)

        log_a = _compute_log_a_single(mask_bits)

        valid: List[int] = []
        for i in range(suffix_len):
            if (mask_bits >> i) & 1:
                continue
            lai = float(log_a[i].detach().cpu().item())
            if math.isfinite(lai):
                valid.append(i)

        if not valid:
            return float("-inf")

        valid_sorted = sorted(
            valid,
            key=lambda i: float(log_a[i].detach().cpu().item()),
            reverse=True,
        )

        k = min(topk_exact_children, len(valid_sorted))
        top_indices = valid_sorted[:k]
        rest_indices = valid_sorted[k:]

        terms: List[float] = []

        for i in top_indices:
            lai = float(log_a[i].detach().cpu().item())
            child_bits = mask_bits | (1 << i)
            child_est = _rb_log_estimate_from_mask_bits(child_bits, depth - 1)

            if math.isfinite(lai) and math.isfinite(child_est):
                terms.append(lai + child_est)

        if rest_indices:
            rest_log_q = _single_log_proposal(
                mask_bits=mask_bits,
                log_a=log_a,
                allowed_indices=rest_indices,
            )

            if rest_log_q:
                j = _sample_from_log_q(rest_log_q)

                lai = float(log_a[j].detach().cpu().item())
                lqj = rest_log_q[j]

                child_bits = mask_bits | (1 << j)
                child_est = _rb_log_estimate_from_mask_bits(child_bits, depth - 1)

                if (
                    math.isfinite(lai)
                    and math.isfinite(lqj)
                    and math.isfinite(child_est)
                ):
                    terms.append(lai - lqj + child_est)

        return _logsumexp_floats(terms)

    sample_log_probabilities: List[float] = []
    sample_probabilities: List[float] = []

    rb_extra_depth = rb_tail_depth if topk_exact_children > 0 else 0
    stop_threshold = exact_tail_steps + rb_extra_depth

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

        log_weight = torch.zeros(bsz, dtype=torch.float64, device=device)
        alive = torch.ones(bsz, dtype=torch.bool, device=device)

        for step_idx in range(suffix_len):
            remaining_count = suffix_len - step_idx

            # Exact / Rao-Blackwellized tail.
            if stop_threshold > 0 and remaining_count <= stop_threshold:
                tail_logs = torch.full(
                    (bsz,),
                    float("-inf"),
                    dtype=torch.float64,
                    device=device,
                )

                depth_here = min(
                    rb_extra_depth,
                    max(remaining_count - exact_tail_steps, 0),
                )

                for b in range(bsz):
                    if not bool(alive[b].item()):
                        continue

                    bits = _mask_bits_from_revealed_row(revealed[b])

                    if depth_here > 0 and topk_exact_children > 0:
                        val = _rb_log_estimate_from_mask_bits(bits, depth_here)
                    else:
                        val = _exact_log_p_from_mask_bits(bits)

                    tail_logs[b] = val

                log_weight = torch.where(
                    alive,
                    log_weight + tail_logs,
                    log_weight,
                )

                alive = alive & torch.isfinite(tail_logs)
                break

            masked = ~revealed

            log_a = _compute_log_a_for_batch(
                suffix=suffix,
                revealed=revealed,
                active_rows=alive,
            )

            log_A = torch.logsumexp(log_a, dim=-1)
            can_continue = alive & torch.isfinite(log_A)

            log_q = _make_log_proposal_for_batch(
                suffix=suffix,
                revealed=revealed,
                log_a=log_a,
                active_rows=can_continue,
            )

            q = torch.zeros_like(log_a, dtype=torch.float64)

            if bool(can_continue.any().item()):
                q[can_continue] = torch.exp(log_q[can_continue])

            dead = ~can_continue
            if bool(dead.any().item()):
                dummy_probs = masked[dead].to(torch.float64)
                dummy_probs = dummy_probs / dummy_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(torch.finfo(torch.float64).tiny)
                q[dead] = dummy_probs

            q = torch.where(masked, q, torch.zeros_like(q))

            q_sum = q.sum(dim=-1, keepdim=True)
            q = q / q_sum.clamp_min(torch.finfo(q.dtype).tiny)

            next_indices_cpu = torch.multinomial(
                q.detach().cpu(),
                num_samples=1,
                replacement=True,
                generator=rng,
            ).squeeze(-1)

            next_indices = next_indices_cpu.to(device)

            chosen_log_a = torch.gather(
                log_a,
                dim=1,
                index=next_indices.unsqueeze(-1),
            ).squeeze(-1)

            chosen_log_q = torch.gather(
                log_q,
                dim=1,
                index=next_indices.unsqueeze(-1),
            ).squeeze(-1)

            increment = chosen_log_a - chosen_log_q

            log_weight = torch.where(
                can_continue,
                log_weight + increment,
                log_weight,
            )

            alive = can_continue

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
                src=torch.ones((bsz, 1), dtype=torch.bool, device=device),
            )

        batch_log_probs = torch.where(
            alive,
            log_weight,
            torch.full_like(log_weight, float("-inf")),
        )

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

    finite_logs = [
        lp for lp in sample_log_probabilities
        if math.isfinite(lp)
    ]

    if not finite_logs:
        log_average_probability = float("-inf")
        average_probability = 0.0
    else:
        max_log = max(finite_logs)
        scaled_sum = sum(
            math.exp(lp - max_log)
            for lp in finite_logs
        )

        log_average_probability = (
            max_log
            + math.log(scaled_sum)
            - math.log(len(sample_log_probabilities))
        )

        try:
            average_probability = float(math.exp(log_average_probability))
        except OverflowError:
            average_probability = float("inf")

    return {
        "probability": average_probability,
        "log_probability": log_average_probability,
        "sample_probabilities": sample_probabilities,
        "sample_log_probabilities": sample_log_probabilities,
        "num_samples": num_samples,
        "estimation_method": "path_sampling_low_confidence",
        "decoding_scheme": "full",
        "temperature": temperature,
        "validated_no_ties": validate_no_ties,
        "proposal": proposal,
        "lookahead_power": lookahead_power,
        "exact_tail_steps": exact_tail_steps,
        "topk_exact_children": topk_exact_children,
        "rb_tail_depth": rb_tail_depth,
        "exact_tail_cache_size": len(exact_cache),
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
