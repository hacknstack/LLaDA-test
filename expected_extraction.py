#!/usr/bin/env python3
from typing import List, Optional

import torch
import torch.nn.functional as F

MASK_ID = 126336


def _sample_next_token(
    logits: torch.Tensor,
    decoding_scheme: str,
    k: int,
    temperature: float,
    generator: Optional[torch.Generator],
) -> int:
    scaled_logits = logits / temperature

    if decoding_scheme == 'top_k':
        top_k = min(k, scaled_logits.shape[-1])
        topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        sampled_rel = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        return int(topk_idx[sampled_rel].item())

    probs = F.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _target_probability(logits: torch.Tensor, target_id: int, decoding_scheme: str, k: int, temperature: float) -> float:
    scaled_logits = logits / temperature
    if decoding_scheme == 'top_k':
        top_k = min(k, scaled_logits.shape[-1])
        topk_vals, topk_idx = torch.topk(scaled_logits, k=top_k, dim=-1)
        in_topk = bool((topk_idx == target_id).any().item())
        if not in_topk:
            return 0.0
        selected_logit = scaled_logits[target_id]
        log_denom = torch.logsumexp(topk_vals, dim=-1)
        return float(torch.exp(selected_logit - log_denom).item())

    return float(F.softmax(scaled_logits, dim=-1)[target_id].item())


@torch.no_grad()
def _compute_llama_rb_sampling_expectation(
    model,
    prefix_ids: List[int],
    suffix_ids: List[int],
    decoding_scheme: str,
    num_samples: int,
    k: int,
    temperature: float,
    seed: Optional[int],
) -> float:
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix = torch.tensor(suffix_ids, dtype=torch.long, device=device)

    if prefix.numel() == 0:
        raise ValueError('prefix_ids must contain at least one token for llama generation.')

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sample_totals = []
    for _ in range(num_samples):
        running_sum = 0.0
        sampled_suffix: List[int] = []

        for t in range(suffix.numel()):
            context = prefix if not sampled_suffix else torch.cat([prefix, torch.tensor(sampled_suffix, dtype=torch.long, device=device)], dim=0)
            logits = model(context.unsqueeze(0)).logits[0, -1]
            target_id = int(suffix[t].item())

            running_sum += _target_probability(
                logits=logits,
                target_id=target_id,
                decoding_scheme=decoding_scheme,
                k=k,
                temperature=temperature,
            )
            sampled_suffix.append(
                _sample_next_token(
                    logits=logits,
                    decoding_scheme=decoding_scheme,
                    k=k,
                    temperature=temperature,
                    generator=generator,
                )
            )

        sample_totals.append(running_sum)

    return float(sum(sample_totals) / num_samples)


@torch.no_grad()
def _compute_llada_rb_sampling_expectation(
    model,
    prefix_ids: List[int],
    suffix_ids: List[int],
    decoding_scheme: str,
    num_samples: int,
    k: int,
    temperature: float,
    seed: Optional[int],
    mask_id: int = MASK_ID,
) -> float:
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    prompt = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    suffix_len = int(suffix.numel())

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sample_totals = []
    for _ in range(num_samples):
        running_sum = 0.0
        x = torch.full((prompt.numel() + suffix_len,), mask_id, dtype=torch.long, device=device)
        x[:prompt.numel()] = prompt

        order = torch.randperm(suffix_len, generator=generator, device=device).tolist()
        for pos in order:
            logits = model(x.unsqueeze(0)).logits[0, prompt.numel() + pos]
            target_id = int(suffix[pos].item())

            running_sum += _target_probability(
                logits=logits,
                target_id=target_id,
                decoding_scheme=decoding_scheme,
                k=k,
                temperature=temperature,
            )

            sampled_id = _sample_next_token(
                logits=logits,
                decoding_scheme=decoding_scheme,
                k=k,
                temperature=temperature,
                generator=generator,
            )
            x[prompt.numel() + pos] = sampled_id

        sample_totals.append(running_sum)

    return float(sum(sample_totals) / num_samples)


@torch.no_grad()
def _compute_expectation(model, prefix_ids: List[int], suffix_ids: List[int], args) -> float:
    """Expected number of suffix-token matches under RB sampling."""
    if args.mode != 'RB_sampling':
        raise ValueError('For expected extraction, --mode must be "RB_sampling".')

    decoding_scheme = args.decoding_scheme if args.decoding_scheme != 'auto' else ('top_k' if args.model_family == 'llama' else 'full')
    if decoding_scheme not in {'full', 'top_k'}:
        raise ValueError('For expected extraction, decoding_scheme must be one of {"full", "top_k"}.')

    num_samples = int(args.num_samples)
    if num_samples <= 0:
        raise ValueError('num_samples must be > 0 for RB sampling.')
    if decoding_scheme == 'top_k' and int(args.k) <= 0:
        raise ValueError('k must be > 0 when decoding_scheme="top_k".')
    if float(args.temperature) <= 0:
        raise ValueError('temperature must be > 0 for RB sampling.')

    if args.model_family == 'llama':
        return _compute_llama_rb_sampling_expectation(
            model=model,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            decoding_scheme=decoding_scheme,
            num_samples=num_samples,
            k=int(args.k),
            temperature=float(args.temperature),
            seed=args.seed,
        )

    if args.model_family == 'llada':
        if args.remasking != 'random':
            raise ValueError('For LLaDA expected extraction, --remasking must be "random".')
        return _compute_llada_rb_sampling_expectation(
            model=model,
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            decoding_scheme=decoding_scheme,
            num_samples=num_samples,
            k=int(args.k),
            temperature=float(args.temperature),
            seed=args.seed,
        )

    raise ValueError('model_family must be one of {"llama", "llada"}.')
