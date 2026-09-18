#!/usr/bin/env python3
"""Colab benchmark for fast-dLLM STS and direct Monte Carlo."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from probabilistic_extraction import (
    _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked,
    _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
)


MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=Path("texts/AliceInWonderlandChapter1.txt"))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument(
        "--window-kind",
        choices=[
            "alice-suffix", "alice-alternating",
            "mit-suffix", "mit-alternating",
            "repeat-eos-suffix", "repeat-eos-alternating",
            "repeat-the-suffix", "repeat-the-alternating",
        ],
        default="alice-suffix",
    )
    parser.add_argument("--sts-samples", type=int, default=500)
    parser.add_argument("--mc-samples", type=int, default=5000)
    parser.add_argument("--large-sts-samples", type=int, default=5000)
    parser.add_argument("--large-mc-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--window-token-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("fast_dllm_benchmark_results.json"))
    return parser.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sts_summary(result):
    logs = torch.tensor(result["sample_log_probabilities"], dtype=torch.float64)
    n = logs.numel()
    log_mean = torch.logsumexp(logs, dim=0) - math.log(n)
    log_second = torch.logsumexp(2.0 * logs, dim=0) - math.log(n)
    mean = float(torch.exp(log_mean))
    second = float(torch.exp(log_second))
    variance = max(0.0, second - mean * mean)
    se = math.sqrt(variance / n)
    return {
        "estimate": mean,
        "log_estimate": float(log_mean),
        "standard_error": se,
        "ci95": [max(0.0, mean - 1.96 * se), mean + 1.96 * se],
        "relative_standard_error": None if mean == 0.0 else se / mean,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; set the Colab NVIDIA library path first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = args.text.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if args.window_token_offset < 0:
        raise RuntimeError("--window-token-offset must be non-negative")
    window_end = args.window_token_offset + 100
    if len(token_ids) < window_end:
        raise RuntimeError(
            f"Need at least {window_end} tokens for this offset, found {len(token_ids)}"
        )
    if args.window_kind.startswith(("alice-", "mit-")):
        window_ids = token_ids[args.window_token_offset:window_end]
    elif args.window_kind.startswith("repeat-eos-"):
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no EOS token.")
        window_ids = [tokenizer.eos_token_id] * 100
    else:
        repeated = tokenizer(" the", add_special_tokens=False)["input_ids"]
        if len(repeated) != 1:
            raise RuntimeError(f"Expected ' the' to be one token, got {repeated}")
        window_ids = repeated * 100
    sequence = torch.tensor([window_ids], dtype=torch.long)
    masked_indexes = (
        list(range(51, 101))
        if args.window_kind.endswith("-suffix")
        else list(range(2, 101, 2))
    )

    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    sync()
    load_seconds = time.perf_counter() - load_start

    forward_counter = {"count": 0}

    def count_forward(_module, _inputs):
        forward_counter["count"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    common = dict(
        model=model,
        sequence_tokens=sequence,
        masked_indexes=masked_indexes,
        steps=50,
        attention_mask=None,
        mask_id=MASK_ID,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
    )

    # Warm kernels and remote-code paths; do not include this in timings.
    _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked(
        **common, num_samples=2, seed=args.seed - 1, batch_size=2,
        return_samples=False,
    )
    _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked(
        **common, num_samples=16, seed=args.seed - 1,
        decoding_scheme="full", k=1, mc_batch_size=16,
    )
    sync()

    results = {
        "configuration": {
            "model": args.model,
            "text": str(args.text),
            "window_kind": args.window_kind,
            "window_token_offset": args.window_token_offset,
            "visible_positions": (
                [1, 50] if args.window_kind.endswith("-suffix")
                else "odd positions 1..99"
            ),
            "masked_positions": (
                [51, 100] if args.window_kind.endswith("-suffix")
                else "even positions 2..100"
            ),
            "temperature": args.temperature,
            "confidence_threshold": args.confidence_threshold,
            "seed": args.seed,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model_load_seconds": load_seconds,
        },
        "runs": {},
    }

    def timed_sts(label, samples, seed):
        forward_counter["count"] = 0
        torch.cuda.reset_peak_memory_stats()
        base_memory = torch.cuda.memory_allocated()
        sync()
        start = time.perf_counter()
        result = _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked(
            **common,
            num_samples=samples,
            seed=seed,
            batch_size=1024,
            return_samples=True,
        )
        sync()
        elapsed = time.perf_counter() - start
        summary = sts_summary(result)
        summary.update({
            "samples": samples,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "model_forward_calls": forward_counter["count"],
            "state_cache_hits": result["state_cache_hits"],
            "state_cache_misses": result["state_cache_misses"],
            "peak_incremental_memory_gib": (
                torch.cuda.max_memory_allocated() - base_memory
            ) / 2**30,
        })
        results["runs"][label] = summary
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({label: summary}), flush=True)

    def timed_mc(label, samples, seed):
        forward_counter["count"] = 0
        torch.cuda.reset_peak_memory_stats()
        base_memory = torch.cuda.memory_allocated()
        sync()
        start = time.perf_counter()
        result = _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked(
            **common,
            num_samples=samples,
            seed=seed,
            decoding_scheme="full",
            k=1,
            mc_batch_size=16384,
        )
        sync()
        elapsed = time.perf_counter() - start
        summary = {
            "samples": samples,
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "ci95_wald": list(result.wald_ci),
            "ci95_wilson": list(result.wilson_ci),
            "relative_standard_error": (
                None if result.estimate == 0.0
                else result.standard_error / result.estimate
            ),
            "hits": result.hits,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "model_forward_calls": forward_counter["count"],
            "peak_incremental_memory_gib": (
                torch.cuda.max_memory_allocated() - base_memory
            ) / 2**30,
        }
        results["runs"][label] = summary
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({label: summary}), flush=True)

    timed_sts("sts_baseline", args.sts_samples, args.seed)
    timed_mc("mc_baseline", args.mc_samples, args.seed)
    timed_sts("sts_large", args.large_sts_samples, args.seed + 1)
    timed_mc("mc_large", args.large_mc_samples, args.seed + 1)

    for suffix in ("baseline", "large"):
        sts = results["runs"][f"sts_{suffix}"]
        mc = results["runs"][f"mc_{suffix}"]
        combined_se = math.hypot(sts["standard_error"], mc["standard_error"])
        results[f"agreement_{suffix}"] = {
            "absolute_difference": abs(sts["estimate"] - mc["estimate"]),
            "combined_standard_error": combined_se,
            "difference_in_combined_se": (
                None if combined_se == 0.0
                else abs(sts["estimate"] - mc["estimate"]) / combined_se
            ),
            "ci_overlap": not (
                sts["ci95"][1] < mc["ci95_wilson"][0]
                or mc["ci95_wilson"][1] < sts["ci95"][0]
            ),
        }

    hook.remove()
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
