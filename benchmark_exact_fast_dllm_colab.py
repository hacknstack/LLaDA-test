#!/usr/bin/env python3
"""Benchmark exact, STS, and MC fast-dLLM estimators on a 90/10 window."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from probabilistic_extraction import (
    _exact_fast_dllm_threshold_probability_dp_from_partially_masked,
    _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked,
    _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
)


MASK_ID = 126336


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=Path("texts/MITLicense.txt"))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--sts-samples", type=int, default=500)
    parser.add_argument("--mc-samples", type=int, default=5000)
    parser.add_argument("--large-sts-samples", type=int, default=5000)
    parser.add_argument("--large-mc-samples", type=int, default=50000)
    parser.add_argument("--exact-reference", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output", type=Path, default=Path("exact_fast_dllm_benchmark_mit_90_10.json"),
    )
    return parser.parse_args()


def sync():
    torch.cuda.synchronize()


def sts_summary(result):
    logs = torch.tensor(result["sample_log_probabilities"], dtype=torch.float64)
    count = logs.numel()
    log_mean = torch.logsumexp(logs, dim=0) - math.log(count)
    log_second = torch.logsumexp(2 * logs, dim=0) - math.log(count)
    mean = float(torch.exp(log_mean))
    variance = max(0.0, float(torch.exp(log_second)) - mean * mean)
    standard_error = math.sqrt(variance / count)
    return {
        "estimate": mean,
        "log_estimate": float(log_mean),
        "standard_error": standard_error,
        "ci95": [
            max(0.0, mean - 1.96 * standard_error),
            min(1.0, mean + 1.96 * standard_error),
        ],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids = tokenizer(
        args.text.read_text(encoding="utf-8", errors="replace"),
        add_special_tokens=False,
    )["input_ids"]
    if len(token_ids) < 100:
        raise RuntimeError(f"Need 100 tokens, found {len(token_ids)}")
    sequence = torch.tensor([token_ids[:100]], dtype=torch.long)
    masked_indexes = list(range(91, 101))

    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    sync()
    load_seconds = time.perf_counter() - load_start
    forward_count = {"value": 0}

    def count_forward(_module, _inputs):
        forward_count["value"] += 1

    hook = model.register_forward_pre_hook(count_forward)
    common = dict(
        model=model,
        sequence_tokens=sequence,
        masked_indexes=masked_indexes,
        steps=10,
        attention_mask=None,
        mask_id=MASK_ID,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
    )

    # Warm model and sampling kernels without retaining estimator state.
    _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked(
        **common, num_samples=2, seed=args.seed - 1, batch_size=2,
        return_samples=False,
    )
    _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked(
        **common, num_samples=16, seed=args.seed - 1, decoding_scheme="full",
        k=1, mc_batch_size=16,
    )
    sync()

    results = {
        "configuration": {
            "model": args.model,
            "text": str(args.text),
            "window_token_offset": 0,
            "visible_positions": [1, 90],
            "masked_positions": [91, 100],
            "temperature": args.temperature,
            "confidence_threshold": args.confidence_threshold,
            "seed": args.seed,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model_load_seconds": load_seconds,
        },
        "runs": {},
    }

    def persist():
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.exact_reference is None:
        forward_count["value"] = 0
        torch.cuda.reset_peak_memory_stats()
        base_memory = torch.cuda.memory_allocated()
        sync()
        started = time.perf_counter()
        exact = _exact_fast_dllm_threshold_probability_dp_from_partially_masked(**common)
        sync()
        elapsed = time.perf_counter() - started
        exact_summary = {
            **exact,
            "seconds": elapsed,
            "measured_model_forward_calls": forward_count["value"],
            "peak_incremental_memory_gib": (
                torch.cuda.max_memory_allocated() - base_memory
            ) / 2**30,
        }
    else:
        exact = {
            "probability": args.exact_reference,
            "log_probability": math.log(args.exact_reference),
        }
        exact_summary = {**exact, "provided_reference": True}
    results["runs"]["exact"] = exact_summary
    persist()
    print(json.dumps({"exact": exact_summary}), flush=True)

    def timed_sts(label, samples, seed):
        forward_count["value"] = 0
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        sync()
        started = time.perf_counter()
        value = _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked(
            **common, num_samples=samples, seed=seed, batch_size=1024,
            return_samples=True,
        )
        sync()
        elapsed = time.perf_counter() - started
        summary = sts_summary(value)
        summary.update({
            "samples": samples,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "model_forward_calls": forward_count["value"],
            "state_cache_hits": value["state_cache_hits"],
            "state_cache_misses": value["state_cache_misses"],
            "peak_incremental_memory_gib": (
                torch.cuda.max_memory_allocated() - base
            ) / 2**30,
        })
        results["runs"][label] = summary
        persist()
        print(json.dumps({label: summary}), flush=True)

    def timed_mc(label, samples, seed):
        forward_count["value"] = 0
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        sync()
        started = time.perf_counter()
        value = _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked(
            **common, num_samples=samples, seed=seed, decoding_scheme="full",
            k=1, mc_batch_size=16384,
        )
        sync()
        elapsed = time.perf_counter() - started
        summary = {
            "estimate": value.estimate,
            "standard_error": value.standard_error,
            "ci95_wald": list(value.wald_ci),
            "ci95_wilson": list(value.wilson_ci),
            "hits": value.hits,
            "samples": samples,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "model_forward_calls": forward_count["value"],
            "peak_incremental_memory_gib": (
                torch.cuda.max_memory_allocated() - base
            ) / 2**30,
        }
        results["runs"][label] = summary
        persist()
        print(json.dumps({label: summary}), flush=True)

    if args.sts_samples > 0:
        timed_sts("sts_baseline", args.sts_samples, args.seed)
    if args.mc_samples > 0:
        timed_mc("mc_baseline", args.mc_samples, args.seed)
    if args.large_sts_samples > 0:
        timed_sts("sts_large", args.large_sts_samples, args.seed + 1)
    if args.large_mc_samples > 0:
        timed_mc("mc_large", args.large_mc_samples, args.seed + 1)

    exact_probability = exact["probability"]
    for label, run in results["runs"].items():
        if label == "exact":
            continue
        run["absolute_error_vs_exact"] = abs(run["estimate"] - exact_probability)
        standard_error = run["standard_error"]
        run["error_in_standard_errors"] = (
            None if standard_error == 0 else run["absolute_error_vs_exact"] / standard_error
        )
    persist()
    hook.remove()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
