#!/usr/bin/env python3
"""Compare standard and --fast STS on a 100-token, 50-mask A100 window."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from probabilistic_extraction import (
    _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
    _path_sampling_low_confidence_probability_fast_from_partially_masked,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=Path("texts/MITLicense.txt"))
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--remasking", choices=["low-confidence", "fast-dllm"],
                        default="low-confidence")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fast-only", action="store_true")
    parser.add_argument("--profile-forward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer(args.text.read_text(encoding="utf-8", errors="replace"),
                    add_special_tokens=False)["input_ids"]
    if len(ids) < 100:
        raise ValueError(f"Need 100 tokens, got {len(ids)}")
    sequence = torch.tensor([ids[:100]], dtype=torch.long)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    forward_events = []
    if args.profile_forward:
        original_forward = model.forward
        def timed_forward(*forward_args, **forward_kwargs):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = original_forward(*forward_args, **forward_kwargs)
            finished.record()
            forward_events.append((started, finished))
            return output
        model.forward = timed_forward
    estimator = (
        _path_sampling_low_confidence_probability_fast_from_partially_masked
        if args.remasking == "low-confidence" else
        _path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked
    )
    common = dict(
        model=model, sequence_tokens=sequence,
        masked_indexes=list(range(51, 101)), steps=50,
        attention_mask=None, mask_id=126336, num_samples=args.samples,
        seed=args.seed, temperature=1.0, batch_size=1024,
        return_samples=True,
    )
    if args.remasking == "fast-dllm":
        common["confidence_threshold"] = args.confidence_threshold

    # The one-state path also checks that model hooks support active projection.
    estimator(**{**common, "num_samples": 1}, fast=True)
    torch.cuda.synchronize()
    forward_events.clear()
    runs = {}
    for fast in ((True,) if args.fast_only else (False, True)):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = estimator(**common, fast=fast)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logs = torch.tensor(result["sample_log_probabilities"], dtype=torch.float64)
        log_mean = torch.logsumexp(logs, dim=0) - math.log(args.samples)
        second = torch.exp(torch.logsumexp(2 * logs, dim=0) - math.log(args.samples))
        mean = torch.exp(log_mean)
        variance = max(0.0, float(second - mean * mean))
        runs["fast" if fast else "standard"] = {
            "seconds": elapsed,
            "log_probability": float(log_mean),
            "probability": float(mean),
            "standard_error": math.sqrt(variance / args.samples),
            "model_forward_rows": result["model_forward_rows"],
            "model_forward_calls": result["model_forward_calls"],
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "sample_log_probabilities": result["sample_log_probabilities"],
        }
        if args.profile_forward:
            runs["fast" if fast else "standard"]["forward_cuda_seconds"] = sum(
                started.elapsed_time(finished) for started, finished in forward_events
            ) / 1000.0
            forward_events.clear()
        print(json.dumps({"run": "fast" if fast else "standard",
                          **{k: v for k, v in runs["fast" if fast else "standard"].items()
                             if k != "sample_log_probabilities"}}), flush=True)

    faster = runs["fast"]
    if args.fast_only:
        if args.output is not None:
            args.output.write_text(json.dumps({
                "text": str(args.text), "remasking": args.remasking,
                "samples": args.samples, "seed": args.seed,
                "fast": {key: value for key, value in faster.items()
                         if key != "sample_log_probabilities"},
            }, indent=2), encoding="utf-8")
        return
    standard = runs["standard"]
    finite_pairs = [(a, b) for a, b in zip(
        standard["sample_log_probabilities"], faster["sample_log_probabilities"]
    ) if math.isfinite(a) and math.isfinite(b)]
    output = {
        "text": str(args.text), "remasking": args.remasking,
        "samples": args.samples, "seed": args.seed,
        "speedup": standard["seconds"] / faster["seconds"],
        "max_finite_sample_log_difference": max(
            (abs(a - b) for a, b in finite_pairs), default=None,
        ),
        "standard": {k: v for k, v in standard.items()
                     if k != "sample_log_probabilities"},
        "fast": {k: v for k, v in faster.items()
                 if k != "sample_log_probabilities"},
    }
    print(json.dumps(output), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
