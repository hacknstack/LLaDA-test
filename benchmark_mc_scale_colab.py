#!/usr/bin/env python3
"""Compare direct Monte Carlo with and without --scale on one A100 window."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModel, AutoTokenizer

from probabilistic_extraction import (
    _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked,
    _monte_carlo_probability_temperature_fast_from_partially_masked,
)
from sliding_window_extraction import _prepare_requested_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=Path("texts/AliceInWonderlandChapter1.txt"))
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=800)
    parser.add_argument("--stride-words", type=int, default=1)
    parser.add_argument("--mask-pattern", choices=["suffix", "mod4"], default="suffix")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reverse", action="store_true",
                        help="Run scaled mode first to check order effects")
    parser.add_argument("--mc-batch-size", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--modes", nargs="+", choices=["low-confidence", "fast-dllm"],
                        default=["low-confidence", "fast-dllm"])
    parser.add_argument("--output", type=Path, default=Path("mc_scale_ab_results.json"))
    args = parser.parse_args()
    args.seq_tokens = 100
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = args.text.read_text(encoding="utf-8", errors="replace")
    word_starts = [match.start() for match in re.finditer(r"\S+", text)]
    (_, char_offset, ids) = _prepare_requested_windows(
        [args.window_index], text, word_starts, tokenizer, args,
    )[0]
    window_ids = ids[:100]
    sequence = torch.tensor([window_ids], dtype=torch.long)
    masked_indexes = (
        list(range(51, 101)) if args.mask_pattern == "suffix"
        else [i for i in range(1, 101) if i % 4 in (0, 3)]
    )
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    common = dict(model=model, sequence_tokens=sequence,
                  masked_indexes=masked_indexes, steps=50,
                  attention_mask=None, mask_id=126336, temperature=1.0,
                  seed=args.seed, mc_batch_size=args.mc_batch_size)
    functions = {
        "low-confidence": _monte_carlo_probability_temperature_fast_from_partially_masked,
        "fast-dllm": _monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked,
    }
    results = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "samples": args.samples, "mc_batch_size": args.mc_batch_size,
               "text": str(args.text), "window_index": args.window_index,
               "window_char_offset": char_offset,
               "window_token_sha256": hashlib.sha256(
                   json.dumps(window_ids).encode("ascii")
               ).hexdigest(),
               "masked_indexes": masked_indexes, "runs": []}

    def run(mode, scale, samples, repetition, warmup=False):
        function = functions[mode]
        kwargs = dict(common, num_samples=samples, scale=scale)
        if mode == "low-confidence":
            kwargs.update(decoding_scheme="full", k=1)
        else:
            kwargs.update(confidence_threshold=0.9, decoding_scheme="full", k=1)
        count = {"forwards": 0}

        def count_forward(_model, _inputs):
            count["forwards"] += 1

        hook = model.register_forward_pre_hook(count_forward)
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_bytes = torch.cuda.memory_allocated()
            start = time.perf_counter()
            result = function(**kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            peak_gib = (torch.cuda.max_memory_allocated() - baseline_bytes) / 2**30
        finally:
            hook.remove()
        record = {"mode": mode, "scale": scale, "repetition": repetition,
                  "samples": samples, "seconds": seconds,
                  "samples_per_second": samples / seconds,
                  "model_forwards": count["forwards"], "peak_incremental_gib": peak_gib,
                  "hits": result.hits, "estimate": result.estimate,
                  "standard_error": result.standard_error}
        print(json.dumps({"warmup" if warmup else "run": record}), flush=True)
        if not warmup:
            results["runs"].append(record)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for mode in args.modes:
        run(mode, False, min(64, args.samples), -1, warmup=True)
        run(mode, True, min(64, args.samples), -1, warmup=True)
        for repetition in range(args.repeats):
            normal_order = (repetition % 2 == 0) != args.reverse
            for scale in ((False, True) if normal_order else (True, False)):
                run(mode, scale, args.samples, repetition)


if __name__ == "__main__":
    main()
