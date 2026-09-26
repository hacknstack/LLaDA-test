#!/usr/bin/env python3
"""Probe model forward batching on DeclarationTranscript window 676."""
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModel, AutoTokenizer

from sliding_window_extraction import _prepare_requested_windows
from probabilistic_extraction import (
    _random_remasking_forward_logits,
    _random_remasking_projection_layer,
)


def main():
    model_name = "GSAI-ML/LLaDA-8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    text = Path("texts/DeclarationTranscript.txt").read_text(encoding="utf-8")
    starts = [match.start() for match in re.finditer(r"\S+", text)]
    args = SimpleNamespace(chunk_chars=800, stride_words=1, seq_tokens=100)
    _, offset, ids = _prepare_requested_windows([676], text, starts, tokenizer, args)[0]
    sequence = torch.tensor(ids[:100], dtype=torch.long, device="cuda")
    masked = torch.tensor([i - 1 for i in range(1, 101) if i % 4 in (0, 3)],
                          dtype=torch.long, device="cuda")
    original = sequence.expand(32, -1).clone()
    original[:, masked] = 126336
    for row in range(32):
        original[row, masked[row]] = sequence[masked[row]]
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    rows = []
    with torch.inference_mode():
        for count in (4, 8, 16, 32):
            inputs = original[:count]
            model(inputs[:1])
            model(inputs)
            torch.cuda.synchronize()

            start = time.perf_counter()
            single_logits = []
            for row in range(count):
                output = model(inputs[row:row + 1])
                single_logits.append(output.logits[0, masked[-1], :].clone())
                del output
            torch.cuda.synchronize()
            single_seconds = time.perf_counter() - start

            start = time.perf_counter()
            output = model(inputs)
            torch.cuda.synchronize()
            batched_seconds = time.perf_counter() - start
            batch_logits = output.logits[:, masked[-1], :]
            single_logits = torch.stack(single_logits)
            differences = single_logits != batch_logits
            changed_fraction = differences.float().mean().item()
            max_abs_logit_difference = (
                single_logits.float() - batch_logits.float()
            ).abs().max().item()
            single_probabilities = torch.softmax(single_logits.float(), dim=-1)
            batch_probabilities = torch.softmax(batch_logits.float(), dim=-1)
            total_variation = (
                single_probabilities - batch_probabilities
            ).abs().sum(dim=-1) / 2
            target_id = sequence[masked[-1]]
            target_probability_difference = (
                single_probabilities[:, target_id] - batch_probabilities[:, target_id]
            ).abs()
            del output, single_logits, batch_logits, differences
            rows.append({"batch_size": count, "singleton_seconds": single_seconds,
                         "batch_seconds": batched_seconds,
                         "speedup": single_seconds / batched_seconds,
                         "changed_logit_fraction": changed_fraction,
                         "max_abs_logit_difference": max_abs_logit_difference,
                         "mean_total_variation": total_variation.mean().item(),
                         "max_total_variation": total_variation.max().item(),
                         "max_target_probability_difference":
                             target_probability_difference.max().item()})
            print(json.dumps(rows[-1]), flush=True)
        projection_layer = _random_remasking_projection_layer(model)
        projection = {"supported": projection_layer is not None}
        if projection_layer is not None:
            first = original[:8]
            positions = torch.stack([
                masked[torch.arange(masked.numel(), device="cuda") != row]
                for row in range(8)
            ])
            full_logits = []
            torch.cuda.synchronize()
            start = time.perf_counter()
            for row in range(8):
                output = model(first[row:row + 1])
                full_logits.append(output.logits[0, positions[row]].clone())
                del output
            torch.cuda.synchronize()
            projection["full_seconds"] = time.perf_counter() - start
            torch.cuda.synchronize()
            start = time.perf_counter()
            projected_logits = []
            for row in range(8):
                projected_logits.append(_random_remasking_forward_logits(
                    model, first[row:row + 1], None,
                    positions[row:row + 1], projection_layer,
                )[0].clone())
            torch.cuda.synchronize()
            projection["projected_seconds"] = time.perf_counter() - start
            full_logits = torch.stack(full_logits)
            projected_logits = torch.stack(projected_logits)
            projection["equal_fraction"] = (
                full_logits == projected_logits
            ).float().mean().item()
            projection["max_abs_logit_difference"] = (
                full_logits.float() - projected_logits.float()
            ).abs().max().item()
            full_prob = torch.softmax(full_logits.float(), dim=-1)
            projected_prob = torch.softmax(projected_logits.float(), dim=-1)
            projection["mean_total_variation"] = (
                (full_prob - projected_prob).abs().sum(dim=-1) / 2
            ).mean().item()
        print(json.dumps({"projection": projection}), flush=True)
    Path(".scale_ab/batch_probe_results.json").write_text(
        json.dumps({"window_index": 676, "char_offset": offset,
                    "rows": rows, "projection": projection}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
