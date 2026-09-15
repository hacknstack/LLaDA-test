# Assessing the Privacy of Large Language Diffusion Models

This repository contains code and result artifacts for running sliding-window extraction experiments for LLaDA and other autoregressive language models.

## Project overview

The main idea is to compute token extraction metrics over overlapping chunks of text. The code supports LLaDA, Llama 3, Llama 2, OLMo 7B and Mistral 7B.

## Key files and folders

### `sliding_window_extraction.py`
The primary entrypoint for sliding-window probabilistic extraction.
- Loads a text file from `texts/`
- Splits it into overlapping chunks using `chunk-chars` and `stride-chars`
- Builds prefix/suffix token windows
- Computes exact or approximate extraction probabilities for each window
- Writes CSV output into `outputs/<text-stem>/<timestamp>/windows.csv`

This script supports multiple model families, including `llada`, `llama`, `llama2`, `olmo`, and `mistral`, plus various decoding and remasking modes.

The script writes two output files by default:
- `windows.csv` — one row per sliding window with extraction data.
- `summary.json` — a run-level summary of parameters, extraction counts, and `p_z` statistics.

With `--verbose`, it also writes `verbose.jsonl`.

#### Usage examples:

```bash
python sliding_window_extraction.py texts/AliceInWonderlandChapter1.txt --mode exact --model-family llama3
python sliding_window_expected_extraction.py texts/1984Chapter1.txt --mode monte-carlo --model-family llada --temperature 1.0
```

Select zero-based windows in any order with `--windows`. The indices refer to
the candidate sequence produced by `--stride-words`; duplicates are evaluated
again and retained in the requested order.

```bash
python sliding_window_extraction.py texts/book.txt --windows 5 1 4 5
```

For partially masked LLaDA low-confidence path sampling or Monte Carlo
estimation, `--verbose` writes one JSONL record per window/sample to
`verbose.jsonl`. Monte Carlo records include the reveal path, hit status,
sampled-confidence tie count, and per-step untempered sampled log-confidences.
Add `--compact` to store per-step candidate values as parallel arrays instead
of repeated JSON objects.
`--windows` and `--max-windows` are mutually exclusive, and `--compact` requires
`--verbose`.

The partially masked low-confidence STS and Monte Carlo estimators use a shared
state evaluator: model forwards run one state at a time in temporary evaluation
mode, with deterministic PyTorch algorithms and autocast disabled. The caller's
module modes and backend settings are restored afterward. Trajectory sampling
is still batched; the MC `model_batch_size` argument is retained for compatibility
but does not batch model forwards. This prioritizes agreement between estimators
and can reduce throughput compared with batched model evaluation.

Direct MC defaults to **16,384 trajectories per batch** (`mc_batch_size`), so a
2–15k-sample run groups all occurrences of each successful state and evaluates
its model logits and FP64 distribution only once. Model forwards remain batch
size one. The candidate tensors grow with masked positions times trajectories,
independently of vocabulary size; lower `mc_batch_size` if memory is limited.
The final trajectory batch skips CPU cache writes because those states will not
recur. Changing trajectory batch size changes seeded draws but preserves the
decoder's sampling and confidence rules.

STS defaults to **1,024 trajectories per batch** (`batch_size`), covering typical
300–1,000-sample runs in one batch while keeping model forwards at batch size one.
Within each step, it evaluates each distinct state once and shares its success
probabilities across trajectories. It retains state results only for later
trajectory batches. Native logits are cached only for verbose diagnostics in
those later batches; ordinary STS uses the smaller successful-transition cache.
This removes unnecessary CPU copies and repeated diagnostics. Changing batch
size changes seeded proposal draws, but preserves the STS estimator formulas.

Both use the same per-state FP64 confidence, normalization, and CDF calculations,
plus a bounded 64 MiB CPU cache of native active logits. `use_state_cache=False`
disables this cache (and STS's successful-transition cache). Invalid numerical
values raise `FloatingPointError` with the step and revealed positions; genuine
zero success probability remains a valid result. Custom models with stochastic
or mutable evaluation behavior are unsupported. CUDA operations that require a
deterministic cuBLAS workspace need `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before
starting Python; unsupported deterministic operations raise a PyTorch error.

Exact low-confidence subset DP uses this same state evaluator and FP64
distribution calculation. Its `state_batch_size` only chunks state scheduling;
model forward batch size is always one. With 10 masked positions, it evaluates
at most 1,023 nonterminal states once each, without a logits cache. It skips only
states with exactly zero incoming probability, never small positive masses.
`num_evaluated_states` and `num_skipped_unreachable_states` report the work done;
`evaluate_all_states=True` evaluates unreachable states too for auditing. This
optimization saves model calls only when some states are unreachable. DP preserves
the smallest-index tie rule and accumulates all transitions in log space, without
probability floors or approximate pruning. `log_probability` is the natural-log result;
`probability` remains FP64-compatible down to `1e-100` (use scientific notation
when displaying or exporting it). Invalid transition masses, total success
masses, and final results raise errors; `-inf` represents genuine zero probability.

Random remasking and DUEL skip vocabulary sorting and CDF construction because
neither needs sampled-confidence competition probabilities. Their reveal policies
differ from low-confidence STS, so their final estimates need not match STS.

Random remasking first draws one uniform permutation of the 50 masked positions
for every sample. It then follows each shuffled path, scoring the target tokens
from the current state before revealing them. All 500 paths run together by
default on recognized LLaDA models, which gives 50 model calls for the usual
50-step workload on an A100 80GB.

The standard GSAI LLaDA output head projects only the positions being scored;
the transformer still sees all 100 tokens. Token distributions are normalized
in FP32 for speed, while path sums and the final arithmetic mean remain in FP64
and log space. `use_selected_logits=False` retains the generic full-output path
for comparison. `batch_size` can be reduced if a different model runs out of
memory. Seeded permutations do not depend on that batch size.

DUEL constructs and scores its path in 50 singleton forwards, without logits
caching. Its FP64 confidence calculation and smallest-index tie rule are
unchanged. At temperature 1 it reuses the ranking normalizer to score the chosen
target; other temperatures normalize only that position with stable FP64
`log_softmax`, unless full diagnostics also require scores for the other
positions. Compact diagnostics avoid that additional work and never change the
chosen score. Validation flags and diagnostic values use compact host transfers;
very large logit offsets retain a normalization-cancellation check. DUEL's 50
dependent reveals cannot be batched together within a single window, so its
speedup is limited when model inference dominates. Both estimators validate
numerical results, preserve legitimate zero probabilities, and restore the
caller's model modes and backend settings even on failure.

Run the CPU regression tests with `python -B -m unittest discover -s tests -v`.
They require PyTorch and use tiny models without downloading model weights.

DUEL is a deterministic, partially masked LLaDA low-confidence estimator. It
requires exactly 50 unique, 1-indexed masked positions in a 100-token window,
full decoding, and a positive temperature. For example, to mask the 50-token
suffix:

```bash
python sliding_window_extraction.py texts/book.txt --mode duel --model-family llada --temperature 1.0 --masked_indexes 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100
```

With DUEL, `--verbose` writes one JSONL record per reveal step (50 records per
window), rather than one record per sampled trajectory. `--num-samples` and
`--seed` are not used by DUEL.

### `results/`
Contains experiment outputs, organized by model and result type.
- `LLaDA 8B Base/` — output folders for the LLaDA 8B Base model
- `Llama 2 7B/` — outputs for Llama 2 7B experiments
- `Llama 3 8B/` — outputs for Llama 3 8B experiments
- `Mistral-7B/` — outputs for Mistral experiments
- `OLMo-7B/` — outputs for OLMo experiments

Under each model folder, experimental runs are grouped by method, temperature, sampling strategy, and text file name.

### `texts/`
Input text files used by the experiments.
- Each `.txt` file is a text source for sliding-window processing.
- Use these files as the `txt_path` argument when running the extraction scripts.

### `plots/`
Holds visualizations generated from experiments.
- Likely contains charts, figures, or analysis plots created from `results/`

### `probabilistic_extraction.py`
Core probabilistic extraction utilities for LLaDA and other families.


### `get_log_likelihood.py`
Utility functions from the LLaDA project for computing log-likelihood under the diffusion-based LLaDA model, adapted to support custom masks.

### `generate.py`
Generation utility functions from the LLaDA project.

### `estimated_windows.py`
Post-processing utility for already-generated outputs directories to create Monte-carlo samples without rerunning sliding_window_extraction.py

### `changeTau.py`
Small analysis utility for thresholding CSV results.
- Counts how many rows in a CSV exceed a given `tau` probability threshold, used to modify the tau without rerunning experiements.








