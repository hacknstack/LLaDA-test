# Estimating Extraction Probabilities in Diffusion Language Models under Confidence-Based Sampling

This repository contains the code for the paper *Estimating Extraction Probabilities in Diffusion Language Models under Confidence-Based Sampling*. It implements the sliding-window extraction experiments for LLaDA and the autoregressive baselines.

Run `sliding_window_extraction.py` from the repository root with a text file in `texts/`. It loads the selected model, slides a token window through the text, estimates an extraction probability `p_z` for each window, and writes `windows.csv` and `summary.json` under `outputs/<text-stem>/<timestamp>/` by default. Install PyTorch, Transformers, and tqdm first; large LLaDA runs require a suitable GPU.


## Parameters

Values in the right column are examples from the experiments or CLI defaults. Supply a flag only when you want to change its default.

| Parameter | What it controls | Values used or default |
| --- | --- | --- |
| `txt_path` | Input UTF-8 text file. | `texts/MITLicense.txt`; the other experiment texts are in `texts/`. |
| `--model-family` | Model family. | `llada` (default), `llama`, `llama2`, `mistral`, `olmo`. |
| `--mode` | Probability estimator. `path_sampling` samples successful trajectories; `monte-carlo` samples the decoder directly; `exact` computes an exact probability; `duel` follows a deterministic greedy-confidence path. Autoregressive families require `exact`. | `exact` (default) |
| `--remasking` | LLaDA reveal rule. `low-confidence` picks the most confident sampled candidate; `threshold` reveals candidates meeting a confidence cutoff; `random` chooses an order independently of candidates. `highest-index` is also available for a fixed index-order path. | `low-confidence` (default), `threshold`, `random`. DUEL uses `low-confidence`. |
| `--masked_indexes` | Required for LLaDA: positions to regenerate in a 100-token sequence. Must be unique integers from 1 to 100. Do not use for autoregressive models. | Fifty-position masks such as `51..100` or `2,4,...,100`; ten-position subsets for exact subset DP. |
| `--decoding-scheme` | Token sampling distribution. `full` uses the whole vocabulary; `top_k` truncates and renormalizes; `auto`, `greedy` (autoregressive), and `ELBO` (LLaDA) are available where supported. | `full` (default and used in the main experiments). |
| `--k` | Vocabulary cutoff for `--decoding-scheme top_k`. | `40` by default; ignored with `full`. |
| `--temperature` | Token sampling temperature. | `1.0` (default and used throughout the main experiments).|
| `--confidence-threshold` | Untempered candidate-confidence cutoff for `--remasking threshold`. | `0.9` (default and used in the experiments). |
| `--num-samples` | Number of path-sampling trajectories or Monte Carlo trials. No effect on `exact` or `duel`. | `20` by default; `5000` in the example below. |
| `--stride-words` | Word starts skipped between consecutive windows. | `1` default; `1` and `5` were used. |
| `--seq-tokens` | Length of each evaluated sequence. | `100` default; masked-index runs require 100. |
| `--windows` | Evaluate selected **zero-based** window indices in the supplied order. Duplicate indices are run again. | For example `--windows 0 6 11`. |
| `--max-windows` | Evaluate only the first N windows. Cannot be combined with `--windows`. | Unset by default; omit both flags for all available windows. |
| `--tau` | Mark a window `extracted` when `p_z >= tau`; does not affect the probability calculation. | `0.001` default and experiment value. |
| `--nocache` | Disable state caching for partially masked low-confidence or threshold sampling. | Off by default. |
| `--scale` | Enable high-sample-count optimizations for partially masked low-confidence or threshold Monte Carlo. | Off by default; useful for very large trial counts. |
| `--output-dir` | Root directory for run outputs. | `outputs` by default. |
| `--detailed` | Write per-sample log estimates to `detailed.jsonl`. | Off by default. |
| `--detailedWithClockTimes` | Write the same estimates plus sample batch wall times to `detailedWithClockTimes.jsonl`. | Off by default. |

## Example configuration

```powershell
$mask50 = @(51..100) # last 50 positions

# Direct Monte Carlo under threshold remasking
python .\sliding_window_extraction.py .\texts\MITLicense.txt --model-family llada --mode monte-carlo --remasking threshold --masked_indexes $mask50 --decoding-scheme full --temperature 1 --confidence-threshold 0.9 --num-samples 5000 --stride-words 1 --max-windows 1
```

This PowerShell example evaluates the first window. Remove `--max-windows 1` to evaluate every available window.

`windows.csv` has one row per evaluated window with `p_z`, its extracted classification, window indices, and any error. `summary.json` records the parameters and aggregate statistics.
