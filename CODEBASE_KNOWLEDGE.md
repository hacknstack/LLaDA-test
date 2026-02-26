# Codebase Knowledge Notes

This repository provides inference and evaluation utilities for **LLaDA (Large Language Diffusion with mAsking)** models.

## High-level layout
- `generate.py`: diffusion-style conditional generation for LLaDA models.
- `get_log_likelihood.py`: conditional likelihood scoring helper.
- `chat.py` and `app.py`: interactive chat and Gradio demo entrypoints.
- `eval_llada.py`, `eval_reverse.py`, and shell wrappers: evaluation workflows.
- `visualization/`: scripts for visualizing generation dynamics.
- `opencompass/`: OpenCompass integration for benchmark evaluation.

## Focus: `generate.py`

`generate.py` is the core sampling implementation and contains four important pieces:

1. `add_gumbel_noise(logits, temperature)`
   - Applies temperature-controlled Gumbel perturbation to logits.
   - Uses `float64` for improved numerical behavior during sampling.
   - Returns transformed token scores used for argmax token proposals.

2. `get_num_transfer_tokens(mask_index, steps)`
   - Computes how many masked tokens should be filled per reverse diffusion step.
   - Splits remaining masked tokens nearly uniformly across the configured step count.

3. `generate(...)`
   - Initializes output as `[prompt | masks]`, then iteratively denoises masked positions.
   - Supports optional classifier-free guidance (`cfg_scale`) by running conditional and unconditional passes.
   - Supports two remasking strategies:
     - `low_confidence`: fills highest-confidence masked tokens each step.
     - `random`: fills randomly selected masked tokens.
   - Supports block-wise (semi-autoregressive) generation via `block_length`.
   - Supports optional EOS/EOT suppression toggles from paper appendices.

4. `main()` demo
   - Loads `GSAI-ML/LLaDA-8B-Instruct` from Hugging Face.
   - Ensures left padding for simpler generation behavior.
   - Encodes example math prompts and prints generated responses.

## Practical sampling behavior notes
- `steps` should generally scale with generation length for best quality.
- `gen_length % block_length == 0` and `steps % num_blocks == 0` are required by assertions.
- The script assumes mask token id `126336`; changing tokenizer/model setup may require updates.
- For instruct usage, prompts are wrapped with `tokenizer.apply_chat_template(...)`.

## Suggested reading order for contributors
1. `README.md` (project goals and usage).
2. `generate.py` (sampling loop and decoding mechanics).
3. `get_log_likelihood.py` (evaluation-side objective utility).
4. `chat.py` / `app.py` (interactive usage patterns).
5. `EVAL.md` and evaluation scripts (benchmark pipelines).
