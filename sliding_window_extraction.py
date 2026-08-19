#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from importlib.metadata import PackageNotFoundError, version as package_version
import re
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from probabilistic_extraction import (
    MAX_EXACT_LOW_CONFIDENCE_MASKED,
    _duel_low_confidence_probability_fast_from_partially_masked,
    compute_autoregressive_probabilistic_extraction,
    compute_diffusion_probabilistic_extraction,
    validate_masked_indexes,
)


DEFAULT_LLADA_MODEL = 'GSAI-ML/LLaDA-8B-Base'
DEFAULT_LLAMA_MODEL = 'NousResearch/Meta-Llama-3-8B'
DEFAULT_LLAMA2_MODEL = 'NousResearch/Llama-2-7b-hf'
DEFAULT_OLMO_MODEL = 'allenai/OLMo-7B-0724-hf'
DEFAULT_MISTRAL_MODEL = 'mistralai/Mistral-7B-v0.1'
MASK_ID = 126336
AUTOREGRESSIVE_MODEL_FAMILIES = {'llama', 'llama2', 'olmo', 'mistral'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sliding-window probabilistic extraction for a single text file.'
    )
    parser.add_argument('txt_path', type=Path, help='Input txt file path (e.g. texts/book.txt)')
    parser.add_argument(
        '--mode',
        choices=['exact', 'monte-carlo', 'path_sampling', 'verbosish', 'duel'],
        default='exact',
        help=(
            "Estimator mode. 'verbosish' runs partially masked low-confidence "
            'path sampling and writes only per-sample log estimates.'
        ),
    )
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--chunk-chars', type=int, default=800)
    parser.add_argument('--stride-words', type=int, default=1)
    parser.add_argument('--seq-tokens', type=int, default=100)
    parser.add_argument('--prefix-tokens', type=int, default=50)
    parser.add_argument('--suffix-tokens', type=int, default=50)
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument('--max-windows', type=int, default=None)
    window_group.add_argument(
        '--windows',
        type=int,
        nargs='+',
        default=None,
        help='Zero-based window indices to evaluate, in the requested order.',
    )
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--model-family', type=str.lower, choices=['llada', 'llama', 'llama2', 'olmo', 'mistral'], default='llada')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=20, help='Samples for Monte Carlo or path sampling')
    parser.add_argument('--seed', type=int, default=None, help='Optional sampling seed')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help=(
            'Write verbose.jsonl for supported partially masked '
            'low-confidence path sampling, Monte Carlo, or DUEL estimation.'
        ),
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use parallel arrays for candidate values in verbose.jsonl (requires --verbose).',
    )
    parser.add_argument('--decoding-scheme', choices=['auto', 'full', 'top_k', 'greedy', 'ELBO', 'elbo', 'random'], default='full')
    parser.add_argument('--k', type=int, default=40, help='Top-k value when --decoding-scheme top_k')
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help=(
            'Sampling temperature. Partially masked exact low-confidence '
            'LLaDA requires a finite value greater than 0.'
        ),
    )
    parser.add_argument('--remasking', choices=['low-confidence', 'target-token-confidence', 'random', 'highest-index'], default='low-confidence',
                        help='Remasking strategy when --model-family llada')
    parser.add_argument(
        '--masked_indexes',
        type=int,
        nargs='+',
        default=None,
        help=(
            '1-indexed positions in the 100-token sequence to mask. Exact '
            f'low-confidence LLaDA supports 1-{MAX_EXACT_LOW_CONFIDENCE_MASKED}; '
            'low-confidence path sampling/verbosish supports 1-100; other '
            'partially masked modes require exactly 50.'
        ),
    )
    return parser.parse_args()


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _resolve_decoding_scheme(args: argparse.Namespace) -> str:
    if args.decoding_scheme != 'auto':
        return args.decoding_scheme
    return 'full' if args.model_family == 'llada' else 'top_k'


def _advance_by_words(start_pos: int, word_count: int, word_start_positions: List[int], text_len: int) -> int:
    if word_count <= 0:
        return start_pos
    idx = bisect_left(word_start_positions, start_pos)
    target_idx = idx + word_count
    return word_start_positions[target_idx] if target_idx < len(word_start_positions) else text_len


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'NaN'
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_verbose_records(
    verbose_file,
    records: List[Dict[str, object]],
    evaluation_index: int,
    window_index: int,
    masked_indexes: Optional[List[int]],
) -> None:
    for item in records:
        record = {
            'evaluation_index': evaluation_index,
            'window_index': window_index,
            'masked_indexes': masked_indexes,
            **item,
        }
        verbose_file.write(
            json.dumps(_json_safe(record), separators=(',', ':'), allow_nan=False)
            + '\n'
        )
    verbose_file.flush()


def _write_verbosish_records(
    verbosish_file,
    records: List[Dict[str, object]],
    evaluation_index: int,
    window_index: int,
) -> None:
    for item in records:
        record = {
            'evaluation_index': evaluation_index,
            'window_index': window_index,
            **item,
        }
        verbosish_file.write(
            json.dumps(_json_safe(record), separators=(',', ':'), allow_nan=False)
            + '\n'
        )
    verbosish_file.flush()


def _prepare_requested_windows(requested, text, word_starts, tokenizer, args):
    invalid = sorted({index for index in requested if index < 0})
    if invalid:
        raise ValueError(f'--windows entries must be >= 0; got {invalid}.')
    wanted, resolved = set(requested), {}
    pos = window_index = 0
    while pos < len(text) and window_index <= max(wanted):
        token_ids = tokenizer(text[pos:pos + args.chunk_chars], add_special_tokens=False)['input_ids']
        if len(token_ids) < args.seq_tokens:
            break
        if window_index in wanted:
            resolved[window_index] = (pos, token_ids)
        pos = _advance_by_words(pos, args.stride_words, word_starts, len(text))
        window_index += 1
    unavailable = sorted(wanted - resolved.keys())
    if unavailable:
        raise ValueError(f'Requested --windows are unavailable or too short: {unavailable}.')
    return [(index, resolved[index][0], resolved[index][1]) for index in requested]


def _iter_window_data(text, word_starts, tokenizer, args, requested_data):
    if requested_data is not None:
        for evaluation_index, (window_index, pos, token_ids) in enumerate(requested_data):
            yield evaluation_index, window_index, pos, token_ids
        return

    pos = window_index = 0
    while pos < len(text):
        if args.max_windows is not None and window_index >= args.max_windows:
            break
        token_ids = tokenizer(
            text[pos:pos + args.chunk_chars], add_special_tokens=False
        )['input_ids']
        if len(token_ids) < args.seq_tokens:
            break
        yield window_index, window_index, pos, token_ids
        pos = _advance_by_words(pos, args.stride_words, word_starts, len(text))
        window_index += 1


def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(
            f"Fast tokenizer load failed for {model_name}: {exc}. "
            "Retrying with use_fast=False."
        )
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )


def _get_transformers_version() -> str:
    try:
        return package_version('transformers')
    except PackageNotFoundError:
        return 'unknown'


def _load_model(model_name: str, model_family: str, device: str):
    model_cls = AutoModel if model_family == 'llada' else AutoModelForCausalLM
    try:
        return model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
        ).to(device).eval()
    except ValueError as exc:
        error_text = str(exc)
        if model_family == 'olmo' and "model type `olmo`" in error_text:
            transformers_version = _get_transformers_version()
            raise RuntimeError(
                "Loading OLMo failed because this checkpoint requires a newer Transformers release. "
                f"Detected transformers=={transformers_version}. "
                "Please upgrade Transformers to >= 4.40.0 and retry. "
                "Model: allenai/OLMo-7B-0724-hf."
            ) from exc
        raise


def _compute_probability(
    model,
    prefix_ids: List[int],
    suffix_ids: List[int],
    args: argparse.Namespace,
    verbose_callback: Optional[Callable[[List[Dict[str, object]]], None]] = None,
) -> Tuple[float, Optional[List[Dict[str, object]]]]:
    prompt_tokens = torch.tensor([prefix_ids], dtype=torch.long)
    target_tokens = torch.tensor([suffix_ids], dtype=torch.long)
    decoding_scheme = _resolve_decoding_scheme(args)

    if args.model_family in AUTOREGRESSIVE_MODEL_FAMILIES:
        if args.masked_indexes is not None:
            raise ValueError('--masked_indexes is only supported when --model-family llada.')
        result = compute_autoregressive_probabilistic_extraction(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            attention_mask=None,
            model_family=args.model_family,
            decoding_scheme=decoding_scheme,
            k=args.k,
            temperature=args.temperature,
        )
        return float(result['probability']), None

    if args.mode == 'duel':
        sequence_tokens = torch.cat([prompt_tokens, target_tokens], dim=1)
        result = _duel_low_confidence_probability_fast_from_partially_masked(
            model=model,
            sequence_tokens=sequence_tokens,
            masked_indexes=args.masked_indexes,
            steps=len(args.masked_indexes),
            attention_mask=None,
            mask_id=MASK_ID,
            temperature=args.temperature,
            verbose=args.verbose,
            verbose_compact=args.compact,
        )
        return float(result['probability']), result.get('verbose_steps')

    estimation_method = 'path_sampling' if args.mode == 'verbosish' else args.mode
    result = compute_diffusion_probabilistic_extraction(
        model=model,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        steps=len(args.masked_indexes) if args.masked_indexes is not None else len(suffix_ids),
        attention_mask=None,
        mask_id=MASK_ID,
        remasking=args.remasking,
        estimation_method=estimation_method,
        num_samples=args.num_samples,
        seed=args.seed,
        decoding_scheme=decoding_scheme,
        k=args.k,
        temperature=args.temperature,
        masked_indexes=args.masked_indexes,
        verbose=args.verbose,
        verbose_compact=args.compact,
        verbose_callback=verbose_callback,
    )
    if args.mode in {'exact', 'path_sampling', 'verbosish'} or str(decoding_scheme).lower() == 'elbo':
        probability = float(result['probability'])
    else:
        probability = float(result['estimate'])
    if verbose_callback is not None and result['method'] == 'monte-carlo':
        return probability, []
    if args.mode == 'verbosish':
        sample_logs = result.get('sample_log_probabilities')
        if sample_logs is None:
            raise RuntimeError('Path sampler did not return per-sample log estimates.')
        return probability, [
            {
                'sample_index': sample_index,
                'sample_log_estimate': float(sample_log_estimate),
            }
            for sample_index, sample_log_estimate in enumerate(sample_logs)
        ]
    return probability, result.get('verbose_samples')


def main() -> None:
    args = parse_args()
    use_exact_low_confidence_dp = (
        args.model_family == 'llada'
        and args.mode == 'exact'
        and args.remasking == 'low-confidence'
        and args.masked_indexes is not None
    )
    use_partially_masked_low_confidence_path_sampling = (
        args.model_family == 'llada'
        and args.mode in {'path_sampling', 'verbosish'}
        and args.remasking == 'low-confidence'
        and args.masked_indexes is not None
    )
    args.masked_indexes = validate_masked_indexes(
        args.masked_indexes,
        expected_count=(
            None
            if (
                use_exact_low_confidence_dp
                or use_partially_masked_low_confidence_path_sampling
            )
            else 50
        ),
    )
    if (
        use_exact_low_confidence_dp
        and len(args.masked_indexes) > MAX_EXACT_LOW_CONFIDENCE_MASKED
    ):
        raise ValueError(
            'Exact low-confidence DP supports at most '
            f'{MAX_EXACT_LOW_CONFIDENCE_MASKED} masked positions; got '
            f'{len(args.masked_indexes)}.'
        )

    if args.stride_words <= 0:
        raise ValueError('--stride-words must be > 0.')
    if args.compact and not args.verbose:
        raise ValueError('--compact requires --verbose.')
    if args.windows is not None:
        duplicates = {index: count for index, count in Counter(args.windows).items() if count > 1}
        if duplicates:
            print(f'Warning: duplicate --windows will be evaluated repeatedly: {duplicates}')

    if args.prefix_tokens + args.suffix_tokens != args.seq_tokens:
        raise ValueError('prefix_tokens + suffix_tokens must equal seq_tokens.')
    if args.masked_indexes is not None:
        if args.model_family != 'llada':
            raise ValueError('--masked_indexes is only supported when --model-family llada.')
        if args.seq_tokens != 100:
            raise ValueError('--masked_indexes is only supported when --seq-tokens is exactly 100.')
    if not args.txt_path.exists() or not args.txt_path.is_file():
        raise FileNotFoundError(f'Input file not found: {args.txt_path}')

    if args.model_name is None:
        default_models = {
            'llada': DEFAULT_LLADA_MODEL,
            'llama': DEFAULT_LLAMA_MODEL,
            'llama2': DEFAULT_LLAMA2_MODEL,
            'olmo': DEFAULT_OLMO_MODEL,
            'mistral': DEFAULT_MISTRAL_MODEL,
        }
        args.model_name = default_models[args.model_family]

    if args.model_family in AUTOREGRESSIVE_MODEL_FAMILIES and args.mode != 'exact':
        raise ValueError("--mode must be 'exact' when --model-family is one of {'llama', 'llama2', 'olmo', 'mistral'}.")
    if args.model_family in AUTOREGRESSIVE_MODEL_FAMILIES and args.remasking != 'low-confidence':
        raise ValueError("--remasking is only used when --model-family llada.")
    decoding_scheme = _resolve_decoding_scheme(args)
    if args.model_family in AUTOREGRESSIVE_MODEL_FAMILIES:
        if decoding_scheme not in {'top_k', 'full', 'greedy'}:
            raise ValueError("--decoding-scheme must be one of {'auto', 'top_k', 'full', 'greedy'} when --model-family is one of {'llama', 'llama2', 'olmo', 'mistral'}.")
        if decoding_scheme in {'top_k', 'full'} and args.temperature <= 0:
            raise ValueError("--temperature must be > 0 when --model-family is one of {'llama', 'llama2', 'olmo', 'mistral'} with --decoding-scheme in {'top_k', 'full'}.")
    else:
        if decoding_scheme.lower() not in {'top_k', 'full', 'elbo', 'random'}:
            raise ValueError("--decoding-scheme must be one of {'auto', 'top_k', 'full', 'ELBO', 'random'} when --model-family llada.")
    if args.model_family == 'llada' and args.mode == 'duel':
        if args.remasking != 'low-confidence':
            raise ValueError("--mode duel requires --remasking low-confidence.")
        if args.masked_indexes is None:
            raise ValueError("--mode duel requires --masked_indexes with exactly 50 positions.")
        if decoding_scheme.lower() != 'full':
            raise ValueError("--mode duel requires --decoding-scheme full (or auto for LLaDA).")
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError("--mode duel requires a finite --temperature greater than 0.")
    if decoding_scheme == 'top_k' and args.k <= 0:
        raise ValueError("--k must be > 0 when --decoding-scheme top_k.")
    if args.model_family == 'llada' and args.remasking == 'target-token-confidence':
        if args.mode != 'exact':
            raise ValueError("--mode must be 'exact' when --remasking target-token-confidence.")
        if args.temperature <= 0:
            raise ValueError("--temperature must be > 0 when --remasking target-token-confidence.")
    if args.model_family == 'llada' and args.remasking == 'highest-index':
        if args.mode != 'exact':
            raise ValueError("--mode must be 'exact' when --remasking highest-index.")
    if args.model_family == 'llada' and args.remasking == 'random':
        if args.mode != 'path_sampling':
            raise ValueError("--mode must be 'path_sampling' when --remasking random.")
        if decoding_scheme.lower() not in {'full', 'top_k'}:
            raise ValueError("--decoding-scheme must be one of {'full', 'top_k'} when --remasking random.")
    if (
        args.model_family == 'llada'
        and args.remasking == 'low-confidence'
        and args.mode in {'path_sampling', 'verbosish'}
    ):
        if args.mode == 'verbosish' and args.masked_indexes is None:
            raise ValueError(
                '--mode verbosish requires --masked_indexes with at least one position.'
            )
        if decoding_scheme.lower() not in {'full', 'top_k'}:
            raise ValueError("--decoding-scheme must be one of {'full', 'top_k'} for low-confidence path sampling.")
        if not math.isclose(args.temperature, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("--temperature must be exactly 1 for low-confidence LLaDA path sampling.")
        if args.masked_indexes is not None and decoding_scheme.lower() != 'full':
            raise ValueError("--decoding-scheme must be 'full' for partially masked low-confidence path sampling.")
    if use_exact_low_confidence_dp:
        if decoding_scheme.lower() != 'full':
            raise ValueError(
                "--decoding-scheme must be 'full' when --masked_indexes is used "
                'with --mode exact and --remasking low-confidence.'
            )
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError(
                '--temperature must be finite and greater than 0 when '
                '--masked_indexes is used with --mode exact and '
                '--remasking low-confidence.'
            )

    if args.verbose:
        valid_path_verbose = (
            args.model_family == 'llada'
            and args.mode == 'path_sampling'
            and args.remasking == 'low-confidence'
            and args.masked_indexes is not None
            and decoding_scheme.lower() == 'full'
            and math.isclose(args.temperature, 1.0, rel_tol=0.0, abs_tol=1e-9)
        )
        valid_mc_verbose = (
            args.model_family == 'llada'
            and args.mode == 'monte-carlo'
            and args.remasking == 'low-confidence'
            and args.masked_indexes is not None
            and decoding_scheme.lower() == 'full'
            and math.isfinite(args.temperature)
            and args.temperature > 0.0
        )
        valid_duel_verbose = (
            args.model_family == 'llada'
            and args.mode == 'duel'
            and args.remasking == 'low-confidence'
            and args.masked_indexes is not None
            and decoding_scheme.lower() == 'full'
            and math.isfinite(args.temperature)
            and args.temperature > 0.0
        )
        if not (valid_path_verbose or valid_mc_verbose or valid_duel_verbose):
            raise ValueError(
                '--verbose requires partially masked LLaDA low-confidence path '
                'sampling at temperature 1, or Monte Carlo/DUEL estimation at '
                'positive temperature, all with full decoding.'
            )

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = _load_tokenizer(args.model_name)
    text = args.txt_path.read_text(encoding='utf-8', errors='replace')
    word_start_positions = [m.start() for m in re.finditer(r'\S+', text)]
    if not word_start_positions:
        raise ValueError('Input text contains no words to slide across.')

    requested_window_data = None
    if args.windows is not None:
        requested_window_data = _prepare_requested_windows(
            args.windows, text, word_start_positions, tokenizer, args
        )

    model = _load_model(args.model_name, args.model_family, device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.output_dir / args.txt_path.stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_possible = max(0, (len(word_start_positions) + args.stride_words - 1) // args.stride_words)
    if requested_window_data is not None:
        total_to_evaluate = len(requested_window_data)
    elif args.max_windows is not None:
        total_to_evaluate = min(total_possible, args.max_windows)
    else:
        total_to_evaluate = total_possible
    pbar = tqdm(total=total_to_evaluate, desc='Sliding windows', unit='window')
    verbose_file = (run_dir / 'verbose.jsonl').open('w', encoding='utf-8') if args.verbose else None
    verbosish_file = (
        (run_dir / 'verbosish.jsonl').open('w', encoding='utf-8')
        if args.mode == 'verbosish'
        else None
    )
    window_data = _iter_window_data(
        text, word_start_positions, tokenizer, args, requested_window_data
    )

    for evaluation_index, window_index, pos, token_ids in window_data:
        n_tokens = len(token_ids)

        z = token_ids[:args.seq_tokens]
        prefix_ids = z[:args.prefix_tokens]
        suffix_ids = z[args.prefix_tokens: args.prefix_tokens + args.suffix_tokens]

        p_z = float('nan')
        extracted = 0
        error = ''

        try:
            verbose_callback = None
            if verbose_file is not None and args.mode == 'monte-carlo':
                def _stream_verbose_batch(samples):
                    _write_verbose_records(
                        verbose_file=verbose_file,
                        records=samples,
                        evaluation_index=evaluation_index,
                        window_index=window_index,
                        masked_indexes=args.masked_indexes,
                    )

                verbose_callback = _stream_verbose_batch

            p_z, verbose_records = _compute_probability(
                model=model,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                args=args,
                verbose_callback=verbose_callback,
            )
            if verbose_file is not None:
                if verbose_records is None:
                    raise RuntimeError('Verbose estimator data was not returned.')
                _write_verbose_records(
                    verbose_file=verbose_file,
                    records=verbose_records,
                    evaluation_index=evaluation_index,
                    window_index=window_index,
                    masked_indexes=args.masked_indexes,
                )
            elif verbosish_file is not None:
                if verbose_records is None:
                    raise RuntimeError('Verbosish estimator data was not returned.')
                _write_verbosish_records(
                    verbosish_file=verbosish_file,
                    records=verbose_records,
                    evaluation_index=evaluation_index,
                    window_index=window_index,
                )
            print(f"pz {p_z}")
            extracted = int(p_z >= args.tau)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            print(error, ":(")

        rows.append(
            {
                'evaluation_index': evaluation_index,
                'window_index': window_index,
                'char_start': pos,
                'char_end': pos + args.chunk_chars,
                'p_z': p_z,
                'extracted': extracted,
                'error': error,
                'n_tokens_in_chunk': n_tokens,
                'sequence_len_tokens': len(z),
            }
        )

        pbar.update(1)

    pbar.close()
    if verbose_file is not None:
        verbose_file.close()
    if verbosish_file is not None:
        verbosish_file.close()

    windows_path = run_dir / 'windows.csv'
    with windows_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'evaluation_index',
                'window_index',
                'char_start',
                'char_end',
                'p_z',
                'extracted',
                'error',
                'n_tokens_in_chunk',
                'sequence_len_tokens',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    scored_vals = [float(r['p_z']) for r in rows if isinstance(r['p_z'], float) and not torch.isnan(torch.tensor(r['p_z']))]
    sorted_vals = sorted(scored_vals)

    num_windows_total = len(rows)
    num_windows_scored = len(scored_vals)
    num_windows_extracted = sum(int(r['extracted']) for r in rows)
    extraction_rate = (num_windows_extracted / num_windows_scored) if num_windows_scored > 0 else 0.0

    summary = {
        'input_file': str(args.txt_path),
        'total_chars': len(text),
        'parameters': {
            'chunk_chars': args.chunk_chars,
            'stride_words': args.stride_words,
            'windows': args.windows,
            'max_windows': args.max_windows,
            'seq_tokens': args.seq_tokens,
            'prefix_tokens': args.prefix_tokens,
            'suffix_tokens': args.suffix_tokens,
            'tau_min': args.tau,
            'mode': args.mode,
            'model_family': args.model_family,
            'model_name': args.model_name,
            'remasking': args.remasking,
            'decoding_scheme': decoding_scheme,
            'k': args.k,
            'temperature': args.temperature,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'masked_indexes': args.masked_indexes,
            'verbose': args.verbose,
            'verbosish': args.mode == 'verbosish',
            'compact': args.compact,
            'verbose_schema': (
                'parallel-arrays' if args.compact else 'candidate-objects'
            ) if args.verbose else None,
            'verbosish_schema': (
                'sample-index-and-log-estimate'
                if args.mode == 'verbosish'
                else None
            ),
        },
        'num_windows_total': num_windows_total,
        'num_windows_scored': num_windows_scored,
        'num_windows_extracted': num_windows_extracted,
        'extraction_rate': extraction_rate,
        'p_z_distribution': {
            'min': min(scored_vals) if scored_vals else None,
            'median': median(scored_vals) if scored_vals else None,
            'mean': mean(scored_vals) if scored_vals else None,
            'max': max(scored_vals) if scored_vals else None,
            'q_0.9': _quantile(sorted_vals, 0.9) if scored_vals else None,
            'q_0.99': _quantile(sorted_vals, 0.99) if scored_vals else None,
            'q_0.999': _quantile(sorted_vals, 0.999) if scored_vals else None,
        },
    }

    summary_path = run_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Output directory: {run_dir}')
    print(f'Extraction rate: {extraction_rate:.6f}')

    valid_rows = [r for r in rows if isinstance(r['p_z'], float) and not torch.isnan(torch.tensor(r['p_z']))]
    top_rows = sorted(valid_rows, key=lambda r: float(r['p_z']), reverse=True)[:5]
    print('Top 5 windows by p_z (char_start, p_z):')
    for r in top_rows:
        print(f"  {r['char_start']}, {float(r['p_z']):.8f}")


if __name__ == '__main__':
    main()
