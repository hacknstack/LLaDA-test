#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from expected_extraction import _compute_expectation


DEFAULT_LLADA_MODEL = 'GSAI-ML/LLaDA-8B-Base'
DEFAULT_LLAMA_MODEL = 'NousResearch/Meta-Llama-3-8B'
MASK_ID = 126336


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sliding-window expected extraction for a single text file.'
    )
    parser.add_argument('txt_path', type=Path, help='Input txt file path (e.g. texts/book.txt)')
    parser.add_argument('--mode', choices=['RB_sampling'], default='RB_sampling')
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--chunk-chars', type=int, default=800)
    parser.add_argument('--stride-chars', type=int, default=10)
    parser.add_argument('--seq-tokens', type=int, default=100)
    parser.add_argument('--prefix-tokens', type=int, default=50)
    parser.add_argument('--suffix-tokens', type=int, default=50)
    parser.add_argument('--max-windows', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--model-family', choices=['llada', 'llama'], default='llada')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=20, help='RB sampling samples when --mode RB_sampling')
    parser.add_argument('--seed', type=int, default=None, help='Optional RB sampling seed')
    parser.add_argument('--decoding-scheme', choices=['auto', 'full', 'top_k'], default='auto')
    parser.add_argument('--k', type=int, default=40, help='Top-k value when --decoding-scheme top_k')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for llama/llada RB sampling')
    parser.add_argument('--remasking', choices=['low-confidence', 'random'], default='low-confidence',
                        help='Remasking strategy when --model-family llada')
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


def _compute_expectation_for_window(model, prefix_ids: List[int], suffix_ids: List[int], args: argparse.Namespace) -> float:
    return float(_compute_expectation(model=model, prefix_ids=prefix_ids, suffix_ids=suffix_ids, args=args))


def main() -> None:
    args = parse_args()

    if args.prefix_tokens + args.suffix_tokens != args.seq_tokens:
        raise ValueError('prefix_tokens + suffix_tokens must equal seq_tokens.')
    if not args.txt_path.exists() or not args.txt_path.is_file():
        raise FileNotFoundError(f'Input file not found: {args.txt_path}')

    if args.model_name is None:
        args.model_name = DEFAULT_LLADA_MODEL if args.model_family == 'llada' else DEFAULT_LLAMA_MODEL

    if args.mode != 'RB_sampling':
        raise ValueError("--mode must be 'RB_sampling' for expected extraction.")

    decoding_scheme = _resolve_decoding_scheme(args)
    if decoding_scheme not in {'top_k', 'full'}:
        raise ValueError("--decoding-scheme must be one of {'auto', 'top_k', 'full'} for expected extraction.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 for expected extraction RB sampling.")
    if decoding_scheme == 'top_k' and args.k <= 0:
        raise ValueError("--k must be > 0 when --decoding-scheme top_k.")
    if args.model_family == 'llada' and args.remasking != 'random':
        raise ValueError("--remasking must be 'random' when --model-family llada.")

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model_cls = AutoModel if args.model_family == 'llada' else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
    ).to(device).eval()

    text = args.txt_path.read_text(encoding='utf-8', errors='replace')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.output_dir / args.txt_path.stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pos = 0
    window_index = 0

    total_possible = max(0, (len(text) + args.stride_chars - 1) // args.stride_chars)

    pbar = tqdm(total=total_possible, desc='Sliding windows', unit='window')

    while pos < len(text):
        if args.max_windows is not None and window_index >= args.max_windows:
            break

        chunk = text[pos: pos + args.chunk_chars]
        token_ids = tokenizer(chunk, add_special_tokens=False)['input_ids']
        n_tokens = len(token_ids)

        if n_tokens < args.seq_tokens:
            pbar.update(1)
            break

        z = token_ids[:args.seq_tokens]
        prefix_ids = z[:args.prefix_tokens]
        suffix_ids = z[args.prefix_tokens: args.prefix_tokens + args.suffix_tokens]

        expected_matches = float('nan')
        error = ''

        try:
            expected_matches = _compute_expectation_for_window(model=model, prefix_ids=prefix_ids, suffix_ids=suffix_ids, args=args)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        rows.append(
            {
                'window_index': window_index,
                'char_start': pos,
                'char_end': pos + args.chunk_chars,
                'expected_matches': expected_matches,
                'error': error,
                'n_tokens_in_chunk': n_tokens,
                'sequence_len_tokens': len(z),
            }
        )

        pos += args.stride_chars
        window_index += 1
        pbar.update(1)

    pbar.close()

    windows_path = run_dir / 'windows.csv'
    with windows_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'window_index',
                'char_start',
                'char_end',
                'expected_matches',
                'error',
                'n_tokens_in_chunk',
                'sequence_len_tokens',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    scored_vals = [float(r['expected_matches']) for r in rows if isinstance(r['expected_matches'], float) and not torch.isnan(torch.tensor(r['expected_matches']))]
    sorted_vals = sorted(scored_vals)

    num_windows_total = len(rows)
    num_windows_scored = len(scored_vals)
    summary = {
        'input_file': str(args.txt_path),
        'total_chars': len(text),
        'parameters': {
            'chunk_chars': args.chunk_chars,
            'stride_chars': args.stride_chars,
            'seq_tokens': args.seq_tokens,
            'prefix_tokens': args.prefix_tokens,
            'suffix_tokens': args.suffix_tokens,
            'tau': args.tau,
            'mode': args.mode,
            'model_family': args.model_family,
            'model_name': args.model_name,
            'remasking': args.remasking,
            'decoding_scheme': decoding_scheme,
            'k': args.k,
            'temperature': args.temperature,
        },
        'num_windows_total': num_windows_total,
        'num_windows_scored': num_windows_scored,
        'expected_matches_distribution': {
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

    valid_rows = [r for r in rows if isinstance(r['expected_matches'], float) and not torch.isnan(torch.tensor(r['expected_matches']))]
    top_rows = sorted(valid_rows, key=lambda r: float(r['expected_matches']), reverse=True)[:5]
    print('Top 5 windows by expected matches (char_start, expected_matches):')
    for r in top_rows:
        print(f"  {r['char_start']}, {float(r['expected_matches']):.8f}")


if __name__ == '__main__':
    main()
