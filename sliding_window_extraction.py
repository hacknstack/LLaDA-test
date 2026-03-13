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

from probabilistic_extraction import compute_probabilistic_extraction


DEFAULT_LLADA_MODEL = 'GSAI-ML/LLaDA-8B-Base'
DEFAULT_LLAMA_MODEL = 'NousResearch/Meta-Llama-3-8B'
MASK_ID = 126336


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sliding-window probabilistic extraction for a single text file.'
    )
    parser.add_argument('txt_path', type=Path, help='Input txt file path (e.g. texts/book.txt)')
    parser.add_argument('--mode', choices=['exact', 'monte-carlo'], default='exact')
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
    parser.add_argument('--num-samples', type=int, default=1000, help='Monte Carlo samples when --mode monte-carlo')
    parser.add_argument('--seed', type=int, default=None, help='Optional Monte Carlo seed')
    parser.add_argument('--decoding-scheme', choices=['top_k', 'greedy'], default='top_k')
    parser.add_argument('--k', type=int, default=40, help='Top-k value when --model-family llama and --decoding-scheme top_k')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature when --model-family llama and --decoding-scheme top_k')
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


def _compute_probability(model, prefix_ids: List[int], suffix_ids: List[int], args: argparse.Namespace) -> float:
    prompt_tokens = torch.tensor([prefix_ids], dtype=torch.long)
    target_tokens = torch.tensor([suffix_ids], dtype=torch.long)

    result = compute_probabilistic_extraction(
        model=model,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        steps=len(suffix_ids),
        attention_mask=None,
        mask_id=MASK_ID,
        remasking='low-confidence',
        estimation_method=args.mode,
        num_samples=args.num_samples,
        seed=args.seed,
        model_family=args.model_family,
        decoding_scheme=args.decoding_scheme,
        k=args.k,
        temperature=args.temperature,
    )

    if args.model_family == 'llama':
        return float(result['probability'])
    if args.mode == 'exact':
        return float(result['probability'])
    return float(result['estimate'])


def main() -> None:
    args = parse_args()

    if args.prefix_tokens + args.suffix_tokens != args.seq_tokens:
        raise ValueError('prefix_tokens + suffix_tokens must equal seq_tokens.')
    if not args.txt_path.exists() or not args.txt_path.is_file():
        raise FileNotFoundError(f'Input file not found: {args.txt_path}')

    if args.model_name is None:
        args.model_name = DEFAULT_LLADA_MODEL if args.model_family == 'llada' else DEFAULT_LLAMA_MODEL

    if args.model_family == 'llama' and args.mode != 'exact':
        raise ValueError("--mode must be 'exact' when --model-family llama.")

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

        p_z = float('nan')
        extracted = 0
        error = ''

        try:
            p_z = _compute_probability(model=model, prefix_ids=prefix_ids, suffix_ids=suffix_ids, args=args)
            extracted = int(p_z >= args.tau)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        rows.append(
            {
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
            'stride_chars': args.stride_chars,
            'seq_tokens': args.seq_tokens,
            'prefix_tokens': args.prefix_tokens,
            'suffix_tokens': args.suffix_tokens,
            'tau_min': args.tau,
            'mode': args.mode,
            'model_family': args.model_family,
            'model_name': args.model_name,
            'decoding_scheme': args.decoding_scheme,
            'k': args.k,
            'temperature': args.temperature,
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