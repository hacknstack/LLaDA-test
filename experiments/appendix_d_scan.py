import argparse
import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from probabilistic_extraction import compute_probabilitic_extraction


@dataclass
class BookMatch:
    row: Dict[str, str]
    matched: bool
    matched_key: Optional[str]
    reason: str


def _normalize(s: str) -> str:
    s = (s or '').lower()
    s = re.sub(r'\.[a-z0-9]{1,5}$', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _safe_name(s: str) -> str:
    s = _normalize(s).replace(' ', '_')
    return s[:80] if len(s) > 80 else s


def _path_candidates(path_value: str) -> List[str]:
    p = path_value.strip()
    parts = [x for x in p.split('/') if x]
    out = [p]
    if parts:
        out.append(parts[-1])
        out.append('/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1])
    return list(dict.fromkeys([_normalize(x) for x in out if x]))


def _iter_dataset_rows(dataset_name: str, split: str, streaming: bool):
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        text = row.get('text') or row.get('content') or ''
        meta_path = (
            row.get('meta', {}).get('pile_set_name') if isinstance(row.get('meta'), dict) else None
        )
        if not meta_path:
            for k in ['id', 'path', 'book_path', 'book_id', 'source']:
                if k in row and isinstance(row[k], str):
                    meta_path = row[k]
                    break
        yield i, row, text, (meta_path or str(i))


def resolve_books(csv_rows: List[Dict[str, str]], dataset_name: str, split: str, streaming: bool, max_scan_rows: Optional[int]):
    wanted = {r['id']: r for r in csv_rows}
    by_id: Dict[str, BookMatch] = {
        r['id']: BookMatch(row=r, matched=False, matched_key=None, reason='unresolved') for r in csv_rows
    }

    title_index = {}
    for r in csv_rows:
        title_index.setdefault(_normalize(r.get('title', '')), []).append(r['id'])

    path_candidates = {r['id']: _path_candidates(r.get('books3_path', '')) for r in csv_rows}

    for idx, row, _text, dataset_key in _iter_dataset_rows(dataset_name, split, streaming):
        if max_scan_rows is not None and idx >= max_scan_rows:
            break

        key_norm = _normalize(dataset_key)
        if not key_norm:
            continue

        matched_ids: List[str] = []
        for bid, cands in path_candidates.items():
            if by_id[bid].matched:
                continue
            if any(c and c in key_norm for c in cands):
                matched_ids.append(bid)

        if not matched_ids:
            title_hits = title_index.get(_normalize(row.get('title', '')), [])
            for bid in title_hits:
                if not by_id[bid].matched:
                    matched_ids.append(bid)

        for bid in matched_ids:
            by_id[bid] = BookMatch(row=wanted[bid], matched=True, matched_key=dataset_key, reason='matched')

        if all(v.matched for v in by_id.values()):
            break

    for bid, m in by_id.items():
        if not m.matched:
            m.reason = 'no_dataset_match_found'

    return by_id


def _extract_book_text(dataset_name: str, split: str, streaming: bool, dataset_key: str) -> Optional[str]:
    key_norm = _normalize(dataset_key)
    for _idx, _row, text, row_key in _iter_dataset_rows(dataset_name, split, streaming):
        if _normalize(row_key) == key_norm:
            return text
    return None


def scan_book(
    text: str,
    tokenizer,
    model,
    steps: int,
    remasking: str,
    estimation_method: str,
    num_samples: int,
    seed: Optional[int],
    add_special_tokens: bool,
    threshold: float,
    char_stride: int = 10,
    chunk_chars: int = 800,
):
    windows = []
    char_start = 0
    stop_reason = 'eof'
    stop_char_start = len(text)

    while char_start < len(text):
        chunk = text[char_start: char_start + chunk_chars]
        tok = tokenizer(chunk, add_special_tokens=add_special_tokens)
        token_ids = tok['input_ids']
        num_tokens_in_chunk = len(token_ids)

        if num_tokens_in_chunk < 100:
            stop_reason = 'fewer_than_100_tokens'
            stop_char_start = char_start
            break

        seq_tokens = token_ids[:100]
        prefix = seq_tokens[:50]
        suffix = seq_tokens[50:100]

        prompt_tokens = torch.tensor([prefix], dtype=torch.long, device=model.device)
        target_tokens = torch.tensor([suffix], dtype=torch.long, device=model.device)

        out = compute_probabilitic_extraction(
            model=model,
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            steps=steps,
            remasking=remasking,
            estimation_method=estimation_method,
            num_samples=num_samples,
            seed=seed,
        )
        pz = out['probability'] if estimation_method == 'exact' else out['estimate']

        windows.append(
            {
                'char_start': char_start,
                'chunk_char_len': len(chunk),
                'num_tokens_in_chunk': num_tokens_in_chunk,
                'seq_len': 100,
                'p_z': float(pz),
                'extracted': bool(pz >= threshold),
                'prefix_token_ids': prefix,
                'suffix_token_ids': suffix,
            }
        )
        char_start += char_stride

    return windows, {'stop_reason': stop_reason, 'stop_char_start': stop_char_start}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return float('nan')
    sv = sorted(vals)
    i = min(len(sv) - 1, max(0, int(math.floor(q * (len(sv) - 1)))))
    return sv[i]


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Appendix D sliding-window probabilistic extraction scan for LLaDA 8B '
            '(low-confidence remasking only; no top-k decoding).'
        )
    )
    parser.add_argument('--books-csv', type=Path, required=True)
    parser.add_argument('--dataset-name', type=str, default='SaylorTwift/the_pile_books3_minus_gutenberg')
    parser.add_argument('--dataset-split', type=str, default='train')
    parser.add_argument('--dataset-streaming', action='store_true')
    parser.add_argument('--max-scan-rows', type=int, default=None, help='Limit rows scanned during matching for debug.')
    parser.add_argument('--model-name', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--remasking', type=str, default='low-confidence')
    parser.add_argument('--estimation-method', type=str, choices=['exact', 'monte-carlo'], default='monte-carlo')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.001)
    parser.add_argument('--add-special-tokens', action='store_true', default=False)
    parser.add_argument('--char-stride', type=int, default=10)
    parser.add_argument('--chunk-chars', type=int, default=800)
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--limit-books', type=int, default=None)
    args = parser.parse_args()

    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir or Path('runs') / f'appendix_d_llda8b_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)

    with args.books_csv.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if args.limit_books is not None:
        rows = rows[: args.limit_books]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    matches = resolve_books(
        csv_rows=rows,
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
        max_scan_rows=args.max_scan_rows,
    )

    missing_rows = []
    all_summary = []

    for row in rows:
        bid = row['id']
        title = row.get('title', '')
        match = matches[bid]

        if not match.matched or not match.matched_key:
            missing_rows.append(
                {
                    'id': bid,
                    'author': row.get('author', ''),
                    'title': title,
                    'year': row.get('year', ''),
                    'status': row.get('status', ''),
                    'books3_path': row.get('books3_path', ''),
                    'reason': match.reason,
                }
            )
            continue

        text = _extract_book_text(
            dataset_name=args.dataset_name,
            split=args.dataset_split,
            streaming=args.dataset_streaming,
            dataset_key=match.matched_key,
        )

        if not text:
            missing_rows.append(
                {
                    'id': bid,
                    'author': row.get('author', ''),
                    'title': title,
                    'year': row.get('year', ''),
                    'status': row.get('status', ''),
                    'books3_path': row.get('books3_path', ''),
                    'reason': 'matched_but_text_unavailable',
                }
            )
            continue

        windows, stop_info = scan_book(
            text=text,
            tokenizer=tokenizer,
            model=model,
            steps=args.steps,
            remasking=args.remasking,
            estimation_method=args.estimation_method,
            num_samples=args.num_samples,
            seed=args.seed,
            add_special_tokens=args.add_special_tokens,
            threshold=args.threshold,
            char_stride=args.char_stride,
            chunk_chars=args.chunk_chars,
        )

        book_dir = run_dir / 'books' / f"{bid}_{_safe_name(title)}"
        _write_jsonl(book_dir / 'windows.jsonl', windows)

        total = len(windows)
        extracted = sum(1 for w in windows if w['extracted'])
        pvals = [w['p_z'] for w in windows]
        top_windows = sorted(
            [{'char_start': w['char_start'], 'p_z': w['p_z']} for w in windows],
            key=lambda x: x['p_z'],
            reverse=True,
        )[:50]

        summary = {
            'id': bid,
            'author': row.get('author', ''),
            'title': title,
            'year': row.get('year', ''),
            'status': row.get('status', ''),
            'resolved_dataset_key': match.matched_key,
            'total_windows': total,
            'num_extracted': extracted,
            'extraction_rate': (extracted / total) if total else 0.0,
            'max_pz': max(pvals) if pvals else float('nan'),
            'pz_99p': _quantile(pvals, 0.99),
            'top_windows': top_windows,
            'stopping_condition': stop_info,
            'config': {
                'char_stride': args.char_stride,
                'chunk_chars': args.chunk_chars,
                'add_special_tokens': args.add_special_tokens,
                'threshold': args.threshold,
                'steps': args.steps,
                'remasking': args.remasking,
                'estimation_method': args.estimation_method,
                'num_samples': args.num_samples,
                'seed': args.seed,
                'sampling_note': (
                    'probabilistic_extraction currently supports low-confidence remasking '
                    'semantics only (temperature=0, no top-k decoding).'
                ),
            },
        }
        _write_json(book_dir / 'summary.json', summary)
        all_summary.append(summary)

    summary_csv = run_dir / 'all_books_summary.csv'
    with summary_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'id', 'author', 'title', 'year', 'status', 'resolved_dataset_key',
                'total_windows', 'num_extracted', 'extraction_rate', 'max_pz', 'pz_99p',
            ],
        )
        writer.writeheader()
        for s in all_summary:
            writer.writerow({k: s.get(k) for k in writer.fieldnames})

    missing_csv = run_dir / 'missing_books.csv'
    with missing_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['id', 'author', 'title', 'year', 'status', 'books3_path', 'reason'],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    _write_jsonl(run_dir / 'missing_books.jsonl', missing_rows)

    _write_json(
        run_dir / 'run_metadata.json',
        {
            'books_csv': str(args.books_csv),
            'num_requested_books': len(rows),
            'num_processed_books': len(all_summary),
            'num_missing_books': len(missing_rows),
            'dataset_name': args.dataset_name,
            'dataset_split': args.dataset_split,
            'dataset_streaming': args.dataset_streaming,
            'model_name': args.model_name,
            'run_dir': str(run_dir),
        },
    )


if __name__ == '__main__':
    main()
