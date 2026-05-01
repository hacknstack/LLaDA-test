#!/usr/bin/env python3

import argparse
import csv
import json
import random
import shutil
import statistics
import zipfile
from pathlib import Path


WINDOWS_FILENAME = "windows.csv"
SUMMARY_FILENAME = "summary.json"


def quantile(sorted_values, q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty list")

    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def estimate_probability(p: float, num_samples: int) -> float:
    hits = 0
    for _ in range(num_samples):
        if random.random() < p:
            hits += 1
    return hits / num_samples


def estimate_windows_csv(input_csv: Path, output_csv: Path, num_samples: int) -> list[float]:
    rows = []
    estimated_pzs = []

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")

        if "p_z" not in fieldnames:
            raise ValueError(f"CSV must contain a 'p_z' column: {input_csv}")

        for row in reader:
            original_p = float(row["p_z"])
            estimated_p = estimate_probability(original_p, num_samples)

            row["p_z"] = str(estimated_p)
            rows.append(row)
            estimated_pzs.append(estimated_p)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return estimated_pzs


def update_summary_json(summary_path: Path, estimated_pzs: list[float]) -> None:
    if not summary_path.exists():
        print(f"WARNING: Missing summary.json: {summary_path}")
        return

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        print(f"WARNING: Invalid or empty summary.json, skipping: {summary_path}")
        return

    total_windows = len(estimated_pzs)
    num_extracted = sum(1 for p in estimated_pzs if p > 0)
    extraction_rate = num_extracted / total_windows if total_windows else 0.0

    sorted_pzs = sorted(estimated_pzs)

    summary.setdefault("parameters", {})
    summary["parameters"]["mode"] = "monte-carlo"
    summary["num_windows_extracted"] = num_extracted
    summary["extraction_rate"] = extraction_rate

    summary["p_z_distribution"] = {
        "min": min(sorted_pzs),
        "median": statistics.median(sorted_pzs),
        "mean": statistics.mean(sorted_pzs),
        "max": max(sorted_pzs),
        "q_0.9": quantile(sorted_pzs, 0.9),
        "q_0.99": quantile(sorted_pzs, 0.99),
        "q_0.999": quantile(sorted_pzs, 0.999),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

def zip_directory(directory: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("num_samples", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("estimated_windows_mirror"),
    )
    script_dir = Path(__file__).parent

    parser.add_argument(
        "--zip-name",
        type=Path,
        default=script_dir / "estimated_windows_mirror.zip",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")

    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    if args.zip_name.exists():
        args.zip_name.unlink()

    shutil.copytree(args.input_dir, args.output_dir)

    windows_files = list(args.input_dir.rglob(WINDOWS_FILENAME))

    if not windows_files:
        raise FileNotFoundError(f"No {WINDOWS_FILENAME} files found in {args.input_dir}")

    for input_windows_csv in windows_files:
        relative_path = input_windows_csv.relative_to(args.input_dir)
        output_windows_csv = args.output_dir / relative_path
        output_summary_json = output_windows_csv.parent / SUMMARY_FILENAME

        estimated_pzs = estimate_windows_csv(
            input_csv=input_windows_csv,
            output_csv=output_windows_csv,
            num_samples=args.num_samples,
        )

        update_summary_json(
            summary_path=output_summary_json,
            estimated_pzs=estimated_pzs,
        )

    zip_directory(args.output_dir, args.zip_name)

    print(f"Processed {len(windows_files)} windows.csv files")
    print(f"Wrote mirror directory to: {args.output_dir}")
    print(f"Wrote zip file to: {args.zip_name}")


if __name__ == "__main__":
    main()