#!/usr/bin/env python3

import csv
import random
import statistics
from pathlib import Path


INPUT_CSV = Path("windows.csv")
OUTPUT_CSV = Path("windows_estimated.csv")
NUM_SAMPLES = 100  # change this as needed


def quantile(sorted_values, q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty list")
    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1")

    if len(sorted_values) == 1:
        return sorted_values[0]

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


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    rows = []
    estimated_pzs = []

    with INPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV has no header")

        if "p_z" not in fieldnames:
            raise ValueError("CSV must contain a 'p_z' column")

        for row in reader:
            original_p = float(row["p_z"])
            estimated_p = estimate_probability(original_p, NUM_SAMPLES)
            row["p_z"] = str(estimated_p)
            rows.append(row)
            estimated_pzs.append(estimated_p)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_windows = len(estimated_pzs)
    num_above_zero = sum(1 for p in estimated_pzs if p > 0)
    num_not_above_zero = total_windows - num_above_zero

    ratio_text = str(num_above_zero / total_windows)

    sorted_pzs = sorted(estimated_pzs)

    print(f"Wrote estimated CSV to: {OUTPUT_CSV}")
    print()
    print(f"The number of windows with p_z above 0: {num_above_zero}")
    print(f"The ratio of windows with p_z above 0 to the rest: {ratio_text}")
    print("for p_z across all windows:")
    print(f'    "min": {min(sorted_pzs)}')
    print(f'    "median": {statistics.median(sorted_pzs)}')
    print(f'    "mean": {statistics.mean(sorted_pzs)}')
    print(f'    "max": {max(sorted_pzs)}')
    print(f'    "q_0.9": {quantile(sorted_pzs, 0.9)}')
    print(f'    "q_0.99": {quantile(sorted_pzs, 0.99)}')
    print(f'    "q_0.999": {quantile(sorted_pzs, 0.999)}')


if __name__ == "__main__":
    main()