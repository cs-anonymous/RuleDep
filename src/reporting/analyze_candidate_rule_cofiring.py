#!/usr/bin/env python3
"""Summarize fired-rule counts over relation-local query-candidate rows."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


def analyze_relation(task: tuple[str, str]) -> tuple[str, Counter[int]]:
    dataset, relation_file = task
    with Path(relation_file).open("rb") as handle:
        split = pickle.load(handle)["train"]

    flat = split["rules_flat"].numpy()
    offsets = split["offsets"].numpy()
    histogram: Counter[int] = Counter()
    for row_id in range(len(offsets) - 1):
        start = int(offsets[row_id])
        end = int(offsets[row_id + 1])
        rules = flat[start:end]
        if len(rules):
            rules = rules[rules >= 0]
        histogram[int(len(np.unique(rules)))] += 1
    return dataset, histogram


def value_at_index(histogram: Counter[int], index: int) -> int:
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if index < cumulative:
            return value
    raise IndexError(index)


def quantile(histogram: Counter[int], probability: float) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    position = (total - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = value_at_index(histogram, lower_index)
    upper = value_at_index(histogram, upper_index)
    return lower + (upper - lower) * (position - lower_index)


def summarize(dataset: str, histogram: Counter[int]) -> dict:
    total = sum(histogram.values())
    total_rules = sum(rule_count * frequency for rule_count, frequency in histogram.items())
    at_least_two = sum(frequency for rule_count, frequency in histogram.items() if rule_count >= 2)
    at_least_three = sum(frequency for rule_count, frequency in histogram.items() if rule_count >= 3)
    return {
        "dataset": dataset,
        "query_candidates": total,
        "mean_fired_rules": total_rules / total if total else 0.0,
        "median_fired_rules": quantile(histogram, 0.5),
        "p90_fired_rules": quantile(histogram, 0.9),
        "p95_fired_rules": quantile(histogram, 0.95),
        "max_fired_rules": max(histogram, default=0),
        "query_candidates_ge_2_rules": at_least_two,
        "fraction_ge_2_rules": at_least_two / total if total else 0.0,
        "query_candidates_ge_3_rules": at_least_three,
        "fraction_ge_3_rules": at_least_three / total if total else 0.0,
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="Dataset directory names under data/")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--jobs", type=int, default=36)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/high_order_analysis/candidate_rule_cofiring_stats.csv"),
    )
    args = parser.parse_args()

    tasks = []
    for dataset in args.datasets:
        relation_files = sorted(glob.glob(str(args.data_root / dataset / "datasets" / "dataset_*.p")))
        if not relation_files:
            raise FileNotFoundError(f"No relation-local datasets found for {dataset}")
        tasks.extend((dataset, path) for path in relation_files)

    histograms = {dataset: Counter() for dataset in args.datasets}
    max_workers = max(1, min(args.jobs, len(tasks), os.cpu_count() or 1))
    print(f"Running {len(tasks)} relation tasks with {max_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_relation, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            dataset, histogram = future.result()
            histograms[dataset].update(histogram)
            if completed % 25 == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} relations", flush=True)

    rows = [summarize(dataset, histograms[dataset]) for dataset in args.datasets]
    write_csv(args.output, rows)
    for row in rows:
        print(
            f"{row['dataset']}: n={row['query_candidates']:,} "
            f"mean={row['mean_fired_rules']:.3f} median={row['median_fired_rules']:.1f} "
            f">=3={row['query_candidates_ge_3_rules']:,} ({row['fraction_ge_3_rules']:.2%})",
            flush=True,
        )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
