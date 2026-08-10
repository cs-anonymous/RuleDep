#!/usr/bin/env python3
"""Summarize dependency-mining sensitivity to the smooth-evidence cap."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


DATASETS = ("KG20C", "WN18RR", "codex-m", "FB15k-237", "codex-l", "YAGO3-10")
CAPS = ("5", "6", "7", "8", "9", "no-cap")


def parse_rule_metrics(path: Path, unseen: int) -> list[tuple[float, float]]:
    metrics = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t", 3)
        if len(parts) < 3:
            continue
        try:
            body = float(parts[0])
            support = float(parts[1])
        except ValueError:
            continue
        metrics.append((support, body + unseen))
    return metrics


def iter_dependency_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                body = float(parts[0])
                support = float(parts[1])
                gain = float(parts[2])
                a, b = sorted((int(parts[3]), int(parts[4])))
            except ValueError:
                continue
            yield (a, b), gain, body, support


def load_dependency_map(paths: tuple[Path, ...]) -> dict[tuple[int, int], float]:
    result = {}
    for path in paths:
        for key, gain, _body, _support in iter_dependency_rows(path):
            result[key] = gain
    return result


def smooth_evidence(support: float, denominator: float, cap: str) -> float:
    confidence = support / denominator if denominator > 0 else 0.0
    confidence = min(max(confidence, 0.0), np.nextafter(1.0, 0.0))
    value = -math.log1p(-confidence)
    return min(value, float(cap)) if cap != "no-cap" else value


def uncapped_evidence(support: float, denominator: float) -> float:
    confidence = support / denominator if denominator > 0 else 0.0
    confidence = min(max(confidence, 0.0), np.nextafter(1.0, 0.0))
    return -math.log1p(-confidence)


def distribution(values: list[float]) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=float))
    if len(absolute) == 0:
        return {key: 0.0 for key in ("mean_abs_gain", "median_abs_gain", "p95_abs_gain", "p99_abs_gain", "p999_abs_gain", "max_abs_gain")}
    return {
        "mean_abs_gain": float(np.mean(absolute)),
        "median_abs_gain": float(np.quantile(absolute, 0.5)),
        "p95_abs_gain": float(np.quantile(absolute, 0.95)),
        "p99_abs_gain": float(np.quantile(absolute, 0.99)),
        "p999_abs_gain": float(np.quantile(absolute, 0.999)),
        "max_abs_gain": float(np.max(absolute)),
    }


def summarize_cap(work: Path, cap: str, unseen: int, baseline: dict[tuple[int, int], float]) -> tuple[dict, dict, dict, dict]:
    synergy_count = 0
    redundancy_count = 0
    total_count = 0
    abs_gains = []
    common_count = 0
    sign_changes = 0
    base_common_values = []
    current_common_values = []
    joint_total = 0
    joint_saturated = 0
    threshold = None if cap == "no-cap" else float(cap)

    for dependency_type, path in (("complementary", work / "synergy.txt"), ("redundant", work / "redundancy.txt")):
        for key, gain, body, support in iter_dependency_rows(path):
            if dependency_type == "complementary":
                synergy_count += 1
            else:
                redundancy_count += 1
            total_count += 1
            abs_gains.append(abs(gain))
            joint_total += 1
            if threshold is not None and uncapped_evidence(support, body + unseen) >= threshold:
                joint_saturated += 1

            baseline_gain = baseline.get(key)
            if baseline_gain is not None:
                common_count += 1
                base_common_values.append(baseline_gain)
                current_common_values.append(gain)
                if np.sign(baseline_gain) != np.sign(gain):
                    sign_changes += 1

    if common_count:
        correlation = float(spearmanr(base_common_values, current_common_values).statistic)
    else:
        correlation = 0.0

    union_count = len(baseline) + total_count - common_count
    dependency_row = {
        "complementary_dependencies": synergy_count,
        "redundant_dependencies": redundancy_count,
        "total_dependencies": total_count,
    }
    gain_row = distribution(abs_gains)
    saturation_row = {
        "joint_evidence_saturated": joint_saturated if threshold is not None else 0,
        "joint_evidence_total": joint_total,
        "joint_saturation_rate": (joint_saturated / joint_total) if threshold is not None and joint_total else 0.0,
    }
    overlap_row = {
        "baseline_cap": "7",
        "common_dependencies": common_count,
        "jaccard_overlap": common_count / union_count if union_count else 1.0,
        "cap7_dependency_retention": common_count / len(baseline) if baseline else 1.0,
        "current_dependency_retention": common_count / total_count if total_count else 1.0,
        "common_gain_spearman": correlation,
        "common_sign_changes": sign_changes,
        "common_sign_change_rate": sign_changes / common_count if common_count else 0.0,
    }
    return dependency_row, gain_row, saturation_row, overlap_row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(dataset: str, work_root: Path, rule_root: Path, unseen: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dependency_stats = []
    gain_stats = []
    saturation_stats = []
    overlap_rows = []

    singleton_metrics = parse_rule_metrics(rule_root / dataset / "rules" / "rule.txt", unseen)
    singleton_uncapped = [uncapped_evidence(s, d) for s, d in singleton_metrics]
    baseline = load_dependency_map((
        work_root / dataset / "cap_7" / "synergy.txt",
        work_root / dataset / "cap_7" / "redundancy.txt",
    ))

    for cap in CAPS:
        work = work_root / dataset / f"cap_{cap}"
        dependency_row, gain_row, saturation_row, overlap_row = summarize_cap(work, cap, unseen, baseline)
        dependency_stats.append({"dataset": dataset, "cap": cap, **dependency_row})
        gain_stats.append({"dataset": dataset, "cap": cap, **gain_row})

        if cap == "no-cap":
            singleton_saturated = 0
        else:
            threshold = float(cap)
            singleton_saturated = sum(value >= threshold for value in singleton_uncapped)
        saturation_stats.append({
            "dataset": dataset,
            "cap": cap,
            "singleton_evidence_saturated": singleton_saturated,
            "singleton_evidence_total": len(singleton_uncapped),
            "singleton_saturation_rate": singleton_saturated / len(singleton_uncapped) if singleton_uncapped else 0.0,
            **saturation_row,
        })
        overlap_rows.append({"dataset": dataset, "cap": cap, **overlap_row})

    return dependency_stats, gain_stats, saturation_stats, overlap_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--rule-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--unseen-negative-examples", type=int, default=3)
    parser.add_argument("--parallel-datasets", type=int, default=2)
    args = parser.parse_args()

    dependency_stats = []
    gain_stats = []
    saturation_stats = []
    overlap_rows = []

    max_workers = max(1, min(args.parallel_datasets, len(DATASETS)))
    output = args.output_root

    # Recycle a worker after every dataset.  In particular, the FB15k-237
    # baseline contains tens of millions of dependency keys; terminating its
    # worker is the most reliable way to return Python allocator arenas to the
    # operating system before another dataset is processed.
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        futures = {
            executor.submit(
                process_dataset,
                dataset,
                args.work_root,
                args.rule_root,
                args.unseen_negative_examples,
            ): dataset
            for dataset in DATASETS
        }
        for future in as_completed(futures):
            dataset = futures[future]
            dependency_rows, gain_rows, saturation_rows, overlap_dataset_rows = future.result()
            dependency_stats.extend(dependency_rows)
            gain_stats.extend(gain_rows)
            saturation_stats.extend(saturation_rows)
            overlap_rows.extend(overlap_dataset_rows)
            dependency_stats.sort(key=lambda row: (DATASETS.index(row["dataset"]), CAPS.index(row["cap"])))
            gain_stats.sort(key=lambda row: (DATASETS.index(row["dataset"]), CAPS.index(row["cap"])))
            saturation_stats.sort(key=lambda row: (DATASETS.index(row["dataset"]), CAPS.index(row["cap"])))
            overlap_rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), CAPS.index(row["cap"])))
            write_csv(output / "cap_dependency_stats.csv", dependency_stats)
            write_csv(output / "cap_gain_distribution.csv", gain_stats)
            write_csv(output / "cap_overlap_vs_7.csv", overlap_rows)
            write_csv(output / "cap_saturation_stats.csv", saturation_stats)
            print(f"completed dataset={dataset}", flush=True)


if __name__ == "__main__":
    main()
