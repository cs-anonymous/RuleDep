#!/usr/bin/env python3
"""Estimate higher-order evidence in relation-local three-rule combinations.

The script follows RuleDep's log-failure evidence definition and deliberately
uses the term "three-rule combination" rather than "triple" to avoid confusion
with knowledge-graph (head, relation, tail) triples.

For each relation-local aggregation dataset it:
  1. keeps the top-K eligible rules by smoothed evidence;
  2. counts distinct three-rule combinations observed on positive training rows;
  3. retains combinations with joint support >= min_support; and
  4. counts combinations whose conservative gain interval lies completely
     outside the gain intervals of all three constituent rule pairs.

The confidence intervals use Wilson intervals on the smoothed Bernoulli counts
(the RuleDep pseudo-negative count is included in the denominator). Interval
arithmetic then gives conservative intervals for pair and three-rule gains.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import math
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np


RuleCombination = Tuple[int, int, int]
RulePair = Tuple[int, int]
RULE_EVIDENCE_BY_DATASET: Dict[str, np.ndarray] = {}


def evidence(confidence: float, cap: float) -> float:
    confidence = min(max(float(confidence), 0.0), 1.0 - 1.0e-15)
    return min(-math.log1p(-confidence), cap)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evidence_interval(successes: int, body_count: int, unseen: int, cap: float) -> Tuple[float, float]:
    lo, hi = wilson_interval(successes, body_count + unseen)
    return evidence(lo, cap), evidence(hi, cap)


def gain_interval(
    joint: Tuple[float, float],
    marginals: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    lower = joint[0] - sum(interval[1] for interval in marginals)
    upper = joint[1] - sum(interval[0] for interval in marginals)
    return lower, upper


def iter_rows(split: dict) -> Iterator[Tuple[int, np.ndarray]]:
    flat = split["rules_flat"].numpy()
    offsets = split["offsets"].numpy()
    golds = split["golds"].reshape(-1).numpy()
    for row_id in range(len(golds)):
        start = int(offsets[row_id])
        end = int(offsets[row_id + 1])
        yield int(golds[row_id] > 0.5), flat[start:end]


def rule_evidence_table(rule_path: Path, unseen: int, cap: float) -> np.ndarray:
    values: List[float] = []
    with rule_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) < 4:
                values.append(0.0)
                continue
            body_size = int(parts[0])
            support = int(parts[1])
            confidence = support / (body_size + unseen) if body_size + unseen > 0 else 0.0
            values.append(evidence(confidence, cap))
    return np.asarray(values, dtype=np.float64)


def selected_relation_rules(split: dict, rule_evidence: np.ndarray, top_k: int, min_evidence: float) -> Set[int]:
    flat = split["rules_flat"].numpy()
    appearing = np.unique(flat)
    appearing = appearing[(appearing >= 0) & (appearing < len(rule_evidence))]
    eligible = appearing[rule_evidence[appearing] >= min_evidence]
    if top_k > 0 and len(eligible) > top_k:
        order = np.argpartition(rule_evidence[eligible], -top_k)[-top_k:]
        eligible = eligible[order]
    return set(int(x) for x in eligible)


def selected_row_rules(row_rules: np.ndarray, selected: Set[int]) -> List[int]:
    return sorted({int(rule) for rule in row_rules if int(rule) in selected})


def count_positive_combinations(
    split: dict,
    selected: Set[int],
) -> Tuple[Counter[RulePair], Counter[RuleCombination]]:
    pair_support: Counter[RulePair] = Counter()
    combination_support: Counter[RuleCombination] = Counter()
    for gold, row_rules in iter_rows(split):
        if not gold:
            continue
        rules = selected_row_rules(row_rules, selected)
        n = len(rules)
        for i in range(n - 1):
            a = rules[i]
            for j in range(i + 1, n):
                pair_support[(a, rules[j])] += 1
        for i in range(n - 2):
            a = rules[i]
            for j in range(i + 1, n - 1):
                b = rules[j]
                for k in range(j + 1, n):
                    combination_support[(a, b, rules[k])] += 1
    return pair_support, combination_support


def build_inverted_rows(split: dict, required_rules: Set[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    all_rows: Dict[int, List[int]] = defaultdict(list)
    positive_rows: Dict[int, List[int]] = defaultdict(list)
    for row_id, (gold, row_rules) in enumerate(iter_rows(split)):
        seen: Set[int] = set()
        for raw_rule in row_rules:
            rule = int(raw_rule)
            if rule not in required_rules or rule in seen:
                continue
            seen.add(rule)
            all_rows[rule].append(row_id)
            if gold:
                positive_rows[rule].append(row_id)
    all_arrays = {rule: np.asarray(rows, dtype=np.int64) for rule, rows in all_rows.items()}
    positive_arrays = {rule: np.asarray(rows, dtype=np.int64) for rule, rows in positive_rows.items()}
    return all_arrays, positive_arrays


def intersection_size(arrays: Sequence[np.ndarray]) -> int:
    if not arrays or any(len(array) == 0 for array in arrays):
        return 0
    result = arrays[0]
    for array in arrays[1:]:
        result = np.intersect1d(result, array, assume_unique=True)
        if len(result) == 0:
            return 0
    return int(len(result))


def analyze_relation(
    dataset: str,
    relation_file: Path,
    rule_evidence: np.ndarray,
    top_k: int,
    min_evidence: float,
    min_support: int,
    unseen: int,
    cap: float,
    gain_margin: float,
) -> dict:
    with relation_file.open("rb") as handle:
        obj = pickle.load(handle)
    split = obj["train"]
    selected = selected_relation_rules(split, rule_evidence, top_k, min_evidence)
    pair_support_counts, support_counts = count_positive_combinations(split, selected)
    observed_pairs = len(pair_support_counts)
    qualified_pair_support = {
        pair: support
        for pair, support in pair_support_counts.items()
        if support >= min_support
    }
    qualified_pairs = len(qualified_pair_support)
    observed = len(support_counts)
    qualified_support = {
        combination: support
        for combination, support in support_counts.items()
        if support >= min_support
    }
    qualified = len(qualified_support)

    significant_pairs_positive = 0
    significant_pairs_negative = 0
    significant_positive = 0
    significant_negative = 0
    if qualified_pairs or qualified:
        required_rules = {rule for pair in qualified_pair_support for rule in pair}
        required_rules.update(rule for combination in qualified_support for rule in combination)
        all_rows, positive_rows = build_inverted_rows(split, required_rules)
        singleton_intervals: Dict[int, Tuple[float, float]] = {}
        pair_intervals: Dict[RulePair, Tuple[float, float]] = {}

        def singleton_interval(rule: int) -> Tuple[float, float]:
            if rule not in singleton_intervals:
                singleton_intervals[rule] = evidence_interval(
                    len(positive_rows.get(rule, ())),
                    len(all_rows.get(rule, ())),
                    unseen,
                    cap,
                )
            return singleton_intervals[rule]

        def pair_gain(rule_a: int, rule_b: int) -> Tuple[float, float]:
            pair = (rule_a, rule_b) if rule_a < rule_b else (rule_b, rule_a)
            if pair not in pair_intervals:
                body_count = intersection_size([all_rows.get(pair[0], np.empty(0, dtype=np.int64)), all_rows.get(pair[1], np.empty(0, dtype=np.int64))])
                support = intersection_size([positive_rows.get(pair[0], np.empty(0, dtype=np.int64)), positive_rows.get(pair[1], np.empty(0, dtype=np.int64))])
                pair_joint = evidence_interval(support, body_count, unseen, cap)
                pair_intervals[pair] = gain_interval(
                    pair_joint,
                    [singleton_interval(pair[0]), singleton_interval(pair[1])],
                )
            return pair_intervals[pair]

        for pair in qualified_pair_support:
            interval = pair_gain(*pair)
            if interval[0] > gain_margin:
                significant_pairs_positive += 1
            elif interval[1] < -gain_margin:
                significant_pairs_negative += 1

        for combination, support in qualified_support.items():
            a, b, c = combination
            body_count = intersection_size([
                all_rows.get(a, np.empty(0, dtype=np.int64)),
                all_rows.get(b, np.empty(0, dtype=np.int64)),
                all_rows.get(c, np.empty(0, dtype=np.int64)),
            ])
            combination_joint = evidence_interval(support, body_count, unseen, cap)
            combination_gain = gain_interval(
                combination_joint,
                [singleton_interval(a), singleton_interval(b), singleton_interval(c)],
            )
            constituent_pair_gains = [pair_gain(a, b), pair_gain(a, c), pair_gain(b, c)]
            pair_upper = max(interval[1] for interval in constituent_pair_gains)
            pair_lower = min(interval[0] for interval in constituent_pair_gains)
            if combination_gain[0] > pair_upper + gain_margin:
                significant_positive += 1
            elif combination_gain[1] < pair_lower - gain_margin:
                significant_negative += 1

    significant = significant_positive + significant_negative
    significant_pairs = significant_pairs_positive + significant_pairs_negative
    relation_id = relation_file.stem.removeprefix("dataset_")
    possible_pairs = math.comb(len(selected), 2) if len(selected) >= 2 else 0
    possible = math.comb(len(selected), 3) if len(selected) >= 3 else 0
    return {
        "dataset": dataset,
        "relation": relation_id,
        "selected_rules": len(selected),
        "possible_pairs": possible_pairs,
        "observed_pairs": observed_pairs,
        "support_qualified_pairs": qualified_pairs,
        "significant_pairs": significant_pairs,
        "significant_pairs_positive": significant_pairs_positive,
        "significant_pairs_negative": significant_pairs_negative,
        "observed_pairs_over_possible": observed_pairs / possible_pairs if possible_pairs else 0.0,
        "qualified_pairs_over_observed": qualified_pairs / observed_pairs if observed_pairs else 0.0,
        "significant_pairs_over_qualified": significant_pairs / qualified_pairs if qualified_pairs else 0.0,
        "possible_combinations": possible,
        "observed_combinations": observed,
        "support_qualified_combinations": qualified,
        "significant_combinations": significant,
        "significant_positive": significant_positive,
        "significant_negative": significant_negative,
        "observed_over_possible": observed / possible if possible else 0.0,
        "qualified_over_observed": qualified / observed if observed else 0.0,
        "significant_over_qualified": significant / qualified if qualified else 0.0,
    }


def aggregate_rows(rows: Sequence[dict], dataset: str) -> dict:
    possible_pairs = sum(int(row["possible_pairs"]) for row in rows)
    observed_pairs = sum(int(row["observed_pairs"]) for row in rows)
    qualified_pairs = sum(int(row["support_qualified_pairs"]) for row in rows)
    significant_pairs_positive = sum(int(row["significant_pairs_positive"]) for row in rows)
    significant_pairs_negative = sum(int(row["significant_pairs_negative"]) for row in rows)
    significant_pairs = significant_pairs_positive + significant_pairs_negative
    observed = sum(int(row["observed_combinations"]) for row in rows)
    possible = sum(int(row["possible_combinations"]) for row in rows)
    qualified = sum(int(row["support_qualified_combinations"]) for row in rows)
    significant_positive = sum(int(row["significant_positive"]) for row in rows)
    significant_negative = sum(int(row["significant_negative"]) for row in rows)
    significant = significant_positive + significant_negative
    return {
        "dataset": dataset,
        "relation": "ALL",
        "selected_rules": sum(int(row["selected_rules"]) for row in rows),
        "possible_pairs": possible_pairs,
        "observed_pairs": observed_pairs,
        "support_qualified_pairs": qualified_pairs,
        "significant_pairs": significant_pairs,
        "significant_pairs_positive": significant_pairs_positive,
        "significant_pairs_negative": significant_pairs_negative,
        "observed_pairs_over_possible": observed_pairs / possible_pairs if possible_pairs else 0.0,
        "qualified_pairs_over_observed": qualified_pairs / observed_pairs if observed_pairs else 0.0,
        "significant_pairs_over_qualified": significant_pairs / qualified_pairs if qualified_pairs else 0.0,
        "possible_combinations": possible,
        "observed_combinations": observed,
        "support_qualified_combinations": qualified,
        "significant_combinations": significant,
        "significant_positive": significant_positive,
        "significant_negative": significant_negative,
        "observed_over_possible": observed / possible if possible else 0.0,
        "qualified_over_observed": qualified / observed if observed else 0.0,
        "significant_over_qualified": significant / qualified if qualified else 0.0,
    }


def analyze_relation_task(task: tuple) -> dict:
    (
        dataset,
        relation_file,
        top_k,
        min_evidence,
        min_support,
        unseen,
        cap,
        gain_margin,
    ) = task
    return analyze_relation(
        dataset,
        Path(relation_file),
        RULE_EVIDENCE_BY_DATASET[dataset],
        top_k,
        min_evidence,
        min_support,
        unseen,
        cap,
        gain_margin,
    )


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relation-local three-rule combinations")
    parser.add_argument("datasets", nargs="+", help="Dataset directory names under data/")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--top-k", type=int, default=500, help="Maximum eligible rules per relation")
    parser.add_argument("--min-evidence", type=float, default=0.05)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--small-min-support", type=int, default=2)
    parser.add_argument("--small-datasets", default="KG20C,WN18RR")
    parser.add_argument("--unseen", type=int, default=3)
    parser.add_argument("--evidence-cap", type=float, default=7.0)
    parser.add_argument("--gain-margin", type=float, default=0.01)
    parser.add_argument("--relation", default="", help="Optional relation id for a diagnostic run")
    parser.add_argument("--jobs", type=int, default=36, help="Parallel (dataset, relation) workers")
    parser.add_argument(
        "--output",
        default="reports/high_order_analysis/pair_vs_three_rule_combination_stats.csv",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    small_datasets = {item.strip() for item in args.small_datasets.split(",") if item.strip()}
    tasks: List[tuple] = []
    for dataset in args.datasets:
        ds_dir = data_root / dataset
        RULE_EVIDENCE_BY_DATASET[dataset] = rule_evidence_table(
            ds_dir / "rules" / "rule.txt", args.unseen, args.evidence_cap
        )
        relation_files = sorted(Path(path) for path in glob.glob(str(ds_dir / "datasets" / "dataset_*.p")))
        if args.relation:
            relation_files = [path for path in relation_files if path.stem == f"dataset_{args.relation}"]
        min_support = args.small_min_support if dataset in small_datasets else args.min_support
        for relation_file in relation_files:
            tasks.append((
                dataset,
                str(relation_file),
                args.top_k,
                args.min_evidence,
                min_support,
                args.unseen,
                args.evidence_cap,
                args.gain_margin,
            ))

    all_rows: List[dict] = []
    max_workers = max(1, min(int(args.jobs), len(tasks), os.cpu_count() or 1))
    print(f"Running {len(tasks)} relation tasks with {max_workers} workers", flush=True)
    if max_workers == 1:
        completed_rows = map(analyze_relation_task, tasks)
        for row in completed_rows:
            all_rows.append(row)
            print(
                f"{row['dataset']} relation={row['relation']} selected={row['selected_rules']} "
                f"pairs={row['observed_pairs']}/{row['support_qualified_pairs']}/{row['significant_pairs']} "
                f"combinations={row['observed_combinations']}/{row['support_qualified_combinations']}/"
                f"{row['significant_combinations']}",
                flush=True,
            )
            gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(analyze_relation_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                dataset = task[0]
                relation_file = Path(task[1])
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed dataset={dataset} relation_file={relation_file}") from exc
                all_rows.append(row)
                print(
                    f"{dataset} relation={row['relation']} selected={row['selected_rules']} "
                    f"pairs={row['observed_pairs']}/{row['support_qualified_pairs']}/{row['significant_pairs']} "
                    f"combinations={row['observed_combinations']}/{row['support_qualified_combinations']}/"
                    f"{row['significant_combinations']}",
                    flush=True,
                )

    all_rows.sort(key=lambda row: (row["dataset"], int(row["relation"])))
    summary_rows: List[dict] = []
    for dataset in args.datasets:
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset]
        summary = aggregate_rows(dataset_rows, dataset)
        summary_rows.append(summary)
        print(
            f"SUMMARY {dataset}: pairs observed={summary['observed_pairs']} "
            f"of possible={summary['possible_pairs']} ({summary['observed_pairs_over_possible']:.6%}), "
            f"qualified={summary['support_qualified_pairs']}, significant={summary['significant_pairs']} "
            f"({summary['significant_pairs_over_qualified']:.4%}); combinations observed="
            f"{summary['observed_combinations']} "
            f"of possible={summary['possible_combinations']} ({summary['observed_over_possible']:.6%}), "
            f"qualified={summary['support_qualified_combinations']} "
            f"({summary['qualified_over_observed']:.4%}), significant={summary['significant_combinations']} "
            f"({summary['significant_over_qualified']:.4%})",
            flush=True,
        )

    output = Path(args.output)
    write_csv(output, all_rows + summary_rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
