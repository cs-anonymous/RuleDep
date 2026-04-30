#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from analyze_official_query_subsets import build_query_features, load_dependency_weights, numeric_feature_names, safe_mean


ROOT = Path("/home/sy/RuleDep")
DATASET = "codex-l"
DATA_DIR = ROOT / "data" / DATASET
EXAMPLE_ROOT = ROOT / "RuleDepDemo" / "frontend" / "public" / "example" / DATASET
OUT_DIR = ROOT / "reports" / "official_query_subset" / "codex_l_experiment_coverage"


def read_relation_names() -> list[str]:
    names = []
    with (DATA_DIR / "relation_ids.del").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def load_official_relation_metrics(experiment: str) -> dict[str, dict[str, float]]:
    relation_names = read_relation_names()
    exp_dir = DATA_DIR / "aggregation" / experiment
    out = {}
    for metric_path in exp_dir.glob("metric-*.json"):
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        relation_id = int(metric["relation"])
        if relation_id >= len(relation_names):
            continue
        stage1 = metric.get("test_after_stage1") or {}
        stage2 = metric.get("test_after_stage2") or stage1
        if "mrr" not in stage1 or "mrr" not in stage2:
            continue
        out[relation_names[relation_id]] = {
            "official_stage1_mrr": float(stage1["mrr"]),
            "official_stage2_mrr": float(stage2["mrr"]),
            "num_test_samples": float(metric.get("num_test_samples") or 0.0),
        }
    return out


def discover_experiments() -> list[str]:
    if not EXAMPLE_ROOT.exists():
        return []
    return sorted(p.name for p in EXAMPLE_ROOT.iterdir() if (p / "relations.json").exists())


def load_experiment_rows(experiment: str) -> list[dict]:
    exp_root = EXAMPLE_ROOT / experiment
    relations_path = exp_root / "relations.json"
    relations = json.loads(relations_path.read_text(encoding="utf-8")).get("relations", [])
    dep_weights = load_dependency_weights(DATASET)
    feature_cache = {}
    rows = []

    for idx, relation in enumerate(relations, start=1):
        rel_dir = exp_root / relation.replace("/", "_")
        qpath = rel_dir / "queries.json"
        if not qpath.exists():
            continue
        qitems = json.loads(qpath.read_text(encoding="utf-8")).get("queries", [])
        print(f"[{experiment}] relation {idx}/{len(relations)} {relation} queries={len(qitems)}", flush=True)
        for q in qitems:
            r1 = float(q.get("gtRankStage1") or 0.0)
            r2 = float(q.get("gtRankStage2") or 0.0)
            if r1 <= 0 or r2 <= 0:
                continue
            fpath = rel_dir / q["filename"]
            if not fpath.exists():
                continue
            key = str(fpath)
            if key not in feature_cache:
                cands = json.loads(fpath.read_text(encoding="utf-8"))
                if not cands:
                    continue
                feature_cache[key] = build_query_features(cands, q, dep_weights)
            row = {
                "dataset": DATASET,
                "experiment": experiment,
                "relation": relation,
                "raw_rr_stage1": 1.0 / r1,
                "raw_rr_stage2": 1.0 / r2,
                "raw_delta_rr": (1.0 / r2) - (1.0 / r1),
            }
            row.update(feature_cache[key])
            rows.append(row)
    return rows


def add_official_scaled_rr(rows: list[dict], experiment: str) -> list[dict]:
    official = load_official_relation_metrics(experiment)
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)

    for relation, group in by_relation.items():
        metrics = official.get(relation)
        raw_s1 = safe_mean([float(r["raw_rr_stage1"]) for r in group])
        raw_s2 = safe_mean([float(r["raw_rr_stage2"]) for r in group])
        if metrics and raw_s1 > 0 and raw_s2 > 0:
            scale1 = metrics["official_stage1_mrr"] / raw_s1
            scale2 = metrics["official_stage2_mrr"] / raw_s2
            official_s1 = metrics["official_stage1_mrr"]
            official_s2 = metrics["official_stage2_mrr"]
            num_test = metrics["num_test_samples"]
        else:
            scale1 = 1.0
            scale2 = 1.0
            official_s1 = raw_s1
            official_s2 = raw_s2
            num_test = float(len(group))
        for row in group:
            row["official_scaled_rr_stage1"] = float(row["raw_rr_stage1"]) * scale1
            row["official_scaled_rr_stage2"] = float(row["raw_rr_stage2"]) * scale2
            row["official_relation_stage1_mrr"] = official_s1
            row["official_relation_stage2_mrr"] = official_s2
            row["official_relation_gain_pt"] = (official_s2 / official_s1 - 1.0) if official_s1 > 0 else 0.0
            row["relation_scale_stage1"] = scale1
            row["relation_scale_stage2"] = scale2
            row["num_test_samples"] = num_test
    return rows


def gain_for_subset(subset: list[dict], stage1_key: str, stage2_key: str) -> tuple[float, float, float]:
    mrr1 = safe_mean([float(r[stage1_key]) for r in subset])
    mrr2 = safe_mean([float(r[stage2_key]) for r in subset])
    return mrr1, mrr2, (mrr2 / mrr1 - 1.0) if mrr1 > 0 else 0.0


def ranking_feature_names(rows: list[dict]) -> list[str]:
    excluded = {
        "raw_rr_stage1",
        "raw_rr_stage2",
        "raw_delta_rr",
        "official_scaled_rr_stage1",
        "official_scaled_rr_stage2",
        "official_relation_stage1_mrr",
        "official_relation_stage2_mrr",
        "official_relation_gain_pt",
        "relation_scale_stage1",
        "relation_scale_stage2",
        "num_test_samples",
    }
    return [name for name in numeric_feature_names(rows) if name not in excluded]


def build_feature_curves(rows: list[dict], coverage: float = 0.20) -> tuple[list[dict], list[dict]]:
    features = ranking_feature_names(rows)
    summary = []
    checks = []

    raw_full = gain_for_subset(rows, "raw_rr_stage1", "raw_rr_stage2")
    scaled_full = gain_for_subset(rows, "official_scaled_rr_stage1", "official_scaled_rr_stage2")
    checks.append(
        {
            "experiment": rows[0]["experiment"],
            "n": len(rows),
            "raw_stage1_mrr": raw_full[0],
            "raw_stage2_mrr": raw_full[1],
            "raw_gain_pt": raw_full[2],
            "official_scaled_stage1_mrr": scaled_full[0],
            "official_scaled_stage2_mrr": scaled_full[1],
            "official_scaled_gain_pt": scaled_full[2],
        }
    )

    n = max(1, int(round(len(rows) * coverage)))
    for feature in features:
        for direction in ("desc", "asc"):
            ordered = sorted(rows, key=lambda r: float(r.get(feature, 0.0)), reverse=(direction == "desc"))
            subset = ordered[:n]
            mrr1, mrr2, gain = gain_for_subset(subset, "official_scaled_rr_stage1", "official_scaled_rr_stage2")
            summary.append(
                {
                    "experiment": rows[0]["experiment"],
                    "coverage": coverage,
                    "n": n,
                    "feature": feature,
                    "sort_direction": direction,
                    "mrr_stage1": mrr1,
                    "mrr_stage2": mrr2,
                    "gain_pt": gain,
                    "threshold": float(subset[-1].get(feature, 0.0)),
                    "positive": gain > 0,
                }
            )
    summary.sort(key=lambda r: float(r["gain_pt"]), reverse=True)
    return summary, checks


def weighted_official_full_gain(experiment: str) -> tuple[float, float, float]:
    metrics = load_official_relation_metrics(experiment)
    total = sum(v["num_test_samples"] for v in metrics.values())
    if total <= 0:
        return math.nan, math.nan, math.nan
    s1 = sum(v["official_stage1_mrr"] * v["num_test_samples"] for v in metrics.values()) / total
    s2 = sum(v["official_stage2_mrr"] * v["num_test_samples"] for v in metrics.values()) / total
    return s1, s2, (s2 / s1 - 1.0) if s1 > 0 else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary_rows: list[dict], check_rows: list[dict], out_path: Path) -> None:
    lines = [
        "# CODEX-L Experiment Coverage Check",
        "",
        "Metric: per-relation multiplicatively scaled query RR. This keeps query-level ordering/subset variation but makes 100% coverage match `metric-*.json` official Stage1/Stage2 MRR.",
        "",
        "## 100% Coverage Check",
        "",
        "| experiment | rows | official metric gain | scaled 100% gain | raw demo 100% gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in check_rows:
        official_s1, official_s2, official_gain = weighted_official_full_gain(row["experiment"])
        lines.append(
            f"| `{row['experiment']}` | {int(row['n'])} | {official_gain * 100:.2f}% | "
            f"{row['official_scaled_gain_pt'] * 100:.2f}% | {row['raw_gain_pt'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Top 20% Coverage Features",
            "",
            "| experiment | rank | feature | order | gain_pt | mrr_stage1 | mrr_stage2 |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    by_exp = defaultdict(list)
    for row in summary_rows:
        by_exp[row["experiment"]].append(row)
    for experiment in sorted(by_exp):
        for rank, row in enumerate(by_exp[experiment][:20], start=1):
            lines.append(
                f"| `{experiment}` | {rank} | `{row['feature']}` | {row['sort_direction']} | "
                f"{row['gain_pt'] * 100:.2f}% | {row['mrr_stage1']:.6f} | {row['mrr_stage2']:.6f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", help="CODEX-L demo experiment name. Repeatable.")
    parser.add_argument("--coverage", type=float, default=0.20)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--reuse-features", action="store_true", help="reuse existing per-experiment query feature CSVs")
    args = parser.parse_args()

    experiments = args.experiment or discover_experiments()
    out_dir = Path(args.out_dir)
    all_summary = []
    all_checks = []

    for experiment in experiments:
        print(f"[load] {experiment}", flush=True)
        feature_path = out_dir / f"{experiment}__query_features.csv"
        if args.reuse_features and feature_path.exists():
            rows = pd.read_csv(feature_path).to_dict("records")
        else:
            rows = load_experiment_rows(experiment)
            rows = add_official_scaled_rr(rows, experiment)
            write_csv(feature_path, rows)
        summary, checks = build_feature_curves(rows, coverage=args.coverage)
        all_summary.extend(summary)
        all_checks.extend(checks)

    write_csv(out_dir / "codex_l_20pct_feature_summary.csv", all_summary)
    write_csv(out_dir / "codex_l_100pct_metric_check.csv", all_checks)
    write_report(all_summary, all_checks, out_dir / "README.md")
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
