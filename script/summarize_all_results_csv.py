#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


METRIC_RE = re.compile(
    r"MRR\s+([0-9]*\.?[0-9]+).*?"
    r"hits@1\s+([0-9]*\.?[0-9]+).*?"
    r"hits@10\s+([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)

EVAL_LOGS = [
    ("eval-maxplus", "eval-maxplus.log"),
    ("eval-noisyor", "eval-noisyor.log"),
]

STRUCTURAL_STAGE1_PREFIXES = {
    "structural_rd",
    "structural_rd_filtered",
    "structural_r2d3",
    "structural_r2d3_filtered",
    "structural_r3d6",
    "structural_r3d6_filtered",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all aggregation/application results into one CSV.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="reports/all_results_summary.csv")
    parser.add_argument("--ensemble-debug", default="reports/all_results_ensemble_debug.json")
    return parser.parse_args()


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def dataset_dirs(data_root: Path) -> List[Path]:
    rows = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "aggregation").exists() or (child / "application").exists():
            rows.append(child)
    return rows


def parse_eval_log(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {"MRR": None, "h@1": None, "h@10": None}
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = METRIC_RE.search(text)
    if not match:
        return {"MRR": None, "h@1": None, "h@10": None}
    return {
        "MRR": float(match.group(1)),
        "h@1": float(match.group(2)),
        "h@10": float(match.group(3)),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_triplet(node: Optional[Dict[str, object]]) -> Dict[str, Optional[float]]:
    if not node:
        return {"MRR": None, "h@1": None, "h@10": None}
    return {
        "MRR": to_float(node.get("mrr")),
        "h@1": to_float(node.get("h1")),
        "h@10": to_float(node.get("h10")),
    }


def build_experiment_rows(dataset: str, exp_dir: Path) -> List[Dict[str, object]]:
    metrics_final = exp_dir / "metrics-final.json"
    if not metrics_final.exists():
        return []

    payload = load_json(metrics_final)
    summary = payload.get("summary") or {}
    sweep_time = to_float((payload.get("time_seconds") or {}).get("sweep"))
    agg_name = exp_dir.name
    rows: List[Dict[str, object]] = []

    stage2_metrics = metric_triplet(summary.get("test") or summary.get("test_after_stage2"))
    rows.append(
        {
            "dataset": dataset,
            "aggregation": agg_name,
            "MRR": stage2_metrics["MRR"],
            "h@1": stage2_metrics["h@1"],
            "h@10": stage2_metrics["h@10"],
            "time": sweep_time,
        }
    )

    if agg_name in STRUCTURAL_STAGE1_PREFIXES:
        stage1_metrics = metric_triplet(summary.get("test_after_stage1"))
        rows.append(
            {
                "dataset": dataset,
                "aggregation": f"{agg_name}__stage1",
                "MRR": stage1_metrics["MRR"],
                "h@1": stage1_metrics["h@1"],
                "h@10": stage1_metrics["h@10"],
                "time": sweep_time,
            }
        )

    return rows


def filtered_preferred_experiment_dirs(agg_dir: Path) -> List[Path]:
    dirs = sorted([p for p in agg_dir.iterdir() if p.is_dir()])
    names = {p.name for p in dirs}
    selected: List[Path] = []
    for exp_dir in dirs:
        name = exp_dir.name
        if name.endswith("_filtered"):
            selected.append(exp_dir)
            continue
        if f"{name}_filtered" in names:
            continue
        selected.append(exp_dir)
    return selected


def aggregate_weighted(rows: Iterable[Dict[str, object]]) -> Dict[str, Optional[float]]:
    rows = list(rows)
    total_weight = sum(int(r["count"]) for r in rows if int(r["count"]) > 0)
    if total_weight <= 0:
        return {"MRR": None, "h@1": None, "h@10": None, "time": None}
    return {
        "MRR": sum(float(r["MRR"]) * int(r["count"]) for r in rows) / total_weight,
        "h@1": sum(float(r["h@1"]) * int(r["count"]) for r in rows) / total_weight,
        "h@10": sum(float(r["h@10"]) * int(r["count"]) for r in rows) / total_weight,
        "time": sum(float(r["time"]) for r in rows),
    }


def build_ensemble_row(dataset: str, agg_dir: Path) -> tuple[Optional[Dict[str, object]], Dict[str, object]]:
    best_by_relation: Dict[int, Dict[str, object]] = {}
    debug: Dict[str, object] = {"dataset": dataset, "selected_relations": {}}

    for exp_dir in filtered_preferred_experiment_dirs(agg_dir):
        metric_files = sorted(exp_dir.glob("metric-*.json"))
        if not metric_files:
            continue
        for metric_path in metric_files:
            metric = load_json(metric_path)
            relation = int(metric["relation"])
            valid_stage2 = metric.get("best_valid_stage2") or {}
            best_valid_mrr = to_float(valid_stage2.get("mrr"))
            final_test = metric.get("test") or metric.get("test_after_stage2")
            num_test_samples = int(metric.get("num_test_samples", 0))
            relation_time = to_float((metric.get("time_seconds") or {}).get("total")) or 0.0
            if best_valid_mrr is None or not final_test or num_test_samples <= 0:
                continue
            current = best_by_relation.get(relation)
            if current is None or best_valid_mrr > float(current["best_valid_mrr"]):
                best_by_relation[relation] = {
                    "experiment": exp_dir.name,
                    "best_valid_mrr": best_valid_mrr,
                    "MRR": float(final_test["mrr"]),
                    "h@1": float(final_test["h1"]),
                    "h@10": float(final_test["h10"]),
                    "count": num_test_samples,
                    "time": relation_time,
                }

    if not best_by_relation:
        return None, debug

    for relation, row in sorted(best_by_relation.items()):
        debug["selected_relations"][str(relation)] = {
            "experiment": row["experiment"],
            "best_valid_stage2_mrr": row["best_valid_mrr"],
            "final_test_mrr": row["MRR"],
            "count": row["count"],
        }

    agg = aggregate_weighted(best_by_relation.values())
    return (
        {
            "dataset": dataset,
            "aggregation": "ensemble_best_valid_stage2",
            "MRR": agg["MRR"],
            "h@1": agg["h@1"],
            "h@10": agg["h@10"],
            "time": agg["time"],
        },
        debug,
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = Path(args.ensemble_debug)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    ensemble_debug: Dict[str, object] = {}

    for ds_dir in dataset_dirs(data_root):
        dataset = ds_dir.name
        app_dir = ds_dir / "application"
        agg_dir = ds_dir / "aggregation"

        for agg_name, log_name in EVAL_LOGS:
            metrics = parse_eval_log(app_dir / log_name)
            rows.append(
                {
                    "dataset": dataset,
                    "aggregation": agg_name,
                    "MRR": metrics["MRR"],
                    "h@1": metrics["h@1"],
                    "h@10": metrics["h@10"],
                    "time": None,
                }
            )

        if agg_dir.exists():
            for exp_dir in filtered_preferred_experiment_dirs(agg_dir):
                rows.extend(build_experiment_rows(dataset, exp_dir))
            ensemble_row, debug = build_ensemble_row(dataset, agg_dir)
            ensemble_debug[dataset] = debug
            if ensemble_row is not None:
                rows.append(ensemble_row)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "aggregation", "MRR", "h@1", "h@10", "time"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "aggregation": row["aggregation"],
                    "MRR": fmt(row["MRR"]),
                    "h@1": fmt(row["h@1"]),
                    "h@10": fmt(row["h@10"]),
                    "time": fmt(row["time"]),
                }
            )

    with debug_path.open("w", encoding="utf-8") as handle:
        json.dump(ensemble_debug, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Wrote ensemble debug to {debug_path}")


if __name__ == "__main__":
    main()
