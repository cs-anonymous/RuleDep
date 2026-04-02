#!/usr/bin/env python3
import csv
import json
from pathlib import Path

DATASETS = [
    "KG20C",
    "codex-m",
    "WN18RR",
    "FB15k-237",
    "codex-l",
    "YAGO3-10",
]

EXPERIMENTS = [
    ("R", "baseline", "structural_rd", "test_after_stage1"),
    ("R2", "baseline", "structural_r2d3", "test_after_stage1"),
    ("R3", "baseline", "structural_r3d6", "test_after_stage1"),
    ("rd", "stage2", "structural_rd", "test_after_stage2"),
    ("r2d3", "stage2", "structural_r2d3", "test_after_stage2"),
    ("r3d6", "stage2", "structural_r3d6", "test_after_stage2"),
    ("synergy", "stage2", "synergy", "test_after_stage2"),
    ("redundancy", "stage2", "redundancy", "test_after_stage2"),
    ("sign_constraint_dependency", "stage2", "sign_constraint_dependency", "test_after_stage2"),
    ("init_dep_with_lift", "stage2", "init_dep_with_lift", "test_after_stage2"),
    ("pos_auto_ratio", "stage2", "pos_auto_ratio", "test_after_stage2"),
]

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "reports" / "experiment_summary" / "six_dataset_stage2_baseline_summary.csv"


def load_metrics(dataset: str, config_dir: str):
    path = ROOT / "data" / dataset / "aggregation" / config_dir / "metrics-final.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_round(value):
    if value is None:
        return ""
    return f"{float(value):.5f}"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "experiment",
        "group",
        "source_dir",
        "metric_source",
        "mrr",
        "h@1",
        "h@10",
        "mrr_raw",
        "h@1_raw",
        "h@10_raw",
        "time_train",
        "time_valid",
        "time_total",
        "time_load_dataloaders",
        "time_other",
        "time_sweep",
        "num_relations",
        "total_test_triples_used_for_weight",
    ]

    rows = []
    for dataset in DATASETS:
        for exp_name, exp_group, config_dir, metric_key in EXPERIMENTS:
            metrics = load_metrics(dataset, config_dir)
            summary = metrics["summary"]
            metric_obj = summary[metric_key]
            time_obj = metrics.get("time_seconds", {})

            rows.append(
                {
                    "dataset": dataset,
                    "experiment": exp_name,
                    "group": exp_group,
                    "source_dir": config_dir,
                    "metric_source": metric_key,
                    "mrr": maybe_round(metric_obj.get("mrr")),
                    "h@1": maybe_round(metric_obj.get("h1")),
                    "h@10": maybe_round(metric_obj.get("h10")),
                    "mrr_raw": maybe_round(metric_obj.get("mrr_raw")),
                    "h@1_raw": maybe_round(metric_obj.get("h1_raw")),
                    "h@10_raw": maybe_round(metric_obj.get("h10_raw")),
                    "time_train": maybe_round(time_obj.get("train")),
                    "time_valid": maybe_round(time_obj.get("eval")),
                    "time_total": maybe_round(time_obj.get("total")),
                    "time_load_dataloaders": maybe_round(time_obj.get("load_dataloaders")),
                    "time_other": maybe_round(time_obj.get("other")),
                    "time_sweep": maybe_round(time_obj.get("sweep")),
                    "num_relations": summary.get("num_relations", ""),
                    "total_test_triples_used_for_weight": summary.get("total_test_triples_used_for_weight", ""),
                }
            )

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(OUT_PATH)


if __name__ == "__main__":
    main()
