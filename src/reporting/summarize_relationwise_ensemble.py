#!/usr/bin/env python3
import csv
import json
from collections import Counter
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
    ("R", "baseline", "structural_none", "best_valid_stage1", "test_after_stage1"),
    ("R2", "baseline", "structural_r2d3", "best_valid_stage1", "test_after_stage1"),
    ("R3", "baseline", "structural_r3d6", "best_valid_stage1", "test_after_stage1"),
    ("rd", "stage2", "structural_rd", "best_valid_stage2", "test_after_stage2"),
    ("r2d3", "stage2", "structural_r2d3", "best_valid_stage2", "test_after_stage2"),
    ("r3d6", "stage2", "structural_r3d6", "best_valid_stage2", "test_after_stage2"),
    ("synergy", "stage2", "synergy", "best_valid_stage2", "test_after_stage2"),
    ("redundancy", "stage2", "redundancy", "best_valid_stage2", "test_after_stage2"),
    ("sign_constraint_dependency", "stage2", "sign_constraint_dependency", "best_valid_stage2", "test_after_stage2"),
    ("init_dep_with_lift", "stage2", "init_dep_with_lift", "best_valid_stage2", "test_after_stage2"),
    ("pos_auto_ratio", "stage2", "pos_auto_ratio", "best_valid_stage2", "test_after_stage2"),
]

ROOT = Path(__file__).resolve().parent.parent
OUT_SUMMARY = ROOT / "reports" / "experiment_summary" / "six_dataset_relationwise_ensemble_summary.csv"
OUT_DETAIL = ROOT / "reports" / "experiment_summary" / "six_dataset_relationwise_ensemble_detail.csv"


def r5(value):
    if value is None:
        return ""
    return f"{float(value):.5f}"


def load_relation_metric(dataset: str, exp_dir: str, relation: int):
    path = ROOT / "data" / dataset / "aggregation" / exp_dir / f"metric-{relation}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_selected(rows):
    total_weight = sum(int(row["num_test_samples"]) for row in rows)
    metrics = ["mrr", "h1", "h10", "mrr_raw", "h1_raw", "h10_raw"]
    out = {metric: 0.0 for metric in metrics}
    for row in rows:
        weight = int(row["num_test_samples"])
        for metric in metrics:
            out[metric] += float(row[metric]) * weight
    if total_weight > 0:
        for metric in metrics:
            out[metric] /= total_weight
    out["total_test_samples"] = total_weight
    return out


def main():
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    summary_rows = []

    for dataset in DATASETS:
        first_dir = ROOT / "data" / dataset / "aggregation" / EXPERIMENTS[0][2]
        relation_ids = sorted(
            int(path.stem.split("-")[1])
            for path in first_dir.glob("metric-*.json")
        )

        selected_rows = []
        selection_counter = Counter()

        for relation in relation_ids:
            candidates = []
            for exp_name, exp_group, exp_dir, valid_key, test_key in EXPERIMENTS:
                metric_obj = load_relation_metric(dataset, exp_dir, relation)
                if metric_obj is None:
                    continue
                valid_obj = metric_obj.get(valid_key) or {}
                test_obj = metric_obj.get(test_key) or {}
                if not valid_obj or not test_obj:
                    continue
                candidates.append(
                    {
                        "dataset": dataset,
                        "relation": relation,
                        "experiment": exp_name,
                        "group": exp_group,
                        "source_dir": exp_dir,
                        "valid_source": valid_key,
                        "test_source": test_key,
                        "valid_mrr": float(valid_obj["mrr"]),
                        "valid_h1": float(valid_obj["h1"]),
                        "valid_h10": float(valid_obj["h10"]),
                        "test_mrr": float(test_obj["mrr"]),
                        "test_h1": float(test_obj["h1"]),
                        "test_h10": float(test_obj["h10"]),
                        "mrr": float(test_obj["mrr"]),
                        "h1": float(test_obj["h1"]),
                        "h10": float(test_obj["h10"]),
                        "mrr_raw": float(test_obj["mrr_raw"]),
                        "h1_raw": float(test_obj["h1_raw"]),
                        "h10_raw": float(test_obj["h10_raw"]),
                        "num_test_samples": int(metric_obj["num_test_samples"]),
                    }
                )

            exp_order = {item[0]: idx for idx, item in enumerate(EXPERIMENTS)}
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda x: (x["valid_mrr"], -exp_order[x["experiment"]]),
            )
            selection_counter[best["experiment"]] += 1
            selected_rows.append(best)
            detail_rows.append(
                {
                    "dataset": dataset,
                    "relation": relation,
                    "experiment": best["experiment"],
                    "group": best["group"],
                    "source_dir": best["source_dir"],
                    "valid_source": best["valid_source"],
                    "test_source": best["test_source"],
                    "valid_mrr": r5(best["valid_mrr"]),
                    "valid_h@1": r5(best["valid_h1"]),
                    "valid_h@10": r5(best["valid_h10"]),
                    "test_mrr": r5(best["test_mrr"]),
                    "test_h@1": r5(best["test_h1"]),
                    "test_h@10": r5(best["test_h10"]),
                    "num_test_samples": best["num_test_samples"],
                }
            )

        if not selected_rows:
            continue
        agg = aggregate_selected(selected_rows)
        summary_row = {
            "dataset": dataset,
            "ensemble_test_mrr": r5(agg["mrr"]),
            "ensemble_test_h@1": r5(agg["h1"]),
            "ensemble_test_h@10": r5(agg["h10"]),
            "ensemble_test_mrr_raw": r5(agg["mrr_raw"]),
            "ensemble_test_h@1_raw": r5(agg["h1_raw"]),
            "ensemble_test_h@10_raw": r5(agg["h10_raw"]),
            "num_relations": len(relation_ids),
            "total_test_samples": agg["total_test_samples"],
        }
        for exp_name, *_rest in EXPERIMENTS:
            summary_row[f"pick_count_{exp_name}"] = selection_counter.get(exp_name, 0)
        summary_rows.append(summary_row)

    summary_fieldnames = [
        "dataset",
        "ensemble_test_mrr",
        "ensemble_test_h@1",
        "ensemble_test_h@10",
        "ensemble_test_mrr_raw",
        "ensemble_test_h@1_raw",
        "ensemble_test_h@10_raw",
        "num_relations",
        "total_test_samples",
    ] + [f"pick_count_{exp_name}" for exp_name, *_ in EXPERIMENTS]

    with open(OUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_fieldnames = [
        "dataset",
        "relation",
        "experiment",
        "group",
        "source_dir",
        "valid_source",
        "test_source",
        "valid_mrr",
        "valid_h@1",
        "valid_h@10",
        "test_mrr",
        "test_h@1",
        "test_h@10",
        "num_test_samples",
    ]
    with open(OUT_DETAIL, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    print(OUT_SUMMARY)
    print(OUT_DETAIL)


if __name__ == "__main__":
    main()
