#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/sy/RuleDep")
DEFAULT_EXPORT_DIR = ROOT / "reports" / "official_query_subset" / "true_official_per_query_rr"
KEYS = [
    "dataset",
    "experiment",
    "relation_id",
    "relation",
    "direction",
    "query_key",
    "known_entity_id",
    "known_entity",
    "target_entity_id",
    "target_gt_entity",
    "query",
]


def load_rows(export_dir: Path, stages: set[str] | None = None) -> pd.DataFrame:
    paths = sorted(glob.glob(str(export_dir / "*" / "*" / "relation-*.csv")))
    if not paths:
        raise FileNotFoundError(f"No relation stage CSV files found under {export_dir}")
    rows = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if stages is not None:
        rows = rows[rows["stage"].isin(stages)].copy()
    return rows


def build_wide(rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["rank", "rr", "rank_raw", "rr_raw"]
    wide = rows.pivot_table(index=KEYS, columns="stage", values=value_cols, aggfunc="first")
    wide.columns = [f"true_official_{metric}_{stage}" for metric, stage in wide.columns]
    return wide.reset_index()


def relation_metric_checks(
    rows: pd.DataFrame,
    run_suffix: str,
    aggregation_dir_name: str = "",
) -> pd.DataFrame:
    checks = []
    grouped = rows.groupby(["dataset", "experiment", "relation_id", "stage"], sort=True)
    for (dataset, experiment, relation_id, stage), group in grouped:
        metric_key = {
            "stage1": "test_after_stage1",
            "stage2": "test_after_stage2",
            "final": "test",
        }.get(stage)
        if metric_key is None:
            raise ValueError(f"Unknown exported stage: {stage}")
        result_dir = aggregation_dir_name or f"{experiment}_{run_suffix}"
        metric_path = ROOT / "data" / dataset / "aggregation" / result_dir / f"metric-{int(relation_id)}.json"
        metric_mrr = None
        metric_rows = None
        if metric_path.exists():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            metric_obj = metric.get(metric_key)
            if metric_obj is None and stage == "stage2":
                metric_obj = metric.get("test_after_stage1")
            if metric_obj is not None:
                metric_mrr = float(metric_obj["mrr"])
            metric_rows = int(metric.get("num_test_samples", 0)) * 2
        exported_mrr = float(group["rr"].mean())
        checks.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "relation_id": int(relation_id),
                "stage": stage,
                "exported_rows": int(len(group)),
                "expected_rows": metric_rows,
                "exported_mrr": exported_mrr,
                "metric_mrr": metric_mrr,
                "abs_diff": None if metric_mrr is None else abs(exported_mrr - metric_mrr),
            }
        )
    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--run_suffix", default="true_rr_rerun_20260501")
    parser.add_argument(
        "--aggregation_dir_name",
        default="",
        help="Shared aggregation result directory name, for example reproduction.",
    )
    parser.add_argument(
        "--stages",
        default="",
        help="Optional comma-separated exported stages to merge, for example stage1,final.",
    )
    parser.add_argument("--out_rows", type=Path, default=DEFAULT_EXPORT_DIR / "true_official_per_query_rr_long.csv")
    parser.add_argument("--out_wide", type=Path, default=DEFAULT_EXPORT_DIR / "true_official_per_query_rr_wide.csv")
    parser.add_argument("--out_checks", type=Path, default=DEFAULT_EXPORT_DIR / "true_official_per_query_rr_checks.csv")
    args = parser.parse_args()

    stages = {value.strip() for value in args.stages.split(",") if value.strip()} or None
    rows = load_rows(args.export_dir, stages=stages)
    wide = build_wide(rows)
    checks = relation_metric_checks(rows, args.run_suffix, args.aggregation_dir_name)

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_rows, index=False)
    wide.to_csv(args.out_wide, index=False)
    checks.to_csv(args.out_checks, index=False)

    max_diff = checks["abs_diff"].dropna().max() if not checks.empty else float("nan")
    missing = int(checks["metric_mrr"].isna().sum()) if "metric_mrr" in checks else 0
    print(f"rows_long={len(rows)} -> {args.out_rows}")
    print(f"rows_wide={len(wide)} -> {args.out_wide}")
    print(f"checks={len(checks)} missing_metric={missing} max_abs_diff={max_diff}")


if __name__ == "__main__":
    main()
