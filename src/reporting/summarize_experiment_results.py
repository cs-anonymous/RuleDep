#!/usr/bin/env python3
"""Summarize application and aggregation metrics across datasets/configurations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


DATASETS = [
    "KG20C",
    "codex-m",
    "WN18RR",
    "FB15k-237",
    "codex-l",
    "YAGO3-10",
]

APPLICATION_LOGS = [
    ("app_rules_maxplus", "eval-maxplus.log", "rules + maxplus"),
    ("app_rules_noisyor", "eval-noisyor.log", "rules + noisyor"),
    ("app_base_ranker_maxplus", "eval_base_ranker_maxplus.log", "base ranker + maxplus"),
    ("app_base_ranker_noisyor", "eval_base_ranker_noisyor.log", "base ranker + noisyor"),
]

CONFIG_SPECS = [
    {
        "config_id": "cfg_synergy_redundancy",
        "config_label": "synergy + redundancy | auto_sqrt",
        "match": {
            "synergy": True,
            "redundancy": True,
            "sign_constraint_dependency": False,
            "init_dep_with_lift": False,
            "pos": "auto_sqrt",
        },
    },
    {
        "config_id": "cfg_synergy_only",
        "config_label": "synergy only | auto_sqrt",
        "match": {
            "synergy": True,
            "redundancy": False,
            "sign_constraint_dependency": False,
            "init_dep_with_lift": False,
            "pos": "auto_sqrt",
        },
    },
    {
        "config_id": "cfg_redundancy_only",
        "config_label": "redundancy only | auto_sqrt",
        "match": {
            "synergy": False,
            "redundancy": True,
            "sign_constraint_dependency": False,
            "init_dep_with_lift": False,
            "pos": "auto_sqrt",
        },
    },
    {
        "config_id": "cfg_dep_sign",
        "config_label": "synergy + redundancy + dependency sign | auto_sqrt",
        "match": {
            "synergy": True,
            "redundancy": True,
            "sign_constraint_dependency": True,
            "init_dep_with_lift": False,
            "pos": "auto_sqrt",
        },
    },
    {
        "config_id": "cfg_lift_init",
        "config_label": "synergy + redundancy + lift init | auto_sqrt",
        "match": {
            "synergy": True,
            "redundancy": True,
            "sign_constraint_dependency": False,
            "init_dep_with_lift": True,
            "pos": "auto_sqrt",
        },
    },
    {
        "config_id": "cfg_auto_ratio",
        "config_label": "synergy + redundancy | auto_ratio",
        "match": {
            "synergy": True,
            "redundancy": True,
            "sign_constraint_dependency": False,
            "init_dep_with_lift": False,
            "pos": "auto_ratio",
        },
    },
]

METRIC_RE = re.compile(
    r"MRR\s+([0-9]*\.?[0-9]+).*?"
    r"hits@1\s+([0-9]*\.?[0-9]+).*?"
    r"hits@10\s+([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root directory that contains dataset folders.")
    parser.add_argument(
        "--output-dir",
        default="reports/experiment_summary",
        help="Directory where summary tables and report will be written.",
    )
    return parser.parse_args()


def matches_config(config: Dict[str, object], target: Dict[str, object]) -> bool:
    return all(config.get(key) == value for key, value in target.items())


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_relation_ids(dataset_dir: Path) -> Optional[int]:
    relation_path = dataset_dir / "relation_ids.del"
    if not relation_path.exists():
        return None
    with relation_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def parse_eval_log(path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "status": "missing",
        "mrr": None,
        "h1": None,
        "h10": None,
        "path": str(path),
    }
    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="ignore")
    match = METRIC_RE.search(text)
    if not match:
        result["status"] = "incomplete"
        return result

    result["status"] = "done"
    result["mrr"] = float(match.group(1))
    result["h1"] = float(match.group(2))
    result["h10"] = float(match.group(3))
    return result


def metric_triplet(metrics: Optional[Dict[str, object]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not metrics:
        return None, None, None
    return (
        to_float(metrics.get("mrr")),
        to_float(metrics.get("h1")),
        to_float(metrics.get("h10")),
    )


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def fmt_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def fmt_delta(value: Optional[float]) -> str:
    return "" if value is None else f"{value:+.6f}"


def escape_md(text: object) -> str:
    return str(text).replace("|", "\\|")


def load_matching_experiments(dataset_dir: Path) -> Dict[str, Dict[str, object]]:
    agg_dir = dataset_dir / "aggregation"
    matched: Dict[str, Dict[str, object]] = {}
    if not agg_dir.exists():
        return matched

    for exp_dir in sorted(agg_dir.glob("exp*")):
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        config = read_json(config_path)
        for spec in CONFIG_SPECS:
            if not matches_config(config, spec["match"]):
                continue
            current = matched.get(spec["config_id"])
            candidate = {
                "exp_dir": exp_dir,
                "config": config,
                "has_metrics_final": (exp_dir / "metrics-final.json").exists(),
            }
            if current is None:
                matched[spec["config_id"]] = candidate
            elif (not current["has_metrics_final"]) and candidate["has_metrics_final"]:
                matched[spec["config_id"]] = candidate
    return matched


def summarize_aggregation_entry(
    dataset_dir: Path,
    total_relations: Optional[int],
    spec: Dict[str, object],
    match_entry: Optional[Dict[str, object]],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "config_id": spec["config_id"],
        "config_label": spec["config_label"],
        "config_status": "not_started",
        "experiment_dir": "",
        "completed_relations": 0,
        "total_relations": total_relations,
        "failed_relations": None,
        "stage1_mrr": None,
        "stage1_h1": None,
        "stage1_h10": None,
        "stage2_mrr": None,
        "stage2_h1": None,
        "stage2_h10": None,
    }

    if match_entry is None:
        return row

    exp_dir = match_entry["exp_dir"]
    row["experiment_dir"] = str(exp_dir)
    metric_files = sorted(exp_dir.glob("metric-*.json"))
    row["completed_relations"] = len(metric_files)

    metrics_final = exp_dir / "metrics-final.json"
    if not metrics_final.exists():
        row["config_status"] = "partial" if metric_files else "started"
        return row

    payload = read_json(metrics_final)
    summary = payload.get("summary") or {}
    failed_relations = payload.get("failed_relations") or {}
    stage1 = summary.get("test_after_stage1")
    stage2 = summary.get("test_after_stage2") or summary.get("test")

    row["config_status"] = "done"
    row["failed_relations"] = len(failed_relations)
    row["completed_relations"] = row["total_relations"] or len(metric_files)
    row["stage1_mrr"], row["stage1_h1"], row["stage1_h10"] = metric_triplet(stage1)
    row["stage2_mrr"], row["stage2_h1"], row["stage2_h10"] = metric_triplet(stage2)
    return row


def add_delta_metrics(row: Dict[str, object]) -> None:
    for metric in ("mrr", "h1", "h10"):
        stage1_value = row.get(f"stage1_{metric}")
        stage2_value = row.get(f"stage2_{metric}")
        row[f"delta_stage2_minus_stage1_{metric}"] = (
            None if stage1_value is None or stage2_value is None else stage2_value - stage1_value
        )


def build_rows(data_root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for dataset in DATASETS:
        dataset_dir = data_root / dataset
        application_dir = dataset_dir / "application"
        total_relations = count_relation_ids(dataset_dir)
        app_metrics = {
            app_id: parse_eval_log(application_dir / log_name)
            for app_id, log_name, _ in APPLICATION_LOGS
        }
        matched_experiments = load_matching_experiments(dataset_dir)

        for app_id, _, app_label in APPLICATION_LOGS:
            metric = app_metrics[app_id]
            long_rows.append(
                {
                    "dataset": dataset,
                    "source_group": "application",
                    "config_id": "",
                    "config_label": "",
                    "stage_or_source": app_id,
                    "label": app_label,
                    "status": metric["status"],
                    "mrr": metric["mrr"],
                    "h1": metric["h1"],
                    "h10": metric["h10"],
                    "completed_relations": "",
                    "total_relations": "",
                    "path": metric["path"],
                }
            )

        for spec in CONFIG_SPECS:
            agg_entry = summarize_aggregation_entry(dataset_dir, total_relations, spec, matched_experiments.get(spec["config_id"]))
            wide_row: Dict[str, object] = {
                "dataset": dataset,
                "config_id": agg_entry["config_id"],
                "config_label": agg_entry["config_label"],
                "config_status": agg_entry["config_status"],
                "completed_relations": agg_entry["completed_relations"],
                "total_relations": agg_entry["total_relations"],
                "failed_relations": agg_entry["failed_relations"],
                "experiment_dir": agg_entry["experiment_dir"],
            }
            for metric_name in (
                "stage1_mrr",
                "stage1_h1",
                "stage1_h10",
                "stage2_mrr",
                "stage2_h1",
                "stage2_h10",
            ):
                wide_row[metric_name] = agg_entry[metric_name]

            add_delta_metrics(wide_row)

            for app_id, _, _ in APPLICATION_LOGS:
                metric = app_metrics[app_id]
                wide_row[f"{app_id}_status"] = metric["status"]
                wide_row[f"{app_id}_mrr"] = metric["mrr"]
                wide_row[f"{app_id}_h1"] = metric["h1"]
                wide_row[f"{app_id}_h10"] = metric["h10"]

            wide_rows.append(wide_row)

            for stage_name, stage_label in (
                ("stage1", "aggregation stage1"),
                ("stage2", "aggregation stage2"),
            ):
                long_rows.append(
                    {
                        "dataset": dataset,
                        "source_group": "aggregation",
                        "config_id": agg_entry["config_id"],
                        "config_label": agg_entry["config_label"],
                        "stage_or_source": stage_name,
                        "label": stage_label,
                        "status": agg_entry["config_status"],
                        "mrr": agg_entry.get(f"{stage_name}_mrr"),
                        "h1": agg_entry.get(f"{stage_name}_h1"),
                        "h10": agg_entry.get(f"{stage_name}_h10"),
                        "completed_relations": agg_entry["completed_relations"],
                        "total_relations": agg_entry["total_relations"],
                        "path": agg_entry["experiment_dir"],
                    }
                )

    return wide_rows, long_rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: fmt_float(value)
                    if key.endswith(("_mrr", "_h1", "_h10")) and isinstance(value, float)
                    else fmt_float(value)
                    if key in {"mrr", "h1", "h10"} and isinstance(value, float)
                    else fmt_delta(value)
                    if key.startswith("delta_") and isinstance(value, float)
                    else value
                    for key, value in row.items()
                }
            )


def best_completed(rows: Iterable[Dict[str, object]], metric_key: str) -> Optional[Dict[str, object]]:
    candidates = [row for row in rows if row.get("config_status") == "done" and row.get(metric_key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric_key])


def best_application_metric(row: Dict[str, object]) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    best_name = None
    best_scores: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None)
    best_mrr = None
    for app_id, _, app_label in APPLICATION_LOGS:
        if row.get(f"{app_id}_status") != "done":
            continue
        mrr = row.get(f"{app_id}_mrr")
        if mrr is None:
            continue
        if best_mrr is None or mrr > best_mrr:
            best_name = app_label
            best_mrr = mrr
            best_scores = (
                row.get(f"{app_id}_h1"),
                row.get(f"{app_id}_h10"),
                mrr,
            )
    if best_name is None:
        return None, None, None, None
    return best_name, best_scores[2], best_scores[0], best_scores[1]


def build_report(wide_rows: List[Dict[str, object]], output_dir: Path) -> None:
    dataset_rows: Dict[str, List[Dict[str, object]]] = {dataset: [] for dataset in DATASETS}
    for row in wide_rows:
        dataset_rows[row["dataset"]].append(row)

    lines: List[str] = []
    lines.append("# Experiment Summary Report")
    lines.append("")
    lines.append(f"- Datasets covered: {', '.join(DATASETS)}")
    lines.append(f"- Canonical aggregation configs: {len(CONFIG_SPECS)}")
    lines.append("")

    lines.append("## Best Completed Aggregation Config Per Dataset")
    lines.append("")
    lines.append("| Dataset | Best config | Stage2 MRR | H@1 | H@10 | Best application | App MRR | Delta vs best app |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
    for dataset in DATASETS:
        best_row = best_completed(dataset_rows[dataset], "stage2_mrr")
        app_name, app_mrr, _, _ = best_application_metric(dataset_rows[dataset][0])
        if best_row is None:
            lines.append(f"| {dataset} | - | - | - | - | {app_name or '-'} | {fmt_float(app_mrr)} | - |")
            continue
        delta_vs_app = None if app_mrr is None else best_row["stage2_mrr"] - app_mrr
        lines.append(
            "| {dataset} | {config} | {mrr} | {h1} | {h10} | {app_name} | {app_mrr} | {delta} |".format(
                dataset=dataset,
                config=escape_md(best_row["config_label"]),
                mrr=fmt_float(best_row["stage2_mrr"]),
                h1=fmt_float(best_row["stage2_h1"]),
                h10=fmt_float(best_row["stage2_h10"]),
                app_name=escape_md(app_name or "-"),
                app_mrr=fmt_float(app_mrr),
                delta=fmt_delta(delta_vs_app),
            )
        )
    lines.append("")

    lines.append("## Average Stage2 Minus Stage1 Gain By Config")
    lines.append("")
    lines.append("| Config | Completed datasets | Avg delta MRR | Avg delta H@1 | Avg delta H@10 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for spec in CONFIG_SPECS:
        rows = [
            row
            for row in wide_rows
            if row["config_id"] == spec["config_id"] and row["config_status"] == "done"
        ]
        delta_mrr = [row["delta_stage2_minus_stage1_mrr"] for row in rows if row["delta_stage2_minus_stage1_mrr"] is not None]
        delta_h1 = [row["delta_stage2_minus_stage1_h1"] for row in rows if row["delta_stage2_minus_stage1_h1"] is not None]
        delta_h10 = [row["delta_stage2_minus_stage1_h10"] for row in rows if row["delta_stage2_minus_stage1_h10"] is not None]
        lines.append(
            "| {config} | {count} | {mrr} | {h1} | {h10} |".format(
                config=escape_md(spec["config_label"]),
                count=len(rows),
                mrr=fmt_delta(mean(delta_mrr)) if delta_mrr else "-",
                h1=fmt_delta(mean(delta_h1)) if delta_h1 else "-",
                h10=fmt_delta(mean(delta_h10)) if delta_h10 else "-",
            )
        )
    lines.append("")

    status_counter = Counter(row["config_status"] for row in wide_rows)
    lines.append("## Aggregation Status Overview")
    lines.append("")
    for status, count in sorted(status_counter.items()):
        lines.append(f"- {status}: {count}")
    lines.append("")

    pending_rows = [row for row in wide_rows if row["config_status"] != "done"]
    if pending_rows:
        lines.append("## Incomplete Or Missing Aggregation Runs")
        lines.append("")
        lines.append("| Dataset | Config | Status | Progress | Experiment dir |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in pending_rows:
            progress = (
                "-"
                if not row["total_relations"]
                else f"{row['completed_relations']}/{row['total_relations']}"
            )
            lines.append(
                f"| {row['dataset']} | {escape_md(row['config_label'])} | {row['config_status']} | {progress} | {escape_md(row['experiment_dir'] or '-')} |"
            )
        lines.append("")

    report_path = output_dir / "experiment_summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_rows, long_rows = build_rows(data_root)

    wide_fieldnames = [
        "dataset",
        "config_id",
        "config_label",
        "config_status",
        "completed_relations",
        "total_relations",
        "failed_relations",
        "experiment_dir",
        "stage1_mrr",
        "stage1_h1",
        "stage1_h10",
        "stage2_mrr",
        "stage2_h1",
        "stage2_h10",
        "delta_stage2_minus_stage1_mrr",
        "delta_stage2_minus_stage1_h1",
        "delta_stage2_minus_stage1_h10",
    ]
    for app_id, _, _ in APPLICATION_LOGS:
        wide_fieldnames.extend(
            [
                f"{app_id}_status",
                f"{app_id}_mrr",
                f"{app_id}_h1",
                f"{app_id}_h10",
            ]
        )

    long_fieldnames = [
        "dataset",
        "source_group",
        "config_id",
        "config_label",
        "stage_or_source",
        "label",
        "status",
        "mrr",
        "h1",
        "h10",
        "completed_relations",
        "total_relations",
        "path",
    ]

    write_csv(output_dir / "experiment_summary_wide.csv", wide_rows, wide_fieldnames)
    write_csv(output_dir / "experiment_summary_long.csv", long_rows, long_fieldnames)
    build_report(wide_rows, output_dir)

    print(f"Wrote {output_dir / 'experiment_summary_wide.csv'}")
    print(f"Wrote {output_dir / 'experiment_summary_long.csv'}")
    print(f"Wrote {output_dir / 'experiment_summary_report.md'}")


if __name__ == "__main__":
    main()
