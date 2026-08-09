#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


PREFERRED_DATASET_ORDER = [
    "KG20C",
    "codex-m",
    "WN18RR",
    "FB15k-237",
    "codex-l",
    "YAGO3-10",
    "hetionet",
    "wikidata5m",
]

EVAL_SOURCES = [
    ("eval-maxplus", "eval-maxplus.log"),
    ("eval-noisyor", "eval-noisyor.log"),
]

STAGE1_DUPLICATE_EXPERIMENTS = {
    "structural_none",
    "structural_rd",
    "structural_r2d3",
    "structural_r3d6",
}

MAIN_STRUCTURAL_EXPERIMENTS = [
    "structural_none",
    "structural_rd",
    "structural_r2d3",
    "structural_r3d6",
]

ESTIMATED_CANONICAL_TIMES = {
    # Estimated from the interrupted codex-l canonical run:
    # 2026-04-05 14:33:58 -> 20:25:16 reached epoch 26/40, 2/69 relations.
    "codex-l": 33685.558747,
}

EXCLUDED_AGGREGATION_PREFIXES = ["best_combination"]

METRIC_RE = re.compile(
    r"MRR\s+([0-9]*\.?[0-9]+).*?"
    r"hits@1\s+([0-9]*\.?[0-9]+).*?"
    r"hits@10\s+([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)
RUNTIME_RE = re.compile(r"Total runtime:\s*([0-9]+):([0-9]{2}):([0-9]{2}(?:\.[0-9]+)?)")
TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3,6})")
TUPLE_RE = re.compile(r"\(([0-9eE+\-.]+),\s*([0-9eE+\-.]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 0407 summary, relation, and dataset analysis reports.")
    parser.add_argument("--root", default=None, help="Repository root. Defaults to the script parent directory.")
    parser.add_argument("--data-root", default=None, help="Data root. Defaults to <root>/data.")
    parser.add_argument("--report-dir", default=None, help="Report output directory. Defaults to <root>/reports/0407.")
    parser.add_argument(
        "--forced-best-config",
        default="",
        help="Force this aggregation name as best_config when that run exists for a dataset.",
    )
    parser.add_argument(
        "--forced-best-config-prefix",
        default="",
        help="If set, pick the best config only among aggregations with this prefix.",
    )
    parser.add_argument(
        "--safe-ensemble-margin",
        type=float,
        default=0.0,
        help="If >0, enable ensemble_safe_valid: use top-valid relation model only when (valid_top1 - valid_top2) >= margin; otherwise fallback to best single config.",
    )
    return parser.parse_args()


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.6f}"


def fmt_float_short(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.5f}"


def fmt_int(value: Optional[int]) -> str:
    return "" if value is None else str(int(value))


def image_block(src: str, alt: str, caption: str, width_pct: int = 60) -> List[str]:
    return [
        f'<p align="center"><img src="{src}" alt="{alt}" width="{width_pct}%"></p>',
        "",
        f"<p align=\"center\"><em>{caption}</em></p>",
        "",
    ]


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(median(vals))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_datasets(data_root: Path) -> List[str]:
    discovered = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        has_split = (child / "train.txt").exists() and (child / "valid.txt").exists() and (child / "test.txt").exists()
        has_result = (child / "application").exists() or (child / "aggregation").exists() or (child / "rules" / "rule.txt").exists()
        if has_split and has_result:
            discovered.append(child.name)

    ordered = [name for name in PREFERRED_DATASET_ORDER if name in discovered]
    extras = sorted(name for name in discovered if name not in ordered)
    return ordered + extras


def parse_eval_log(path: Path) -> Dict[str, Optional[float]]:
    result = {"MRR": None, "h@1": None, "h@10": None, "time": None}
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="ignore")
    metric_match = METRIC_RE.search(text)
    if metric_match:
        result["MRR"] = float(metric_match.group(1))
        result["h@1"] = float(metric_match.group(2))
        result["h@10"] = float(metric_match.group(3))
    runtime_match = RUNTIME_RE.search(text)
    if runtime_match:
        hours = int(runtime_match.group(1))
        minutes = int(runtime_match.group(2))
        seconds = float(runtime_match.group(3))
        result["time"] = hours * 3600 + minutes * 60 + seconds
    return result


def parse_canonical_log(exp_dir: Path) -> Dict[str, Optional[float]]:
    result = {"MRR": None, "h@1": None, "h@10": None, "time": None, "status": "missing"}
    log_path = exp_dir / "canonical.log"
    if not log_path.exists():
        return result

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    has_head = any(exp_dir.glob("head_mrr_*.p"))
    has_tail = any(exp_dir.glob("tail_mrr_*.p"))
    tuples = TUPLE_RE.findall(text)
    timestamps = TIMESTAMP_RE.findall(text)
    done = "Done" in text and has_head and has_tail

    if timestamps:
        try:
            start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S,%f")
            end = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S,%f")
            result["time"] = max(0.0, (end - start).total_seconds())
        except ValueError:
            result["time"] = None

    if not done:
        result["status"] = "partial"
        return result

    if len(tuples) >= 3:
        result["MRR"] = float(tuples[-3][0])
        result["h@1"] = float(tuples[-2][0])
        result["h@10"] = float(tuples[-1][0])
        result["status"] = "done"
    return result


def metric_triplet(node: Optional[Dict[str, object]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not node:
        return None, None, None
    return to_float(node.get("mrr")), to_float(node.get("h1")), to_float(node.get("h10"))


def selected_valid_mrr(metric: Dict[str, object]) -> Optional[float]:
    selection = metric.get("model_selection") or {}
    selected_stage = str(selection.get("selected_stage") or "")
    if selected_stage == "dependency":
        return to_float(((metric.get("best_valid_stage2") or {}).get("mrr")))
    return to_float(((metric.get("best_valid_stage1") or {}).get("mrr")))


def selected_test_metrics(metric: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    final_test = metric.get("test") or metric.get("test_after_stage2")
    return metric_triplet(final_test)


def list_aggregation_dirs(agg_root: Path) -> List[Path]:
    if not agg_root.exists():
        return []
    return sorted(path for path in agg_root.iterdir() if path.is_dir() and path.name != "canonical")


def is_excluded_aggregation(name: str) -> bool:
    return any(str(name).startswith(prefix) for prefix in EXCLUDED_AGGREGATION_PREFIXES)


def build_overall_rows(data_root: Path, datasets: List[str], safe_ensemble_margin: float = 0.0) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ensemble_debug: Dict[str, object] = {}

    for dataset in datasets:
        ds_dir = data_root / dataset
        app_dir = ds_dir / "application"
        agg_dir = ds_dir / "aggregation"

        for aggregation, log_name in EVAL_SOURCES:
            metrics = parse_eval_log(app_dir / log_name)
            if metrics["MRR"] is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "aggregation": aggregation,
                    "MRR": metrics["MRR"],
                    "h@1": metrics["h@1"],
                    "h@10": metrics["h@10"],
                    "time": metrics["time"],
                }
            )

        for exp_dir in list_aggregation_dirs(agg_dir):
            if is_excluded_aggregation(exp_dir.name):
                continue
            metrics_path = exp_dir / "metrics-final.json"
            if not metrics_path.exists():
                continue
            payload = read_json(metrics_path)
            summary = payload.get("summary") or {}
            total_time = to_float(((payload.get("time_seconds") or {}).get("total")))
            stage2 = summary.get("test") or summary.get("test_after_stage2")
            stage1 = summary.get("test_after_stage1")
            stage2_mrr, stage2_h1, stage2_h10 = metric_triplet(stage2)
            if stage2_mrr is not None:
                rows.append(
                    {
                        "dataset": dataset,
                        "aggregation": exp_dir.name,
                        "MRR": stage2_mrr,
                        "h@1": stage2_h1,
                        "h@10": stage2_h10,
                        "time": total_time,
                    }
                )
            if exp_dir.name in STAGE1_DUPLICATE_EXPERIMENTS:
                stage1_mrr, stage1_h1, stage1_h10 = metric_triplet(stage1)
                if stage1_mrr is not None:
                    rows.append(
                        {
                            "dataset": dataset,
                            "aggregation": f"{exp_dir.name}__stage1",
                            "MRR": stage1_mrr,
                            "h@1": stage1_h1,
                            "h@10": stage1_h10,
                            "time": total_time,
                        }
                    )

        canonical = parse_canonical_log(agg_dir / "canonical")
        if canonical["status"] == "done":
            rows.append(
                {
                    "dataset": dataset,
                    "aggregation": "canonical",
                    "MRR": canonical["MRR"],
                    "h@1": canonical["h@1"],
                    "h@10": canonical["h@10"],
                    "time": canonical["time"],
                }
            )

        ensemble_rows, debug_payload = build_ensemble_rows(dataset, agg_dir, safe_ensemble_margin=safe_ensemble_margin)
        ensemble_debug[dataset] = debug_payload
        rows.extend(ensemble_rows)

    rows.sort(key=lambda row: (PREFERRED_DATASET_ORDER.index(row["dataset"]) if row["dataset"] in PREFERRED_DATASET_ORDER else 999, row["dataset"], row["aggregation"]))
    return rows, ensemble_debug


def build_ensemble_rows(dataset: str, agg_dir: Path, safe_ensemble_margin: float = 0.0) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    best_valid_by_relation: Dict[int, Dict[str, object]] = {}
    best_test_by_relation: Dict[int, Dict[str, object]] = {}
    candidates_by_relation: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    debug: Dict[str, object] = {
        "dataset": dataset,
        "selected_relations_valid": {},
        "selected_relations_test": {},
        "selected_relations_safe_valid": {},
        "safe_ensemble_margin": safe_ensemble_margin,
        "safe_ensemble_fallback_config": "",
    }

    # used to identify fallback best single config over remaining candidates
    per_exp_weighted_valid_sum: Dict[str, float] = defaultdict(float)
    per_exp_weighted_cnt: Dict[str, int] = defaultdict(int)

    for exp_dir in list_aggregation_dirs(agg_dir):
        if is_excluded_aggregation(exp_dir.name):
            continue
        for metric_path in sorted(exp_dir.glob("metric-*.json")):
            metric = read_json(metric_path)
            relation = int(metric["relation"])
            valid_mrr = selected_valid_mrr(metric)
            test_mrr, test_h1, test_h10 = selected_test_metrics(metric)
            count = int(metric.get("num_test_samples", 0))
            time_total = to_float(((metric.get("time_seconds") or {}).get("total"))) or 0.0
            if valid_mrr is None or test_mrr is None or count <= 0:
                continue
            current_valid = best_valid_by_relation.get(relation)
            if current_valid is None or valid_mrr > float(current_valid["valid_mrr"]):
                best_valid_by_relation[relation] = {
                    "experiment": exp_dir.name,
                    "valid_mrr": valid_mrr,
                    "MRR": test_mrr,
                    "h@1": test_h1,
                    "h@10": test_h10,
                    "count": count,
                    "time": time_total,
                    "selected_stage": str((metric.get("model_selection") or {}).get("selected_stage") or ""),
                }

            current_test = best_test_by_relation.get(relation)
            if current_test is None or test_mrr > float(current_test["MRR"]):
                best_test_by_relation[relation] = {
                    "experiment": exp_dir.name,
                    "valid_mrr": valid_mrr,
                    "MRR": test_mrr,
                    "h@1": test_h1,
                    "h@10": test_h10,
                    "count": count,
                    "time": time_total,
                    "selected_stage": str((metric.get("model_selection") or {}).get("selected_stage") or ""),
                }

            row_obj = {
                "experiment": exp_dir.name,
                "valid_mrr": valid_mrr,
                "MRR": test_mrr,
                "h@1": test_h1,
                "h@10": test_h10,
                "count": count,
                "time": time_total,
                "selected_stage": str((metric.get("model_selection") or {}).get("selected_stage") or ""),
            }
            candidates_by_relation[relation].append(row_obj)

            per_exp_weighted_valid_sum[exp_dir.name] += float(valid_mrr) * count
            per_exp_weighted_cnt[exp_dir.name] += count

    def summarize_ensemble(best_by_relation: Dict[int, Dict[str, object]], aggregation_name: str, debug_key: str) -> Optional[Dict[str, object]]:
        if not best_by_relation:
            return None

        total_weight = sum(int(row["count"]) for row in best_by_relation.values())
        if total_weight <= 0:
            return None

        def weighted(metric_key: str) -> float:
            return sum(float(row[metric_key]) * int(row["count"]) for row in best_by_relation.values()) / total_weight

        total_time = sum(float(row["time"]) for row in best_by_relation.values())
        for relation, row in sorted(best_by_relation.items()):
            debug[debug_key][str(relation)] = {
                "experiment": row["experiment"],
                "selected_stage": row["selected_stage"],
                "selected_valid_mrr": row["valid_mrr"],
                "test_mrr": row["MRR"],
                "count": row["count"],
            }
        return {
            "dataset": dataset,
            "aggregation": aggregation_name,
            "MRR": weighted("MRR"),
            "h@1": weighted("h@1"),
            "h@10": weighted("h@10"),
            "time": total_time,
        }

    ensemble_rows: List[Dict[str, object]] = []
    valid_row = summarize_ensemble(best_valid_by_relation, "ensemble_best_valid", "selected_relations_valid")
    test_row = summarize_ensemble(best_test_by_relation, "ensemble_best_test", "selected_relations_test")

    safe_row: Optional[Dict[str, object]] = None
    if safe_ensemble_margin > 0.0 and per_exp_weighted_cnt:
        # fallback to best single remaining config (weighted valid MRR across relations)
        fallback_exp = max(
            per_exp_weighted_cnt.keys(),
            key=lambda name: (per_exp_weighted_valid_sum[name] / per_exp_weighted_cnt[name]) if per_exp_weighted_cnt[name] > 0 else -1.0,
        )
        debug["safe_ensemble_fallback_config"] = fallback_exp

        safe_by_relation: Dict[int, Dict[str, object]] = {}
        for relation, arr in candidates_by_relation.items():
            if not arr:
                continue
            sorted_by_valid = sorted(arr, key=lambda row: float(row["valid_mrr"]), reverse=True)
            top1 = sorted_by_valid[0]
            top2_valid = float(sorted_by_valid[1]["valid_mrr"]) if len(sorted_by_valid) >= 2 else float(top1["valid_mrr"])
            margin = float(top1["valid_mrr"]) - top2_valid

            fallback = next((row for row in arr if str(row["experiment"]) == fallback_exp), top1)
            picked = top1 if margin >= safe_ensemble_margin else fallback

            safe_by_relation[relation] = picked
            debug["selected_relations_safe_valid"][str(relation)] = {
                "experiment": picked["experiment"],
                "selected_stage": picked["selected_stage"],
                "selected_valid_mrr": picked["valid_mrr"],
                "test_mrr": picked["MRR"],
                "count": picked["count"],
                "margin": margin,
                "used_fallback": bool(margin < safe_ensemble_margin),
            }

        safe_row = summarize_ensemble(safe_by_relation, "ensemble_safe_valid", "selected_relations_safe_valid")

    if valid_row is not None:
        ensemble_rows.append(valid_row)
    if test_row is not None:
        ensemble_rows.append(test_row)
    if safe_row is not None:
        ensemble_rows.append(safe_row)
    return ensemble_rows, debug


def overall_rows_to_csv_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "dataset": row["dataset"],
                "aggregation": row["aggregation"],
                "MRR": fmt_float(row["MRR"]),
                "h@1": fmt_float(row["h@1"]),
                "h@10": fmt_float(row["h@10"]),
                "time": fmt_float(row["time"]),
            }
        )
    return csv_rows


def _pick_best_row(
    candidates: List[Dict[str, object]],
    mapping: Dict[str, Dict[str, object]],
    forced_best_config: str = "",
    forced_best_config_prefix: str = "",
) -> Optional[Dict[str, object]]:
    best_row = max(candidates, key=lambda row: float(row["MRR"])) if candidates else None
    if forced_best_config:
        forced = mapping.get(forced_best_config)
        if forced is not None:
            return forced
    if forced_best_config_prefix:
        prefix_candidates = [
            row for row in candidates if str(row.get("aggregation", "")).startswith(forced_best_config_prefix)
        ]
        if prefix_candidates:
            return max(prefix_candidates, key=lambda row: float(row["MRR"]))
    return best_row


def build_best_config_rows(
    rows: List[Dict[str, object]],
    datasets: List[str],
    forced_best_config: str = "",
    forced_best_config_prefix: str = "",
) -> List[Dict[str, object]]:
    by_dataset: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["aggregation"]] = row

    out: List[Dict[str, object]] = []
    for dataset in datasets:
        mapping = by_dataset.get(dataset, {})
        candidates = [
            row
            for name, row in mapping.items()
            if name not in {"eval-maxplus", "eval-noisyor", "canonical", "ensemble_best_valid", "ensemble_best_test", "ensemble_safe_valid"}
            and not is_excluded_aggregation(name)
            and not name.endswith("__stage1")
        ]
        best_row = _pick_best_row(
            candidates,
            mapping,
            forced_best_config=forced_best_config,
            forced_best_config_prefix=forced_best_config_prefix,
        )
        row = {
            "dataset": dataset,
            "best_config": best_row["aggregation"] if best_row else "",
            "best_config_mrr": fmt_float(best_row["MRR"]) if best_row else "",
            "structural_none": fmt_float((mapping.get("structural_none") or {}).get("MRR")),
            "structural_rd": fmt_float((mapping.get("structural_rd") or {}).get("MRR")),
            "structural_r2d3": fmt_float((mapping.get("structural_r2d3") or {}).get("MRR")),
            "structural_r3d6": fmt_float((mapping.get("structural_r3d6") or {}).get("MRR")),
            "canonical": fmt_float((mapping.get("canonical") or {}).get("MRR")),
            "ensemble_best_valid": fmt_float((mapping.get("ensemble_best_valid") or {}).get("MRR")),
            "ensemble_best_test": fmt_float((mapping.get("ensemble_best_test") or {}).get("MRR")),
            "ensemble_safe_valid": fmt_float((mapping.get("ensemble_safe_valid") or {}).get("MRR")),
            "eval-maxplus": fmt_float((mapping.get("eval-maxplus") or {}).get("MRR")),
            "eval-noisyor": fmt_float((mapping.get("eval-noisyor") or {}).get("MRR")),
        }
        out.append(row)
    return out


def estimate_ruledep_stage_times(data_root: Path, best_config_rows: List[Dict[str, object]], overall_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Estimate relation-wise wall-clock time from per-relation epoch counts.

    Existing metrics only persist combined per-relation time.  We split each
    relation's time by stage1/stage2 epoch ratio, then divide the relation-wise
    estimate by two because the current main runs use multiprocess=2.  Canonical
    is an old serial runner, so its parsed wall-clock time is not divided.
    """

    canonical_time_by_dataset = {
        str(row["dataset"]): to_float(row.get("time"))
        for row in overall_rows
        if row.get("aggregation") == "canonical"
    }

    out: List[Dict[str, object]] = []
    for row in best_config_rows:
        dataset = str(row["dataset"])
        best_config = str(row.get("best_config") or "")
        if not best_config:
            continue

        exp_dir = data_root / dataset / "aggregation" / best_config
        if not exp_dir.exists():
            continue

        stage1_time = 0.0
        stage2_time = 0.0
        total_time = 0.0
        relation_count = 0
        missing_epoch_count = 0
        for metric_path in sorted(exp_dir.glob("metric-*.json")):
            metric = read_json(metric_path)
            relation_total = to_float((metric.get("time_seconds") or {}).get("total")) or 0.0
            train_info = metric.get("train") or {}
            stage1_info = train_info.get("stage1_rule_only") or {}
            stage2_info = train_info.get("stage2_dependency_only") or {}
            stage1_epochs = to_float(stage1_info.get("epochs_trained")) or 0.0
            stage2_epochs = to_float(stage2_info.get("epochs_trained")) or 0.0
            epoch_total = stage1_epochs + stage2_epochs
            if epoch_total <= 0:
                missing_epoch_count += 1
                stage1_epochs = 1.0
                stage2_epochs = 0.0
                epoch_total = 1.0

            stage1_time += relation_total * stage1_epochs / epoch_total
            stage2_time += relation_total * stage2_epochs / epoch_total
            total_time += relation_total
            relation_count += 1

        out.append(
            {
                "dataset": dataset,
                "best_config": best_config,
                "canonical_time_s": canonical_time_by_dataset.get(dataset, ESTIMATED_CANONICAL_TIMES.get(dataset)),
                "canonical_time_source": "actual" if canonical_time_by_dataset.get(dataset) is not None else ("estimated" if dataset in ESTIMATED_CANONICAL_TIMES else ""),
                "ruledep_stage1_time_s": stage1_time / 2.0,
                "ruledep_stage2_time_s": stage2_time / 2.0,
                "ruledep_total_time_s": total_time / 2.0,
                "relation_count": relation_count,
                "missing_epoch_count": missing_epoch_count,
            }
        )
    return out


def overall_best_config_map(
    rows: List[Dict[str, object]],
    forced_best_config: str = "",
    forced_best_config_prefix: str = "",
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        name = row["aggregation"]
        if name in {"eval-maxplus", "eval-noisyor", "canonical", "ensemble_best_valid", "ensemble_best_test", "ensemble_safe_valid"} or name.endswith("__stage1") or is_excluded_aggregation(name):
            continue
        grouped[row["dataset"]].append(row)
    for dataset, subset in grouped.items():
        if not subset:
            continue
        per_dataset_map = {str(row["aggregation"]): row for row in subset}
        best_row = _pick_best_row(
            subset,
            per_dataset_map,
            forced_best_config=forced_best_config,
            forced_best_config_prefix=forced_best_config_prefix,
        )
        if best_row is not None:
            mapping[dataset] = str(best_row["aggregation"])
    return mapping


def count_dict_file(path: Path) -> int:
    rows = [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if path.name.endswith(".txt") and rows and rows[0].strip().isdigit() and "\t" not in rows[0] and " " not in rows[0]:
        return int(rows[0].strip())
    return len(rows)


def count_entities_relations(ds_dir: Path) -> Tuple[int, int]:
    ent = reln = 0
    for path in [ds_dir / "entity_ids.del", ds_dir / "entities.dict", ds_dir / "entity2id.txt"]:
        if path.exists():
            ent = count_dict_file(path)
            break
    for path in [ds_dir / "relation_ids.del", ds_dir / "relations.dict", ds_dir / "relation2id.txt"]:
        if path.exists():
            reln = count_dict_file(path)
            break
    return ent, reln


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def parse_rule_line(line: str) -> Optional[Dict[str, object]]:
    parts = line.split("\t", 3)
    if len(parts) < 4:
        return None
    try:
        body_size = int(parts[0])
        support = int(parts[1])
        score = float(parts[2])
    except ValueError:
        return None
    return {"body_size": body_size, "support": support, "score": score, "rule": parts[3]}


def is_d_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    return len(parts) == 2 and (parts[1].count("(A,") + parts[1].count(",A)") == 1)


def is_z_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    return len(parts) == 2 and parts[1].strip() == ""


def is_b_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    return len(parts) == 2 and "(X,Y)" in parts[0].strip()


def rule_type(rule_str: str) -> str:
    if is_z_rule(rule_str):
        return "Z"
    if is_b_rule(rule_str):
        return "B"
    if is_d_rule(rule_str):
        return "Ud"
    return "Uc"


def coarse_rule_type(rule_t: str) -> str:
    return "B" if rule_t == "B" else "U"


def load_relation_map(ds_dir: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for candidate in ["relation_ids.del", "relations.dict", "relation2id.txt"]:
        path = ds_dir / candidate
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    lhs, rhs = line.split("\t", 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    lhs, rhs = parts[0], parts[1]
                if lhs.isdigit():
                    mapping[int(lhs)] = rhs
        if mapping:
            return mapping
    return mapping


def count_split_by_relation(ds_dir: Path, split_name: str) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    path = ds_dir / f"{split_name}.del"
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                relation = int(parts[1])
            except ValueError:
                continue
            counts[relation] += 1
    return counts


def parse_rules(ds_dir: Path, relation_map: Dict[int, str]) -> Tuple[Counter, Dict[int, Counter], Dict[int, Tuple[Optional[int], str]]]:
    relname_to_id = {name: idx for idx, name in relation_map.items()}
    dataset_rule_counts: Counter = Counter()
    per_relation_rule_counts: Dict[int, Counter] = defaultdict(Counter)
    rule_index_meta: Dict[int, Tuple[Optional[int], str]] = {}

    rule_path = ds_dir / "rules" / "rule.txt"
    if not rule_path.exists():
        return dataset_rule_counts, per_relation_rule_counts, rule_index_meta

    valid_rule_idx = 0
    with rule_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parsed = parse_rule_line(line)
            if not parsed:
                continue
            valid_rule_idx += 1
            rule = str(parsed["rule"])
            head = rule.split("<=", 1)[0].strip()
            rel_name = head.split("(", 1)[0]
            rel_id = relname_to_id.get(rel_name)
            t = rule_type(rule)
            dataset_rule_counts[t] += 1
            if rel_id is not None:
                per_relation_rule_counts[rel_id][t] += 1
            rule_index_meta[valid_rule_idx] = (rel_id, t)

    return dataset_rule_counts, per_relation_rule_counts, rule_index_meta


def add_dependency_type_counts(counter: Counter, prefix: str, type_a: str, type_b: str) -> None:
    coarse_a = coarse_rule_type(type_a)
    coarse_b = coarse_rule_type(type_b)
    coarse_pair = "".join(sorted([coarse_a, coarse_b]))
    counter[f"{prefix}_{coarse_pair}"] += 1

    fine_pair = sorted([type_a, type_b])
    counter[f"{prefix}_{fine_pair[0]}_{fine_pair[1]}"] += 1


def parse_dependency_counts(
    ds_dir: Path,
    rule_index_meta: Dict[int, Tuple[Optional[int], str]],
    filenames: List[Tuple[str, str]],
    by_relation: bool,
) -> Tuple[Counter, Dict[int, Counter]]:
    dataset_counter: Counter = Counter()
    per_relation_counter: Dict[int, Counter] = defaultdict(Counter)

    for filename, kind in filenames:
        path = ds_dir / "rules" / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                parts = raw.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    a = int(parts[0])
                    b = int(parts[1])
                except ValueError:
                    continue
                if a not in rule_index_meta or b not in rule_index_meta:
                    continue
                rel_a, type_a = rule_index_meta[a]
                rel_b, type_b = rule_index_meta[b]

                dataset_counter[kind] += 1
                add_dependency_type_counts(dataset_counter, kind, type_a, type_b)

                if by_relation and rel_a is not None and rel_a == rel_b:
                    per_relation_counter[rel_a][kind] += 1
                    add_dependency_type_counts(per_relation_counter[rel_a], kind, type_a, type_b)

    return dataset_counter, per_relation_counter


def aggregation_stage_time(metric: Dict[str, object]) -> Optional[float]:
    return to_float(((metric.get("time_seconds") or {}).get("total")))


def gain_bucket(rel_gain_pct: float) -> str:
    if rel_gain_pct > 3.0:
        return "positive"
    if rel_gain_pct < 0.0:
        return "negative"
    return "neutral"


def stage1_bucket(value: float) -> str:
    if value < 0.2:
        return "[0.0,0.2)"
    if value < 0.4:
        return "[0.2,0.4)"
    if value < 0.6:
        return "[0.4,0.6)"
    return "[0.6,1.0]"


def quartile_bucket(values: List[float], x: float) -> str:
    if not values:
        return "NA"
    q1, q2, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    if x <= q1:
        return "Q1"
    if x <= q2:
        return "Q2"
    if x <= q3:
        return "Q3"
    return "Q4"


def parse_weight_rows(rows: List[List[object]]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for item in rows or []:
        if len(item) < 4:
            continue
        out[str(item[0])] = (float(item[1]), float(item[3]))
    return out


def build_relation_rows(data_root: Path, best_config_map: Dict[str, str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for dataset, aggregation in sorted(best_config_map.items()):
        ds_dir = data_root / dataset
        relation_map = load_relation_map(ds_dir)
        train_counts = count_split_by_relation(ds_dir, "train")
        valid_counts = count_split_by_relation(ds_dir, "valid")
        test_counts = count_split_by_relation(ds_dir, "test")
        dataset_rule_counts, per_relation_rule_counts, rule_index_meta = parse_rules(ds_dir, relation_map)
        _filtered_dataset_dep, filtered_relation_dep = parse_dependency_counts(
            ds_dir,
            rule_index_meta,
            [("synergy_filtered.txt", "filtered_synergy"), ("redundancy_filtered.txt", "filtered_redundancy")],
            by_relation=True,
        )

        baseline_dir = ds_dir / "aggregation" / "structural_none"
        selected_dir = ds_dir / "aggregation" / aggregation
        if not baseline_dir.exists() or not selected_dir.exists():
            continue

        baseline_metrics: Dict[int, Dict[str, object]] = {}
        for metric_path in sorted(baseline_dir.glob("metric-*.json")):
            obj = read_json(metric_path)
            baseline_metrics[int(obj["relation"])] = obj

        for metric_path in sorted(selected_dir.glob("metric-*.json")):
            obj = read_json(metric_path)
            relation = int(obj["relation"])
            baseline_obj = baseline_metrics.get(relation)
            if baseline_obj is None:
                continue

            baseline_stage1 = baseline_obj.get("test_after_stage1") or {}
            selected_stage1 = obj.get("test_after_stage1") or {}
            final_test = obj.get("test") or obj.get("test_after_stage2") or {}
            baseline_stage1_mrr = to_float(baseline_stage1.get("mrr"))
            selected_stage1_mrr = to_float(selected_stage1.get("mrr"))
            final_test_mrr = to_float(final_test.get("mrr"))
            if baseline_stage1_mrr is None or final_test_mrr is None:
                continue

            total_gain = final_test_mrr - baseline_stage1_mrr
            rel_gain_pct = (total_gain / max(baseline_stage1_mrr, 1e-12)) * 100.0
            selected_stage1_gain = None
            if selected_stage1_mrr is not None:
                selected_stage1_gain = selected_stage1_mrr - baseline_stage1_mrr
            stage2_gain_vs_selected_stage1 = None
            if selected_stage1_mrr is not None:
                stage2_gain_vs_selected_stage1 = final_test_mrr - selected_stage1_mrr

            rc = per_relation_rule_counts.get(relation, Counter())
            total_rules = int(sum(rc.values()))
            depc = filtered_relation_dep.get(relation, Counter())
            rule_weight_map = parse_weight_rows((obj.get("params") or {}).get("rule_type_weights") or [])
            dep_final_weight_map = parse_weight_rows((obj.get("params") or {}).get("dependency_type_weights_final") or [])
            dep_trial_weight_map = parse_weight_rows((obj.get("params") or {}).get("dependency_type_weights_trial") or [])
            dep_weight_source = dep_final_weight_map or dep_trial_weight_map
            selection = obj.get("model_selection") or {}

            row = {
                "dataset": dataset,
                "aggregation": aggregation,
                "relation": relation,
                "relation_name": relation_map.get(relation, ""),
                "train_triple_count": int(train_counts.get(relation, 0)),
                "valid_triple_count": int(valid_counts.get(relation, 0)),
                "test_triple_count": int(test_counts.get(relation, obj.get("num_test_samples", 0))),
                "baseline_stage1_mrr": baseline_stage1_mrr,
                "selected_config_stage1_mrr": selected_stage1_mrr,
                "final_test_mrr": final_test_mrr,
                "gain_vs_baseline": total_gain,
                "rel_gain_pct": rel_gain_pct,
                "selected_stage1_gain_vs_baseline": selected_stage1_gain,
                "stage2_gain_vs_selected_stage1": stage2_gain_vs_selected_stage1,
                "gain_bucket": gain_bucket(rel_gain_pct),
                "selected_stage": str(selection.get("selected_stage") or ""),
                "dependency_stage_attempted": bool(selection.get("dependency_stage_attempted", False)),
                "dependency_stage_accepted": bool(selection.get("dependency_stage_accepted", False)),
                "valid_stage1_mrr": to_float(((obj.get("best_valid_stage1") or {}).get("mrr"))),
                "valid_stage2_mrr": to_float(((obj.get("best_valid_stage2") or {}).get("mrr"))),
                "selected_valid_mrr": selected_valid_mrr(obj),
                "num_relation_rules": int(obj.get("num_relation_rules", 0)),
                "num_relation_dependencies": int(obj.get("num_relation_dependencies", 0)),
                "num_relation_rule_types": int(obj.get("num_relation_rule_types", 0)),
                "num_relation_dependency_types": int(obj.get("num_relation_dependency_types", 0)),
                "num_relation_dependency_type_source_pairs": int(obj.get("num_relation_dependency_type_source_pairs", 0)),
                "parsed_total_rules": total_rules,
                "B_rule_count": int(rc.get("B", 0)),
                "Uc_rule_count": int(rc.get("Uc", 0)),
                "Ud_rule_count": int(rc.get("Ud", 0)),
                "Z_rule_count": int(rc.get("Z", 0)),
                "B_ratio": safe_div(int(rc.get("B", 0)), total_rules),
                "Uc_ratio": safe_div(int(rc.get("Uc", 0)), total_rules),
                "Ud_ratio": safe_div(int(rc.get("Ud", 0)), total_rules),
                "filtered_synergy_count": int(depc.get("filtered_synergy", 0)),
                "filtered_redundancy_count": int(depc.get("filtered_redundancy", 0)),
                "filtered_dep_total": int(depc.get("filtered_synergy", 0) + depc.get("filtered_redundancy", 0)),
                "filtered_dep_per_rule": safe_div(
                    int(depc.get("filtered_synergy", 0) + depc.get("filtered_redundancy", 0)),
                    total_rules,
                ),
                "dep_per_rule": safe_div(int(obj.get("num_relation_dependencies", 0)), max(int(obj.get("num_relation_rules", 0)), 1)),
                "stage1_headroom": 1.0 - baseline_stage1_mrr,
                "relation_time_seconds": aggregation_stage_time(obj),
                "rule_weight_B": None,
                "rule_weight_U": None,
                "rule_weight_Uc": None,
                "rule_weight_Ud": None,
                "dep_weight_BB": None,
                "dep_weight_BU": None,
                "dep_weight_UU": None,
                "dep_weight_B_Uc": None,
                "dep_weight_B_Ud": None,
                "dep_weight_Uc_Uc": None,
                "dep_weight_Uc_Ud": None,
                "dep_weight_Ud_Ud": None,
            }

            for key, (_support, weight) in rule_weight_map.items():
                if key == "B":
                    row["rule_weight_B"] = weight
                elif key == "U":
                    row["rule_weight_U"] = weight
                elif key == "Uc":
                    row["rule_weight_Uc"] = weight
                elif key == "Ud":
                    row["rule_weight_Ud"] = weight

            for key, (_support, weight) in dep_weight_source.items():
                if key == '["B", "B"]':
                    row["dep_weight_BB"] = weight
                elif key == '["B", "U"]':
                    row["dep_weight_BU"] = weight
                elif key == '["U", "U"]':
                    row["dep_weight_UU"] = weight
                elif key == '["B", "Uc"]':
                    row["dep_weight_B_Uc"] = weight
                elif key == '["B", "Ud"]':
                    row["dep_weight_B_Ud"] = weight
                elif key == '["Uc", "Uc"]':
                    row["dep_weight_Uc_Uc"] = weight
                elif key == '["Uc", "Ud"]':
                    row["dep_weight_Uc_Ud"] = weight
                elif key == '["Ud", "Ud"]':
                    row["dep_weight_Ud_Ud"] = weight

            rows.append(row)

    dep_density_values = [float(row["dep_per_rule"]) for row in rows if row["dep_per_rule"] is not None]
    for row in rows:
        row["stage1_bucket"] = stage1_bucket(float(row["baseline_stage1_mrr"]))
        row["dep_density_bucket"] = quartile_bucket(dep_density_values, float(row["dep_per_rule"] or 0.0))

    rows.sort(key=lambda row: (PREFERRED_DATASET_ORDER.index(row["dataset"]) if row["dataset"] in PREFERRED_DATASET_ORDER else 999, row["dataset"], int(row["relation"])))
    return rows


def relation_rows_to_csv_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    csv_rows: List[Dict[str, object]] = []
    for row in rows:
        csv_rows.append(
            {
                key: (
                    fmt_float(value)
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
        )
    return csv_rows


def build_group_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for group in ["positive", "neutral", "negative"]:
        subset = [row for row in rows if row["gain_bucket"] == group]
        out.append(
            {
                "group": group,
                "num_relations": len(subset),
                "selected_dependency_ratio": mean_or_none(1.0 if row["selected_stage"] == "dependency" else 0.0 for row in subset),
                "avg_baseline_stage1_mrr": mean_or_none(row["baseline_stage1_mrr"] for row in subset),
                "median_baseline_stage1_mrr": median_or_none(row["baseline_stage1_mrr"] for row in subset),
                "avg_rel_gain_pct": mean_or_none(row["rel_gain_pct"] for row in subset),
                "median_rel_gain_pct": median_or_none(row["rel_gain_pct"] for row in subset),
                "avg_dep_per_rule": mean_or_none(row["dep_per_rule"] for row in subset),
                "median_dep_per_rule": median_or_none(row["dep_per_rule"] for row in subset),
                "avg_filtered_dep_per_rule": mean_or_none(row["filtered_dep_per_rule"] for row in subset),
                "avg_B_ratio": mean_or_none(row["B_ratio"] for row in subset),
                "avg_Uc_ratio": mean_or_none(row["Uc_ratio"] for row in subset),
                "avg_Ud_ratio": mean_or_none(row["Ud_ratio"] for row in subset),
                "avg_test_triple_count": mean_or_none(row["test_triple_count"] for row in subset),
            }
        )
    return out


def build_dataset_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    for dataset, subset in sorted(grouped.items(), key=lambda item: PREFERRED_DATASET_ORDER.index(item[0]) if item[0] in PREFERRED_DATASET_ORDER else 999):
        counts = Counter(row["gain_bucket"] for row in subset)
        out.append(
            {
                "dataset": dataset,
                "aggregation": subset[0]["aggregation"],
                "num_relations": len(subset),
                "positive_relations": counts.get("positive", 0),
                "neutral_relations": counts.get("neutral", 0),
                "negative_relations": counts.get("negative", 0),
                "avg_rel_gain_pct": mean_or_none(row["rel_gain_pct"] for row in subset),
                "avg_baseline_stage1_mrr": mean_or_none(row["baseline_stage1_mrr"] for row in subset),
                "avg_dep_per_rule": mean_or_none(row["dep_per_rule"] for row in subset),
                "avg_B_ratio": mean_or_none(row["B_ratio"] for row in subset),
                "avg_Uc_ratio": mean_or_none(row["Uc_ratio"] for row in subset),
                "avg_Ud_ratio": mean_or_none(row["Ud_ratio"] for row in subset),
            }
        )
    return out


def build_bucket_summary(rows: List[Dict[str, object]], bucket_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[bucket_key])].append(row)

    out = []
    for bucket, subset in sorted(grouped.items()):
        counts = Counter(row["gain_bucket"] for row in subset)
        out.append(
            {
                "bucket": bucket,
                "num_relations": len(subset),
                "positive_ratio": safe_div(counts.get("positive", 0), len(subset)),
                "negative_ratio": safe_div(counts.get("negative", 0), len(subset)),
                "avg_rel_gain_pct": mean_or_none(row["rel_gain_pct"] for row in subset),
                "median_rel_gain_pct": median_or_none(row["rel_gain_pct"] for row in subset),
                "avg_baseline_stage1_mrr": mean_or_none(row["baseline_stage1_mrr"] for row in subset),
                "avg_dep_per_rule": mean_or_none(row["dep_per_rule"] for row in subset),
            }
        )
    return out


def weighted_average(rows: List[Dict[str, object]], key: str, weight_key: str = "test_triple_count") -> Optional[float]:
    acc = 0.0
    total = 0.0
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key, 0)
        if value is None:
            continue
        weight = float(weight)
        if weight <= 0:
            continue
        acc += float(value) * weight
        total += weight
    if total <= 0:
        return None
    return acc / total


def build_type_weight_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = [
        "rule_weight_B",
        "rule_weight_U",
        "rule_weight_Uc",
        "rule_weight_Ud",
        "dep_weight_BB",
        "dep_weight_BU",
        "dep_weight_UU",
        "dep_weight_B_Uc",
        "dep_weight_B_Ud",
        "dep_weight_Uc_Uc",
        "dep_weight_Uc_Ud",
        "dep_weight_Ud_Ud",
    ]
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    out = []
    for dataset, subset in sorted(grouped.items(), key=lambda item: PREFERRED_DATASET_ORDER.index(item[0]) if item[0] in PREFERRED_DATASET_ORDER else 999):
        row = {"dataset": dataset, "aggregation": subset[0]["aggregation"]}
        for key in keys:
            row[key] = weighted_average(subset, key)
        out.append(row)
    return out


def signed_sqrt_gain(value: float) -> float:
    value = float(value)
    if value == 0.0:
        return 0.0
    return math.copysign(math.sqrt(abs(value)), value)


def apply_signed_sqrt_gain_axis(ax, gains: List[float], ylabel: str) -> None:
    if not gains:
        ax.set_ylabel(ylabel)
        return

    transformed = [signed_sqrt_gain(g) for g in gains]
    ymin = min(transformed)
    ymax = max(transformed)
    span = max(ymax - ymin, 1.0)
    pad = 0.08 * span
    ax.set_ylim(ymin - pad, ymax + pad)

    tick_candidates = [-64, -49, -36, -25, -16, -9, -4, -1, 0, 1, 4, 9, 16, 25, 36, 49, 64]
    gmin = min(gains)
    gmax = max(gains)
    selected = [t for t in tick_candidates if gmin - 1e-9 <= t <= gmax + 1e-9]
    if 0 not in selected:
        selected.append(0)
    selected = sorted(set(selected))

    if selected:
        ax.set_yticks([signed_sqrt_gain(v) for v in selected])
        ax.set_yticklabels([f"{int(v)}" for v in selected])

    ax.set_ylabel(ylabel)


def plot_gain_vs_stage1(rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    datasets = sorted({row["dataset"] for row in rows}, key=lambda name: PREFERRED_DATASET_ORDER.index(name) if name in PREFERRED_DATASET_ORDER else 999)
    cmap = plt.get_cmap("tab10")
    all_gains = [float(row["rel_gain_pct"]) for row in rows]
    for idx, dataset in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        plt.scatter(
            [row["baseline_stage1_mrr"] for row in subset],
            [signed_sqrt_gain(float(row["rel_gain_pct"])) for row in subset],
            s=[max(20, math.sqrt(max(row["test_triple_count"], 1)) * 3) for row in subset],
            alpha=0.75,
            label=dataset,
            color=cmap(idx % 10),
        )
    plt.axhline(signed_sqrt_gain(3.0), color="green", linestyle="--", linewidth=1)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Baseline structural_none stage1 MRR")
    apply_signed_sqrt_gain_axis(plt.gca(), all_gains, "Relative Gain on Final Test (%) [signed sqrt]")
    plt.title("Gain vs Baseline Stage1 Strength")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gain_vs_dep_density(rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    datasets = sorted({row["dataset"] for row in rows}, key=lambda name: PREFERRED_DATASET_ORDER.index(name) if name in PREFERRED_DATASET_ORDER else 999)
    cmap = plt.get_cmap("tab10")
    all_gains = [float(row["rel_gain_pct"]) for row in rows]
    for idx, dataset in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        xs = [math.log10(1.0 + float(row["dep_per_rule"] or 0.0)) for row in subset]
        ys = [signed_sqrt_gain(float(row["rel_gain_pct"])) for row in subset]
        plt.scatter(xs, ys, alpha=0.75, label=dataset, color=cmap(idx % 10))
    plt.axhline(signed_sqrt_gain(3.0), color="green", linestyle="--", linewidth=1)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("log10(1 + dependency per rule)")
    apply_signed_sqrt_gain_axis(plt.gca(), all_gains, "Relative Gain on Final Test (%) [signed sqrt]")
    plt.title("Gain vs Dependency Density")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bucket_summary(summary_rows: List[Dict[str, object]], title: str, out_path: Path) -> None:
    if plt is None:
        return
    labels = [row["bucket"] for row in summary_rows]
    positive = [100.0 * float(row["positive_ratio"] or 0.0) for row in summary_rows]
    negative = [100.0 * float(row["negative_ratio"] or 0.0) for row in summary_rows]
    avg_gain = [float(row["avg_rel_gain_pct"] or 0.0) for row in summary_rows]
    avg_gain_sqrt = [signed_sqrt_gain(v) for v in avg_gain]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, positive, width=width, label="positive ratio (%)", color="#5B8FF9")
    ax1.bar(x + width / 2, negative, width=width, label="negative ratio (%)", color="#E8684A")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Ratio (%)")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, avg_gain_sqrt, color="#222222", marker="o", label="avg gain (%)")
    apply_signed_sqrt_gain_axis(ax2, avg_gain, "Average Gain (%) [signed sqrt]")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dataset_gain_mix(dataset_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    labels = [row["dataset"] for row in dataset_rows]
    pos = [int(row["positive_relations"]) for row in dataset_rows]
    neu = [int(row["neutral_relations"]) for row in dataset_rows]
    neg = [int(row["negative_relations"]) for row in dataset_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(9, 5))
    plt.bar(x, pos, label="positive", color="#5B8FF9")
    plt.bar(x, neu, bottom=pos, label="neutral", color="#C9CCD4")
    plt.bar(x, neg, bottom=np.array(pos) + np.array(neu), label="negative", color="#E8684A")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("# relations")
    plt.title("Relation Gain Mix by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_type_weight_summary(type_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None or not type_rows:
        return
    datasets = [row["dataset"] for row in type_rows]
    rule_keys = ["rule_weight_B", "rule_weight_U", "rule_weight_Uc", "rule_weight_Ud"]
    dep_keys = ["dep_weight_BB", "dep_weight_BU", "dep_weight_UU", "dep_weight_Uc_Ud"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(datasets))
    width = 0.18
    for idx, key in enumerate(rule_keys):
        vals = [row.get(key) if row.get(key) is not None else np.nan for row in type_rows]
        axes[0].bar(x + (idx - 1.5) * width, vals, width=width, label=key.replace("rule_weight_", ""))
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=25, ha="right")
    axes[0].set_title("Average Learned Rule-Type Weights")
    axes[0].legend(fontsize=8)

    width2 = 0.2
    for idx, key in enumerate(dep_keys):
        vals = [row.get(key) if row.get(key) is not None else np.nan for row in type_rows]
        axes[1].bar(x + (idx - 1.5) * width2, vals, width=width2, label=key.replace("dep_weight_", ""))
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=25, ha="right")
    axes[1].set_title("Average Learned Dependency-Type Weights")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_dataset_stats_rows(data_root: Path, datasets: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        ds_dir = data_root / dataset
        relation_map = load_relation_map(ds_dir)
        entities, relations = count_entities_relations(ds_dir)
        dataset_rule_counts, _per_relation_rule_counts, rule_index_meta = parse_rules(ds_dir, relation_map)
        filtered_dep_counts, _ = parse_dependency_counts(
            ds_dir,
            rule_index_meta,
            [("synergy_filtered.txt", "filtered_synergy"), ("redundancy_filtered.txt", "filtered_redundancy")],
            by_relation=False,
        )

        total_rules = int(sum(dataset_rule_counts.values()))
        raw_synergy = line_count(ds_dir / "rules" / "synergy.txt")
        raw_redundancy = line_count(ds_dir / "rules" / "redundancy.txt")
        raw_total = raw_synergy + raw_redundancy
        filtered_total = int(filtered_dep_counts.get("filtered_synergy", 0) + filtered_dep_counts.get("filtered_redundancy", 0))

        row: Dict[str, object] = {
            "dataset": dataset,
            "#entity": entities,
            "#relation": relations,
            "#train": line_count(ds_dir / "train.txt"),
            "#valid": line_count(ds_dir / "valid.txt"),
            "#test": line_count(ds_dir / "test.txt"),
            "#rule": total_rules,
            "#B_rule": int(dataset_rule_counts.get("B", 0)),
            "#Uc_rule": int(dataset_rule_counts.get("Uc", 0)),
            "#Ud_rule": int(dataset_rule_counts.get("Ud", 0)),
            "#Z_rule": int(dataset_rule_counts.get("Z", 0)),
            "#synergy": raw_synergy,
            "#redundancy": raw_redundancy,
            "#filtered_synergy": int(filtered_dep_counts.get("filtered_synergy", 0)),
            "#filtered_redundancy": int(filtered_dep_counts.get("filtered_redundancy", 0)),
            "rules_per_relation": safe_div(total_rules, relations),
            "raw_dependency_total": raw_total,
            "filtered_dependency_total": filtered_total,
            "raw_dep_per_rule": safe_div(raw_total, total_rules),
            "filtered_dep_per_rule": safe_div(filtered_total, total_rules),
            "B_rule_ratio": safe_div(int(dataset_rule_counts.get("B", 0)), total_rules),
            "Uc_rule_ratio": safe_div(int(dataset_rule_counts.get("Uc", 0)), total_rules),
            "Ud_rule_ratio": safe_div(int(dataset_rule_counts.get("Ud", 0)), total_rules),
            "Z_rule_ratio": safe_div(int(dataset_rule_counts.get("Z", 0)), total_rules),
        }

        for actual_kind, counter in [("filtered_synergy", filtered_dep_counts), ("filtered_redundancy", filtered_dep_counts)]:
            for coarse in ["BB", "BU", "UU"]:
                row[f"#{actual_kind}_{coarse}"] = int(counter.get(f"{actual_kind}_{coarse}", 0))
            for fine in ["B_B", "B_Uc", "B_Ud", "Uc_Uc", "Uc_Ud", "Ud_Ud"]:
                row[f"#{actual_kind}_{fine}"] = int(counter.get(f"{actual_kind}_{fine}", 0))

        rows.append(row)

    rows.sort(key=lambda row: PREFERRED_DATASET_ORDER.index(row["dataset"]) if row["dataset"] in PREFERRED_DATASET_ORDER else 999)
    return rows


def dataset_stats_to_csv_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    csv_rows = []
    for row in rows:
        csv_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                csv_row[key] = fmt_float(value)
            else:
                csv_row[key] = value
        csv_rows.append(csv_row)
    return csv_rows


def build_overall_results_md(
    rows: List[Dict[str, object]],
    best_config_rows: List[Dict[str, object]],
    best_config_map: Dict[str, str],
    time_comparison_rows: Optional[List[Dict[str, object]]] = None,
) -> str:
    by_dataset: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["aggregation"]] = row

    lines: List[str] = []
    lines.append("# 0407 Overall Results")
    lines.append("")
    lines.append("This table summarizes all completed experiments in the current warehouse `test` Indicators and time should be expressed in seconds as much as possible.")
    lines.append("")
    lines.append("Related forms:")
    lines.append("")
    lines.append("- `all_results_summary.csv`")
    lines.append("- `best_config_by_dataset.csv`")
    lines.append("- `overall_time_comparison.csv`")
    lines.append("- `all_results_ensemble_debug.json`")
    lines.append("")
    lines.append("Description:")
    lines.append("")
    lines.append("- `eval-maxplus` / `eval-noisyor` from application Log.")
    lines.append("- `canonical` Parse the directory according to the old format:`head_mrr_*.p + tail_mrr_*.p + canonical.log in Done`. ")
    lines.append("- `best_combination*` The configuration has been excluded as a whole and does not participate in this report.")
    lines.append("- `ensemble_best_valid` Yes relation Press in the remaining configurations selected valid MRR Overall after mold selection test Summary.")
    lines.append("- `ensemble_best_test` Yes relation Press in the remaining configurations test MRR Overall after mold selection test summary(oracle upper bound).")
    lines.append("- `ensemble_safe_valid` Is a stable version: priority valid choice, and in the unstable relation Fall back to the best single model at the dataset level.")
    lines.append("- `structural_rd__stage1 / structural_r2d3__stage1 / structural_r3d6__stage1` It's from the same experiment stage1 test. ")
    lines.append("- In the time comparison table,RuleDep stage1/stage2 time use per-relation `epochs_trained` Proportional estimates, and based on the current `multiprocess=2` divide by 2; canonical It is a serial old process and does not divide by 2. ")
    lines.append("")
    lines.append("## Best Non-canonical Config Per Dataset")
    lines.append("")
    lines.append("| Dataset | Best config | Best MRR | Ensemble-valid | Ensemble-safe | Ensemble-test | Canonical | Best app |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for dataset in best_config_map:
        mapping = by_dataset.get(dataset, {})
        best_cfg = best_config_map.get(dataset, "")
        best_mrr = fmt_float((mapping.get(best_cfg) or {}).get("MRR"))
        ensemble_valid_mrr = fmt_float((mapping.get("ensemble_best_valid") or {}).get("MRR"))
        ensemble_safe_mrr = fmt_float((mapping.get("ensemble_safe_valid") or {}).get("MRR"))
        ensemble_test_mrr = fmt_float((mapping.get("ensemble_best_test") or {}).get("MRR"))
        canonical_mrr = fmt_float((mapping.get("canonical") or {}).get("MRR"))
        best_app = max(
            [
                (mapping.get("eval-maxplus") or {}).get("MRR"),
                (mapping.get("eval-noisyor") or {}).get("MRR"),
            ],
            key=lambda value: -1.0 if value is None else float(value),
        )
        lines.append(f"| {dataset} | {best_cfg or '-'} | {best_mrr or '-'} | {ensemble_valid_mrr or '-'} | {ensemble_safe_mrr or '-'} | {ensemble_test_mrr or '-'} | {canonical_mrr or '-'} | {fmt_float(best_app) or '-'} |")

    lines.append("")
    lines.append("## Estimated Runtime Breakdown")
    lines.append("")
    lines.append("| Dataset | RuleDep config | Canonical time (s) | Canonical source | RuleDep stage1 est. (s) | RuleDep stage2 est. (s) | RuleDep total est. (s) |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for row in time_comparison_rows or []:
        lines.append(
            "| {dataset} | {config} | {canonical} | {source} | {stage1} | {stage2} | {total} |".format(
                dataset=row["dataset"],
                config=row["best_config"],
                canonical=fmt_float(row.get("canonical_time_s")) or "-",
                source=row.get("canonical_time_source") or "-",
                stage1=fmt_float(row.get("ruledep_stage1_time_s")) or "-",
                stage2=fmt_float(row.get("ruledep_stage2_time_s")) or "-",
                total=fmt_float(row.get("ruledep_total_time_s")) or "-",
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Number of datasets covered:`{len(best_config_rows)}`")
    lines.append(f"- Has been completed canonical Data set:`{sum(1 for row in best_config_rows if row['canonical'])}`")
    lines.append(f"- Yes ensemble-valid Data set:`{sum(1 for row in best_config_rows if row['ensemble_best_valid'])}`")
    lines.append(f"- Yes ensemble-safe Data set:`{sum(1 for row in best_config_rows if row.get('ensemble_safe_valid'))}`")
    lines.append(f"- Yes ensemble-test Data set:`{sum(1 for row in best_config_rows if row.get('ensemble_best_test'))}`")
    lines.append("")
    return "\n".join(lines)


def build_relation_analysis_md(
    best_config_map: Dict[str, str],
    relation_rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    dataset_rows: List[Dict[str, object]],
) -> str:
    def _to_float(row: Dict[str, object], key: str) -> float:
        return float(row[key])

    def _md_escape(text: object) -> str:
        return str(text).replace("|", "\\|")

    def _median_text(subset: List[Dict[str, object]], key: str) -> str:
        vals = sorted(float(row[key]) for row in subset if row.get(key) not in (None, ""))
        if not vals:
            return "-"
        return fmt_float_short(vals[len(vals) // 2])

    lines: List[str] = []
    lines.append("# 0407 Relation-wise Analysis")
    lines.append("")
    lines.append("This section focuses on a more fine-grained problem: when we select the optimal aggregation After configuration,dependency What exactly has been improved? relation, Which ones again? relation brought about negative migration.")
    lines.append("")
    lines.append("Related forms:")
    lines.append("")
    lines.append("- `relation_dependency_analysis.csv`")
    lines.append("- `relation_gain_group_summary.csv`")
    lines.append("- `relation_gain_dataset_summary.csv`")
    lines.append("- `relation_gain_stage1_bucket_summary.csv`")
    lines.append("- `relation_gain_dep_density_bucket_summary.csv`")
    lines.append("- `relation_relative_gain_gt_3pct_best_config.csv`")
    lines.append("- `relation_relative_gain_lt_0_best_config.csv`")
    lines.append("- `relation_stage2_gain_gt_3pt_best_config.csv`")
    lines.append("")
    lines.append("The experimental caliber is as follows:")
    lines.append("")
    lines.append("- `baseline = structural_none stage1 test_mrr`")
    lines.append("- `final = best_config final test_mrr`")
    lines.append("- `rel_gain_pct = 100 * (final - baseline) / baseline`")
    lines.append("")
    lines.append("The optimal data set-level configuration used by each data set is as follows:")
    lines.append("")
    for dataset in sorted(best_config_map.keys(), key=lambda name: PREFERRED_DATASET_ORDER.index(name) if name in PREFERRED_DATASET_ORDER else 999):
        lines.append(f"- `{dataset}` -> `{best_config_map[dataset]}`")
    lines.append("")

    total_relations = len(relation_rows)
    pos_count = sum(1 for row in relation_rows if float(row["rel_gain_pct"]) > 3.0)
    neg_count = sum(1 for row in relation_rows if float(row["rel_gain_pct"]) < 0.0)
    neu_count = total_relations - pos_count - neg_count
    lines.append("## Main Findings")
    lines.append("")
    lines.append(f"in all `{total_relations}` a relation in,`{pos_count}` a relation The relative gain exceeds `3%`, `{neu_count}` a relation fall on `0%-3%` The stable improvement range of `{neg_count}` a relation Negative migration occurs. On the whole,dependency The benefits are not evenly distributed, but appear more concentrated in a group of "baseline Not yet saturated but with strong structural signal” relation On.")
    lines.append("")
    lines.extend(image_block("plot_gain_vs_stage1.png", "Gain vs Stage1", "Figure 1: relation-level relative gain versus stage1 baseline MRR."))
    lines.extend(image_block("plot_gain_vs_dep_density.png", "Gain vs Dependency Density", "Figure 2: relation-level relative gain versus dependency density."))

    positive_group = next((row for row in group_rows if row["group"] == "positive"), None)
    negative_group = next((row for row in group_rows if row["group"] == "negative"), None)
    positive_rows = [row for row in relation_rows if float(row["rel_gain_pct"]) > 3.0]
    negative_rows = [row for row in relation_rows if float(row["rel_gain_pct"]) < 0.0]
    if positive_group and negative_group:
        lines.append("## Positive vs Negative Relations")
        lines.append("")
        lines.append(
            f"Positive gain relation average stage1 MRR for `{fmt_float_short(positive_group['avg_baseline_stage1_mrr'])}`, below negative gain relation of `{fmt_float_short(negative_group['avg_baseline_stage1_mrr'])}`; "
            f"And its average dependency density for `{fmt_float_short(positive_group['avg_dep_per_rule'])}`, above negative gain relation of `{fmt_float_short(negative_group['avg_dep_per_rule'])}`. "
            " This shows dependency It is easier to help those who still have room for improvement and whose rules interact more densely. relation. "
        )
        lines.append("")
        lines.append("Looking further at the median statistics, positive gain relation The typical scale characteristics are as follows:")
        lines.append("")
        lines.append(f"- `train triples` Median:`{_median_text(positive_rows, 'train_triple_count')}`, Negative gain is `{_median_text(negative_rows, 'train_triple_count')}`")
        lines.append(f"- `test triples` Median:`{_median_text(positive_rows, 'test_triple_count')}`, Negative gain is `{_median_text(negative_rows, 'test_triple_count')}`")
        lines.append(f"- `#rules` Median:`{_median_text(positive_rows, 'num_relation_rules')}`, Negative gain is `{_median_text(negative_rows, 'num_relation_rules')}`")
        lines.append(f"- `#dependencies` Median:`{_median_text(positive_rows, 'num_relation_dependencies')}`, Negative gain is `{_median_text(negative_rows, 'num_relation_dependencies')}`")
        lines.append(f"- `dep_per_rule` Median:`{_median_text(positive_rows, 'dep_per_rule')}`, Negative gain is `{_median_text(negative_rows, 'dep_per_rule')}`")
        lines.append("")
        pos_dep = sum(1 for row in positive_rows if row["selected_stage"] == "dependency")
        pos_rule = sum(1 for row in positive_rows if row["selected_stage"] == "rule_only")
        neg_dep = sum(1 for row in negative_rows if row["selected_stage"] == "dependency")
        neg_rule = sum(1 for row in negative_rows if row["selected_stage"] == "rule_only")
        lines.append("Press the final selected stage Look, positive gain relation more often falls on dependency stage: ")
        lines.append("")
        lines.append(f"- Positive gain relation: `dependency = {pos_dep}`, `rule_only = {pos_rule}`")
        lines.append(f"- negative gain relation: `dependency = {neg_dep}`, `rule_only = {neg_rule}`")
        lines.append("")
        lines.extend(image_block("plot_stage1_bucket_summary.png", "Stage1 Bucket Summary", "Figure 3: average gain across stage1 baseline buckets."))
        lines.extend(image_block("plot_dep_density_bucket_summary.png", "Dependency Density Bucket Summary", "Figure 4: average gain across dependency-density buckets."))

    lines.append("## Dataset-level Pattern")
    lines.append("")
    lines.append("on different data sets relation-level The difference in gain distribution is obvious, indicating that dependency The benefits not only depend on a single relation The local structure also depends on the rule pool of the entire data set and the quality of candidate dependency edges.")
    lines.append("")
    lines.append("| Dataset | Config | Positive | Neutral | Negative | Avg gain pct |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['aggregation']} | {row['positive_relations']} | {row['neutral_relations']} | {row['negative_relations']} | {fmt_float_short(row['avg_rel_gain_pct']) or '-'} |"
        )
    lines.append("")
    lines.extend(image_block("plot_dataset_gain_mix.png", "Dataset Gain Mix", "Figure 5: positive, neutral, and negative relation counts across datasets."))
    lines.extend(image_block("plot_type_weight_summary.png", "Type Weight Summary", "Figure 6: average learned type weights associated with relation-level gain."))

    positive_examples = sorted(positive_rows, key=lambda row: float(row["rel_gain_pct"]), reverse=True)[:10]
    negative_examples = sorted(negative_rows, key=lambda row: float(row["rel_gain_pct"]))[:10]

    lines.append("## Representative Positive Relations")
    lines.append("")
    lines.append("The representative positive example is not necessarily the one with the most training samples. relation. The more common patterns are:baseline There are already available rule signals, but they are not yet saturated; at the same time, the number of rules and dependency The number reaches a certain size, allowing the model to pass rule interaction Make further corrections.")
    lines.append("")
    lines.append("- baseline stage1 It’s not saturated yet, but it’s already there to a certain extent rule signal")
    lines.append("- Often have medium to high rule Number and sum dependency number")
    lines.append("- In many cases, we ultimately chose `dependency` stage, Instead of just relying on stronger ones stage1")
    lines.append("- More suitable to be supported by multiple complementary rules")
    lines.append("")
    lines.append("form file `relation_relative_gain_gt_3pct_best_config.csv` A complete list is given, and the most representative positive examples and their size information are listed below:")
    lines.append("")
    lines.append("| Dataset | Relation | Baseline | Final | Rel gain | Train | Test | #Rules | #Deps | Dep/Rule | Selected stage |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in positive_examples:
        relation_name = _md_escape(row["relation_name"] or row["relation"])
        lines.append(
            f"| {_md_escape(row['dataset'])} | {relation_name} | "
            f"{fmt_float_short(_to_float(row, 'baseline_stage1_mrr'))} | "
            f"{fmt_float_short(_to_float(row, 'final_test_mrr'))} | "
            f"{fmt_float_short(_to_float(row, 'rel_gain_pct'))}% | "
            f"{int(float(row['train_triple_count']))} | "
            f"{int(float(row['test_triple_count']))} | "
            f"{int(float(row['num_relation_rules']))} | "
            f"{int(float(row['num_relation_dependencies']))} | "
            f"{fmt_float_short(_to_float(row, 'dep_per_rule'))} | "
            f"{_md_escape(row['selected_stage'])} |"
        )
    lines.append("")
    lines.append("Two types of patterns can be seen from these representative positive examples:")
    lines.append("")
    lines.append("- One category is `FB15k-237` That kind of high rule number, high dependency counting dense relation, dependency More like existing rule pool Make a strong combination.")
    lines.append("- Another category is `hetionet: DrD` This kind of scale is not large, but the structure is very clear. relation, small quantity high quality dependency It can also bring significant benefits.")
    lines.append("")

    lines.append("## Representative Negative Relations")
    lines.append("")
    lines.append("Negative examples usually correspond to two risks: one is baseline Already strong in itself, extra dependency It is easy to over-correct; the second is dependency Although there are many, the quality is unstable.valid Side preferences cannot be stably transferred to test. ")
    lines.append("")
    for row in negative_examples:
        lines.append(
            f"- `{row['dataset']}` / `{row['relation_name'] or row['relation']}`: baseline `{fmt_float_short(row['baseline_stage1_mrr'])}`, final `{fmt_float_short(row['final_test_mrr'])}`, rel_gain `{fmt_float_short(row['rel_gain_pct'])}%`, selected_stage `{row['selected_stage']}`"
        )
    lines.append("")

    stage2_gain_rows = [
        row
        for row in relation_rows
        if row.get("stage2_gain_vs_selected_stage1") is not None and float(row["stage2_gain_vs_selected_stage1"]) > 0.03
    ]
    stage2_gain_rows = sorted(stage2_gain_rows, key=lambda row: float(row["stage2_gain_vs_selected_stage1"]), reverse=True)

    lines.append("## Stage2 vs Stage1: Gain pt > 3")
    lines.append("")
    lines.append("The following relations satisfy `stage2_gain_vs_selected_stage1 > 0.03` (i.e. to increase beyond 3 percentage points).")
    lines.append("")
    lines.append("| Dataset | Relation | Selected stage1 MRR | Final MRR | Stage2 gain (pt) | Selected stage |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in stage2_gain_rows:
        relation_name = _md_escape(row["relation_name"] or row["relation"])
        lines.append(
            f"| {_md_escape(row['dataset'])} | {relation_name} | "
            f"{fmt_float_short(_to_float(row, 'selected_config_stage1_mrr'))} | "
            f"{fmt_float_short(_to_float(row, 'final_test_mrr'))} | "
            f"{fmt_float_short(100.0 * _to_float(row, 'stage2_gain_vs_selected_stage1'))} | "
            f"{_md_escape(row['selected_stage'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_dataset_analysis_md(rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# 0407 Dataset Analysis")
    lines.append("")
    lines.append("This table counts the size of each data set,rule quantity,dependency quantity, and by type aggregated rule/dependency structure.")
    lines.append("")
    lines.append("Related forms:")
    lines.append("")
    lines.append("- `dataset_size_rule_dependency_stats.csv`")
    lines.append("")
    lines.append("## Headline Table")
    lines.append("")
    lines.append("| Dataset | #entity | #relation | #train | #valid | #test | #rule | #filtered_dep | filtered_dep_per_rule |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['#entity']} | {row['#relation']} | {row['#train']} | {row['#valid']} | {row['#test']} | {row['#rule']} | {row['filtered_dependency_total']} | {fmt_float_short(row['filtered_dep_per_rule']) or '-'} |"
        )
    lines.append("")

    if rows:
        largest_rule = max(rows, key=lambda row: int(row["#rule"]))
        largest_filtered = max(rows, key=lambda row: float(row["filtered_dep_per_rule"] or 0.0))
        most_b = max(rows, key=lambda row: float(row["B_rule_ratio"] or 0.0))
        most_ud = max(rows, key=lambda row: float(row["Ud_rule_ratio"] or 0.0))
        lines.append("## Highlights")
        lines.append("")
        lines.append(f"- Dataset with the most rules:`{largest_rule['dataset']}`, share `{largest_rule['#rule']}` Article rule. ")
        lines.append(f"- filtered dependency The densest data set:`{largest_filtered['dataset']}`, `filtered_dep_per_rule = {fmt_float_short(largest_filtered['filtered_dep_per_rule'])}`. ")
        lines.append(f"- `B` rule The highest proportion:`{most_b['dataset']}`, `B_rule_ratio = {fmt_float_short(most_b['B_rule_ratio'])}`. ")
        lines.append(f"- `Ud` rule The highest proportion:`{most_ud['dataset']}`, `Ud_rule_ratio = {fmt_float_short(most_ud['Ud_rule_ratio'])}`. ")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `FB15k-237 / codex-m / codex-l / YAGO3-10` This kind of data set is more like a large-scale relation-wise rule aggregation scene.")
    lines.append("- `KG20C / WN18RR` The number of relationships is small, but rule and dependency The structure is more compact and suitable for controlled comparison.")
    lines.append("- `hetionet` of graph High in scale and semantic complexity, usually more dependent on stronger structural bias. ")
    lines.append("- `wikidata5m` Currently there are only rule application As a result,dependency / aggregation The statistics are still empty, which is directly reflected in the table.")
    lines.append("")
    return "\n".join(lines)


def build_readme(report_dir: Path) -> str:
    lines = []
    lines.append("# 0407 Report Summary")
    lines.append("")
    lines.append("This catalog contains three parts of analysis:")
    lines.append("")
    lines.append("1. Overall indicator summary and summary")
    lines.append("2. Dataset-by-relationship gain analysis")
    lines.append("3. Data set size and rule/dependency statistics")
    lines.append("")
    lines.append("## Key Files")
    lines.append("")
    lines.append("- `all_results_summary.csv`")
    lines.append("- `all_results_summary.md`")
    lines.append("- `best_config_by_dataset.csv`")
    lines.append("- `all_results_ensemble_debug.json`")
    lines.append("- `relation_dependency_analysis.csv`")
    lines.append("- `relation_relative_gain_gt_3pct_best_config.csv`")
    lines.append("- `relation_relative_gain_lt_0_best_config.csv`")
    lines.append("- `dependency_relation_analysis.md`")
    lines.append("- `dataset_size_rule_dependency_stats.csv`")
    lines.append("- `dataset_analysis.md`")
    if plt is not None:
        lines.append("- `plot_gain_vs_stage1.png`")
        lines.append("- `plot_gain_vs_dep_density.png`")
        lines.append("- `plot_stage1_bucket_summary.png`")
        lines.append("- `plot_dep_density_bucket_summary.png`")
        lines.append("- `plot_dataset_gain_mix.png`")
        lines.append("- `plot_type_weight_summary.png`")
    lines.append("")
    lines.append(f"Generate time directory:`{report_dir}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / "reports" / "0407"
    report_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(data_root)

    overall_rows, ensemble_debug = build_overall_rows(data_root, datasets, safe_ensemble_margin=float(args.safe_ensemble_margin or 0.0))
    overall_csv_rows = overall_rows_to_csv_rows(overall_rows)
    write_csv(
        report_dir / "all_results_summary.csv",
        overall_csv_rows,
        ["dataset", "aggregation", "MRR", "h@1", "h@10", "time"],
    )
    with (report_dir / "all_results_ensemble_debug.json").open("w", encoding="utf-8") as handle:
        json.dump(ensemble_debug, handle, indent=2, ensure_ascii=False)

    forced_best_config = str(args.forced_best_config or "").strip()
    forced_best_config_prefix = str(args.forced_best_config_prefix or "").strip()
    best_config_rows = build_best_config_rows(
        overall_rows,
        datasets,
        forced_best_config=forced_best_config,
        forced_best_config_prefix=forced_best_config_prefix,
    )
    write_csv(
        report_dir / "best_config_by_dataset.csv",
        best_config_rows,
        [
            "dataset",
            "best_config",
            "best_config_mrr",
            "structural_none",
            "structural_rd",
            "structural_r2d3",
            "structural_r3d6",
            "canonical",
            "ensemble_best_valid",
            "ensemble_safe_valid",
            "ensemble_best_test",
            "eval-maxplus",
            "eval-noisyor",
        ],
    )
    best_config_map = overall_best_config_map(
        overall_rows,
        forced_best_config=forced_best_config,
        forced_best_config_prefix=forced_best_config_prefix,
    )
    time_comparison_rows = estimate_ruledep_stage_times(data_root, best_config_rows, overall_rows)
    write_csv(
        report_dir / "overall_time_comparison.csv",
        [
            {
                key: fmt_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in time_comparison_rows
        ],
        [
            "dataset",
            "best_config",
            "canonical_time_s",
            "canonical_time_source",
            "ruledep_stage1_time_s",
            "ruledep_stage2_time_s",
            "ruledep_total_time_s",
            "relation_count",
            "missing_epoch_count",
        ],
    )

    (report_dir / "all_results_summary.md").write_text(
        build_overall_results_md(overall_rows, best_config_rows, best_config_map, time_comparison_rows),
        encoding="utf-8",
    )

    relation_rows = build_relation_rows(data_root, best_config_map)
    if relation_rows:
        relation_csv_rows = relation_rows_to_csv_rows(relation_rows)
        relation_fieldnames = list(relation_rows[0].keys())
        write_csv(report_dir / "relation_dependency_analysis.csv", relation_csv_rows, relation_fieldnames)

        group_rows = build_group_summary(relation_rows)
        write_csv(
            report_dir / "relation_gain_group_summary.csv",
            [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in group_rows],
            list(group_rows[0].keys()),
        )

        dataset_rows = build_dataset_summary(relation_rows)
        write_csv(
            report_dir / "relation_gain_dataset_summary.csv",
            [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in dataset_rows],
            list(dataset_rows[0].keys()),
        )

        stage1_bucket_rows = build_bucket_summary(relation_rows, "stage1_bucket")
        write_csv(
            report_dir / "relation_gain_stage1_bucket_summary.csv",
            [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in stage1_bucket_rows],
            list(stage1_bucket_rows[0].keys()),
        )

        dep_bucket_rows = build_bucket_summary(relation_rows, "dep_density_bucket")
        write_csv(
            report_dir / "relation_gain_dep_density_bucket_summary.csv",
            [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in dep_bucket_rows],
            list(dep_bucket_rows[0].keys()),
        )

        type_rows = build_type_weight_summary(relation_rows)
        write_csv(
            report_dir / "relation_type_weight_summary.csv",
            [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in type_rows],
            list(type_rows[0].keys()),
        )

        positive_rows = [row for row in relation_rows if float(row["rel_gain_pct"]) > 3.0]
        negative_rows = [row for row in relation_rows if float(row["rel_gain_pct"]) < 0.0]
        positive_fields = [
            "dataset",
            "aggregation",
            "relation",
            "relation_name",
            "test_triple_count",
            "baseline_stage1_mrr",
            "selected_config_stage1_mrr",
            "final_test_mrr",
            "gain_vs_baseline",
            "rel_gain_pct",
            "stage2_gain_vs_selected_stage1",
            "selected_stage",
        ]
        positive_csv_rows = []
        for row in positive_rows:
            positive_csv_rows.append({key: fmt_float_short(row[key]) if isinstance(row[key], float) else row[key] for key in positive_fields})
        write_csv(report_dir / "relation_relative_gain_gt_3pct_best_config.csv", positive_csv_rows, positive_fields)

        negative_csv_rows = []
        for row in negative_rows:
            negative_csv_rows.append({key: fmt_float_short(row[key]) if isinstance(row[key], float) else row[key] for key in positive_fields})
        write_csv(report_dir / "relation_relative_gain_lt_0_best_config.csv", negative_csv_rows, positive_fields)

        stage2_gain_gt_3pt_rows = [
            row
            for row in relation_rows
            if row.get("stage2_gain_vs_selected_stage1") is not None and float(row["stage2_gain_vs_selected_stage1"]) > 0.03
        ]
        stage2_gain_fields = [
            "dataset",
            "aggregation",
            "relation",
            "relation_name",
            "test_triple_count",
            "baseline_stage1_mrr",
            "selected_config_stage1_mrr",
            "final_test_mrr",
            "selected_stage1_gain_vs_baseline",
            "stage2_gain_vs_selected_stage1",
            "selected_stage",
        ]
        stage2_gain_csv_rows = []
        for row in stage2_gain_gt_3pt_rows:
            csv_row: Dict[str, object] = {}
            for key in stage2_gain_fields:
                value = row.get(key)
                csv_row[key] = fmt_float_short(value) if isinstance(value, float) else value
            stage2_gain = row.get("stage2_gain_vs_selected_stage1")
            csv_row["stage2_gain_pt"] = fmt_float_short(float(stage2_gain) * 100.0) if isinstance(stage2_gain, float) else ""
            stage2_gain_csv_rows.append(csv_row)
        write_csv(
            report_dir / "relation_stage2_gain_gt_3pt_best_config.csv",
            stage2_gain_csv_rows,
            stage2_gain_fields + ["stage2_gain_pt"],
        )

        (report_dir / "dependency_relation_analysis.md").write_text(
            build_relation_analysis_md(best_config_map, relation_rows, group_rows, dataset_rows),
            encoding="utf-8",
        )

        plot_gain_vs_stage1(relation_rows, report_dir / "plot_gain_vs_stage1.png")
        plot_gain_vs_dep_density(relation_rows, report_dir / "plot_gain_vs_dep_density.png")
        plot_bucket_summary(stage1_bucket_rows, "Gain by Stage1 Bucket", report_dir / "plot_stage1_bucket_summary.png")
        plot_bucket_summary(dep_bucket_rows, "Gain by Dependency Density Bucket", report_dir / "plot_dep_density_bucket_summary.png")
        plot_dataset_gain_mix(dataset_rows, report_dir / "plot_dataset_gain_mix.png")
        plot_type_weight_summary(type_rows, report_dir / "plot_type_weight_summary.png")

    dataset_stats_rows = build_dataset_stats_rows(data_root, datasets)
    dataset_stats_csv_rows = dataset_stats_to_csv_rows(dataset_stats_rows)
    write_csv(report_dir / "dataset_size_rule_dependency_stats.csv", dataset_stats_csv_rows, list(dataset_stats_rows[0].keys()))
    (report_dir / "dataset_analysis.md").write_text(build_dataset_analysis_md(dataset_stats_rows), encoding="utf-8")
    (report_dir / "README.md").write_text(build_readme(report_dir), encoding="utf-8")

    print(f"Wrote overall summary to {report_dir / 'all_results_summary.csv'}")
    print(f"Wrote best config table to {report_dir / 'best_config_by_dataset.csv'}")
    print(f"Wrote relation analysis to {report_dir / 'relation_dependency_analysis.csv'}")
    print(f"Wrote dataset analysis to {report_dir / 'dataset_size_rule_dependency_stats.csv'}")
    print(f"Wrote README to {report_dir / 'README.md'}")
    if plt is None:
        print("matplotlib not available; skipped plot generation")


if __name__ == "__main__":
    main()
