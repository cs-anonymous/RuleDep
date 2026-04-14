#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ModuleNotFoundError:
    plt = None
    LinearSegmentedColormap = None


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

TYPE_GROUPING_EXPERIMENTS = [
    ("structural_none", "none"),
    ("structural_rd", "rd"),
    ("structural_r2d3", "r2d3"),
    ("structural_r3d6", "r3d6"),
]

TYPED_EXPERIMENTS = [
    ("structural_r2d3", "r2d3"),
    ("structural_r3d6", "r3d6"),
]

RULE_WEIGHT_KEYS = ["B", "U", "Uc", "Ud"]
DEP_WEIGHT_KEYS = ["BB", "BU", "UU", "B_Uc", "B_Ud", "Uc_Uc", "Uc_Ud", "Ud_Ud"]

COLOR_PALETTE = [
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#bc80bd",
    "#ccebc5",
    "#a6cee3",
    "#33a02c",
    "#e31a1c",
    "#fdbf6f",
    "#8c510a",
    "#d73027",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 0407 type-weight and rule/dependency-weight analyses.")
    parser.add_argument("--root", default=None, help="Repository root. Defaults to the script parent directory.")
    parser.add_argument("--data-root", default=None, help="Data root. Defaults to <root>/data.")
    parser.add_argument("--report-dir", default=None, help="Report output directory. Defaults to <root>/reports/0407.")
    return parser.parse_args()


def dataset_sort_key(name: str) -> int:
    return PREFERRED_DATASET_ORDER.index(name) if name in PREFERRED_DATASET_ORDER else 999


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_float(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.6f}"


def fmt_float_short(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.5f}"


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def image_block(src: str, alt: str, caption: str, width_pct: int = 60) -> List[str]:
    return [
        f'<p align="center"><img src="{src}" alt="{alt}" width="{width_pct}%"></p>',
        "",
        f"<p align=\"center\"><em>{caption}</em></p>",
        "",
    ]


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    return vals[len(vals) // 2]


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def normalize_dep_key(raw_key: str) -> str:
    key = raw_key.replace('["', "").replace('"]', "").replace('", "', "_")
    key = key.replace('"', "")
    coarse_map = {"B_B": "BB", "B_U": "BU", "U_B": "BU", "U_U": "UU"}
    return coarse_map.get(key, key)


@dataclass
class RunningStats:
    count: int = 0
    sum_original: float = 0.0
    sum_trained: float = 0.0
    sum_abs_delta: float = 0.0
    sum_abs_trained: float = 0.0
    pos_count: int = 0
    neg_count: int = 0
    near_zero_001: int = 0
    near_zero_005: int = 0
    min_trained: float = math.inf
    max_trained: float = -math.inf
    max_abs_trained: float = 0.0

    def add(self, original: float, trained: float) -> None:
        self.count += 1
        self.sum_original += original
        self.sum_trained += trained
        self.sum_abs_delta += abs(trained - original)
        self.sum_abs_trained += abs(trained)
        if trained > 0:
            self.pos_count += 1
        elif trained < 0:
            self.neg_count += 1
        if abs(trained) < 0.01:
            self.near_zero_001 += 1
        if abs(trained) < 0.05:
            self.near_zero_005 += 1
        self.min_trained = min(self.min_trained, trained)
        self.max_trained = max(self.max_trained, trained)
        self.max_abs_trained = max(self.max_abs_trained, abs(trained))

    def as_row(self, dataset: str, aggregation: str, component: str) -> Dict[str, object]:
        return {
            "dataset": dataset,
            "aggregation": aggregation,
            "component": component,
            "count": self.count,
            "mean_original": safe_div(self.sum_original, self.count),
            "mean_trained": safe_div(self.sum_trained, self.count),
            "mean_abs_delta": safe_div(self.sum_abs_delta, self.count),
            "mean_abs_trained": safe_div(self.sum_abs_trained, self.count),
            "positive_ratio": safe_div(self.pos_count, self.count),
            "negative_ratio": safe_div(self.neg_count, self.count),
            "near_zero_ratio_0.01": safe_div(self.near_zero_001, self.count),
            "near_zero_ratio_0.05": safe_div(self.near_zero_005, self.count),
            "min_trained": None if self.count == 0 else self.min_trained,
            "max_trained": None if self.count == 0 else self.max_trained,
            "max_abs_trained": None if self.count == 0 else self.max_abs_trained,
        }


def load_best_configs(report_dir: Path) -> Dict[str, str]:
    path = report_dir / "best_config_by_dataset.csv"
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row["dataset"]
            best = row["best_config"].strip()
            if best:
                mapping[dataset] = best
    return mapping


def load_type_weight_rows(data_root: Path, datasets: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        agg_root = data_root / dataset / "aggregation"
        for exp_name, grouping in TYPE_GROUPING_EXPERIMENTS:
            exp_dir = agg_root / exp_name
            metrics_final = exp_dir / "metrics-final.json"
            if not metrics_final.exists():
                continue
            payload = read_json(metrics_final)
            summary = payload.get("summary") or {}
            test_mrr = float((summary.get("test") or summary.get("test_after_stage2") or {}).get("mrr"))

            weight_sums: Dict[str, float] = defaultdict(float)
            dep_sums: Dict[str, float] = defaultdict(float)
            total_weight = 0.0

            for metric_path in sorted(exp_dir.glob("metric-*.json")):
                obj = read_json(metric_path)
                count = int(obj.get("num_test_samples", 0))
                if count <= 0:
                    continue
                params = obj.get("params") or {}
                total_weight += count
                for item in params.get("rule_type_weights") or []:
                    if len(item) < 4:
                        continue
                    weight_sums[str(item[0])] += count * float(item[3])
                dep_items = params.get("dependency_type_weights_final") or params.get("dependency_type_weights_trial") or []
                for item in dep_items:
                    if len(item) < 4:
                        continue
                    key = normalize_dep_key(str(item[0]))
                    dep_sums[key] += count * float(item[3])

            row: Dict[str, object] = {
                "dataset": dataset,
                "experiment": exp_name,
                "type_grouping": grouping,
                "test_mrr": test_mrr,
            }
            for key in RULE_WEIGHT_KEYS:
                row[f"rule_weight_{key}"] = None if total_weight <= 0 or key not in weight_sums else weight_sums[key] / total_weight
            for key in DEP_WEIGHT_KEYS:
                row[f"dep_weight_{key}"] = None if total_weight <= 0 or key not in dep_sums else dep_sums[key] / total_weight
            rows.append(row)

    rows.sort(key=lambda row: (dataset_sort_key(row["dataset"]), row["dataset"], row["experiment"]))
    return rows


def build_type_grouping_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    out = []
    for dataset, subset in sorted(grouped.items(), key=lambda item: dataset_sort_key(item[0])):
        best = max(subset, key=lambda row: float(row["test_mrr"]))
        row = {
            "dataset": dataset,
            "best_grouping_experiment": best["experiment"],
            "best_type_grouping": best["type_grouping"],
            "best_grouping_mrr": best["test_mrr"],
        }
        for exp_name, grouping in TYPE_GROUPING_EXPERIMENTS:
            match = next((x for x in subset if x["experiment"] == exp_name), None)
            row[exp_name] = None if match is None else match["test_mrr"]
        for key in RULE_WEIGHT_KEYS:
            row[f"best_rule_weight_{key}"] = best.get(f"rule_weight_{key}")
        for key in DEP_WEIGHT_KEYS:
            row[f"best_dep_weight_{key}"] = best.get(f"dep_weight_{key}")
        out.append(row)
    return out


def build_type_grouping_global_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment"]].append(row)
    out = []
    for exp_name, subset in sorted(grouped.items(), key=lambda item: [name for name, _ in TYPE_GROUPING_EXPERIMENTS].index(item[0])):
        row = {
            "experiment": exp_name,
            "type_grouping": subset[0]["type_grouping"],
            "num_datasets": len(subset),
            "avg_test_mrr": mean_or_none(item["test_mrr"] for item in subset),
        }
        for key in RULE_WEIGHT_KEYS:
            row[f"rule_weight_{key}"] = mean_or_none(item.get(f"rule_weight_{key}") for item in subset)
        for key in DEP_WEIGHT_KEYS:
            row[f"dep_weight_{key}"] = mean_or_none(item.get(f"dep_weight_{key}") for item in subset)
        out.append(row)
    return out


def plot_best_type_grouping_counts(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    counts = Counter(row["best_type_grouping"] for row in summary_rows)
    labels = ["none", "rd", "r2d3", "r3d6"]
    values = [counts.get(label, 0) for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=COLOR_PALETTE[: len(labels)])
    plt.ylabel("# datasets")
    plt.title("Best Type Grouping Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_r3d6_rule_weights(global_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    row = next((item for item in global_rows if item["experiment"] == "structural_r3d6"), None)
    if row is None:
        return
    labels = ["B", "Uc", "Ud"]
    values = [row.get(f"rule_weight_{label}") or 0.0 for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=COLOR_PALETTE[: len(labels)])
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1)
    plt.ylabel("Average trained rule-type weight")
    plt.title("Global R3D6 Rule-Type Weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_typed_dependency_interactions(global_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    row = next((item for item in global_rows if item["experiment"] == "structural_r3d6"), None)
    if row is None:
        return
    labels = ["BB", "B_Uc", "B_Ud", "Uc_Uc", "Uc_Ud", "Ud_Ud"]
    values = [row.get(f"dep_weight_{label}") or 0.0 for label in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color=COLOR_PALETTE[: len(labels)])
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1)
    plt.ylabel("Average trained dependency-type weight")
    plt.title("Global R3D6 Dependency Interactions")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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


def choose_best_typed_experiment(data_root: Path, dataset: str) -> Optional[Dict[str, object]]:
    agg_root = data_root / dataset / "aggregation"
    best: Optional[Dict[str, object]] = None
    for exp_name, grouping in TYPED_EXPERIMENTS:
        metrics_final = agg_root / exp_name / "metrics-final.json"
        if not metrics_final.exists():
            continue
        payload = read_json(metrics_final)
        summary = payload.get("summary") or {}
        test_mrr = (summary.get("test") or summary.get("test_after_stage2") or {}).get("mrr")
        if test_mrr is None:
            continue
        candidate = {
            "dataset": dataset,
            "best_typed_experiment": exp_name,
            "best_typed_grouping": grouping,
            "best_typed_mrr": float(test_mrr),
        }
        if best is None or float(candidate["best_typed_mrr"]) > float(best["best_typed_mrr"]):
            best = candidate
    return best


def parse_learned_type_rows(items: List[object], normalize_key: bool = False) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in items or []:
        if len(item) < 4:
            continue
        key = normalize_dep_key(str(item[0])) if normalize_key else str(item[0])
        try:
            support = float(item[1])
            original = float(item[2])
            trained = float(item[3])
        except (TypeError, ValueError):
            continue
        delta = trained - original
        out.append(
            {
                "type": key,
                "support": support,
                "original": original,
                "trained": trained,
                "delta": delta,
                "abs_delta": abs(delta),
                "impact": support * trained,
            }
        )
    return out


def dominant_type(items: List[Dict[str, object]]) -> Dict[str, object]:
    if not items:
        return {
            "type": "",
            "support": None,
            "trained": None,
            "abs_delta": None,
            "impact": None,
        }
    best = max(items, key=lambda row: (float(row["impact"]), float(row["abs_delta"]), float(row["support"]), str(row["type"])))
    return {
        "type": str(best["type"]),
        "support": float(best["support"]),
        "trained": float(best["trained"]),
        "abs_delta": float(best["abs_delta"]),
        "impact": float(best["impact"]),
    }


def normalized_entropy(counter: Counter) -> Optional[float]:
    total = sum(counter.values())
    if total <= 0:
        return None
    non_zero = [count for count in counter.values() if count > 0]
    if len(non_zero) <= 1:
        return 0.0
    probs = [count / total for count in non_zero]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    return entropy / math.log(len(non_zero), 2)


def build_relation_type_weight_rows(
    data_root: Path,
    datasets: List[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dataset_rows: List[Dict[str, object]] = []
    relation_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        best = choose_best_typed_experiment(data_root, dataset)
        if best is None:
            continue
        dataset_rows.append(best)
        exp_dir = data_root / dataset / "aggregation" / str(best["best_typed_experiment"])
        relation_map = load_relation_map(data_root / dataset)

        for metric_path in sorted(exp_dir.glob("metric-*.json")):
            obj = read_json(metric_path)
            params = obj.get("params") or {}
            relation = int(obj.get("relation", -1))
            rule_items = parse_learned_type_rows(params.get("rule_type_weights") or [], normalize_key=False)
            dep_trial_items = parse_learned_type_rows(params.get("dependency_type_weights_trial") or [], normalize_key=True)
            dep_final_items = parse_learned_type_rows(params.get("dependency_type_weights_final") or [], normalize_key=True)
            dep_source = "final" if dep_final_items else ("trial" if dep_trial_items else "none")
            dep_items = dep_final_items if dep_final_items else dep_trial_items
            dom_rule = dominant_type(rule_items)
            dom_dep = dominant_type(dep_items)

            row: Dict[str, object] = {
                "dataset": dataset,
                "aggregation": best["best_typed_experiment"],
                "type_grouping": best["best_typed_grouping"],
                "typed_test_mrr": best["best_typed_mrr"],
                "relation": relation,
                "relation_name": relation_map.get(relation, ""),
                "num_test_samples": int(obj.get("num_test_samples", 0)),
                "selected_stage": str((obj.get("model_selection") or {}).get("selected_stage") or ""),
                "dependency_weight_source": dep_source,
                "dominant_rule_type": dom_rule["type"],
                "dominant_rule_support": dom_rule["support"],
                "dominant_rule_weight": dom_rule["trained"],
                "dominant_rule_abs_delta": dom_rule["abs_delta"],
                "dominant_rule_impact": dom_rule["impact"],
                "dominant_dep_type": dom_dep["type"],
                "dominant_dep_support": dom_dep["support"],
                "dominant_dep_weight": dom_dep["trained"],
                "dominant_dep_abs_delta": dom_dep["abs_delta"],
                "dominant_dep_impact": dom_dep["impact"],
            }

            rule_lookup = {str(item["type"]): item for item in rule_items}
            dep_lookup = {str(item["type"]): item for item in dep_items}
            for key in RULE_WEIGHT_KEYS:
                item = rule_lookup.get(key)
                row[f"rule_weight_{key}"] = None if item is None else item["trained"]
                row[f"rule_support_{key}"] = None if item is None else item["support"]
                row[f"rule_abs_delta_{key}"] = None if item is None else item["abs_delta"]
                row[f"rule_impact_{key}"] = None if item is None else item["impact"]
            for key in DEP_WEIGHT_KEYS:
                item = dep_lookup.get(key)
                row[f"dep_weight_{key}"] = None if item is None else item["trained"]
                row[f"dep_support_{key}"] = None if item is None else item["support"]
                row[f"dep_abs_delta_{key}"] = None if item is None else item["abs_delta"]
                row[f"dep_impact_{key}"] = None if item is None else item["impact"]

            b = row.get("rule_weight_B")
            uc = row.get("rule_weight_Uc")
            ud = row.get("rule_weight_Ud")
            row["rule_order_ud_lt_b_lt_uc"] = None if None in (b, uc, ud) else bool(float(ud) < float(b) < float(uc))
            relation_rows.append(row)

    relation_rows.sort(key=lambda row: (dataset_sort_key(str(row["dataset"])), str(row["dataset"]), int(row["relation"])))
    return dataset_rows, relation_rows


def build_dataset_type_weight_summary(
    dataset_rows: List[Dict[str, object]],
    relation_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in relation_rows:
        grouped[str(row["dataset"])].append(row)

    best_lookup = {str(row["dataset"]): row for row in dataset_rows}
    out: List[Dict[str, object]] = []
    for dataset in sorted(grouped.keys(), key=dataset_sort_key):
        subset = grouped[dataset]
        best = best_lookup[dataset]
        rule_counter = Counter(row["dominant_rule_type"] for row in subset if row["dominant_rule_type"])
        dep_counter = Counter(row["dominant_dep_type"] for row in subset if row["dominant_dep_type"])
        typed_rule_relations = sum(1 for row in subset if row["dominant_rule_type"])
        typed_dep_relations = sum(1 for row in subset if row["dominant_dep_type"])
        row: Dict[str, object] = {
            "dataset": dataset,
            "aggregation": best["best_typed_experiment"],
            "type_grouping": best["best_typed_grouping"],
            "typed_test_mrr": best["best_typed_mrr"],
            "num_relations": len(subset),
            "num_relations_with_rule_type_weights": typed_rule_relations,
            "num_relations_with_dep_type_weights": typed_dep_relations,
            "top_rule_type": "" if not rule_counter else rule_counter.most_common(1)[0][0],
            "top_rule_type_share": None if not rule_counter else rule_counter.most_common(1)[0][1] / len(subset),
            "top_dep_type": "" if not dep_counter else dep_counter.most_common(1)[0][0],
            "top_dep_type_share": None if not dep_counter else dep_counter.most_common(1)[0][1] / len(subset),
            "rule_type_entropy": normalized_entropy(rule_counter),
            "dep_type_entropy": normalized_entropy(dep_counter),
            "ud_lt_b_lt_uc_ratio": mean_or_none(
                1.0 if row2["rule_order_ud_lt_b_lt_uc"] else 0.0
                for row2 in subset
                if row2["rule_order_ud_lt_b_lt_uc"] is not None
            ),
        }
        for key in RULE_WEIGHT_KEYS:
            row[f"dominant_rule_share_{key}"] = rule_counter.get(key, 0) / len(subset)
            row[f"median_rule_weight_{key}"] = mean_or_none(r[f"rule_weight_{key}"] for r in subset)
            row[f"median_rule_impact_{key}"] = median_or_none(r[f"rule_impact_{key}"] for r in subset)
        for key in DEP_WEIGHT_KEYS:
            row[f"dominant_dep_share_{key}"] = dep_counter.get(key, 0) / len(subset)
            row[f"median_dep_weight_{key}"] = mean_or_none(r[f"dep_weight_{key}"] for r in subset)
            row[f"median_dep_impact_{key}"] = median_or_none(r[f"dep_impact_{key}"] for r in subset)
        out.append(row)
    return out


def build_global_type_weight_summary(
    dataset_rows: List[Dict[str, object]],
    relation_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    grouping_counter = Counter(str(row["best_typed_grouping"]) for row in dataset_rows)
    for grouping, count in sorted(grouping_counter.items()):
        out.append({"scope": "dataset_best_typed_grouping", "name": grouping, "value": count})

    rule_counter = Counter(str(row["dominant_rule_type"]) for row in relation_rows if row["dominant_rule_type"])
    for key, count in sorted(rule_counter.items()):
        out.append({"scope": "dominant_rule_type", "name": key, "value": count})

    dep_counter = Counter(str(row["dominant_dep_type"]) for row in relation_rows if row["dominant_dep_type"])
    for key, count in sorted(dep_counter.items()):
        out.append({"scope": "dominant_dep_type", "name": key, "value": count})

    ordering_subset = [row for row in relation_rows if row["rule_order_ud_lt_b_lt_uc"] is not None]
    out.append(
        {
            "scope": "r3d6_relation_ordering",
            "name": "Ud<B<Uc",
            "value": mean_or_none(1.0 if row["rule_order_ud_lt_b_lt_uc"] else 0.0 for row in ordering_subset),
        }
    )
    return out


def plot_dataset_rule_type_dominance(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None or not summary_rows:
        return
    datasets = [str(row["dataset"]) for row in summary_rows]
    labels = ["B", "U", "Uc", "Ud"]
    colors = {"B": COLOR_PALETTE[0], "U": COLOR_PALETTE[1], "Uc": COLOR_PALETTE[2], "Ud": COLOR_PALETTE[3]}
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    plt.figure(figsize=(10, 4))
    for label in labels:
        vals = np.asarray([float(row.get(f"dominant_rule_share_{label}") or 0.0) * 100.0 for row in summary_rows], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=label, color=colors[label])
        bottom += vals
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Share of relations (%)")
    plt.title("Dominant Rule Type by Dataset")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dataset_dependency_type_dominance(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None or not summary_rows:
        return
    datasets = [str(row["dataset"]) for row in summary_rows]
    labels = DEP_WEIGHT_KEYS
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(labels))]
    plt.figure(figsize=(12, 4))
    for idx, label in enumerate(labels):
        vals = np.asarray([float(row.get(f"dominant_dep_share_{label}") or 0.0) * 100.0 for row in summary_rows], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=label, color=colors[idx])
        bottom += vals
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Share of relations (%)")
    plt.title("Dominant Dependency Interaction Type by Dataset")
    plt.legend(ncol=4, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dataset_rule_type_impact_heatmap(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None or LinearSegmentedColormap is None or not summary_rows:
        return
    datasets = [str(row["dataset"]) for row in summary_rows]
    labels = ["B", "U", "Uc", "Ud"]
    matrix = np.full((len(datasets), len(labels)), np.nan)
    for i, row in enumerate(summary_rows):
        for j, label in enumerate(labels):
            value = row.get(f"median_rule_impact_{label}")
            matrix[i, j] = np.nan if value in (None, "") else float(value)
    plt.figure(figsize=(7, 4))
    cmap = LinearSegmentedColormap.from_list("custom_type_weight", COLOR_PALETTE)
    cmap.set_bad(color="#f2f2f2")
    im = plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(datasets)), datasets)
    plt.title("Median Support-weighted Rule-Type Importance")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(im)
    cbar.set_label("median support x trained_weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_simple_csv(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = True
        for raw in handle:
            if first:
                first = False
                continue
            line = raw.strip()
            if not line:
                continue
            yield line.split(",")


def relation_id_from_filename(path: Path, prefix: str) -> Optional[int]:
    name = path.stem
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    try:
        return int(tail)
    except ValueError:
        return None


def load_label_to_id_map(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            mapping[str(parts[1])] = idx
    return mapping


def token_to_id(token: str, mapping: Dict[str, int]) -> Optional[int]:
    if token in mapping:
        return int(mapping[token])
    try:
        return int(token)
    except ValueError:
        return None


def load_test_gold_maps(dataset_dir: Path) -> Tuple[Dict[Tuple[int, int], set], Dict[Tuple[int, int], set]]:
    # sp query key: (subject, relation) -> gold objects
    # po query key: (relation, object) -> gold subjects
    sp_gold: Dict[Tuple[int, int], set] = defaultdict(set)
    po_gold: Dict[Tuple[int, int], set] = defaultdict(set)
    test_path = dataset_dir / "test.txt"
    entity_map = load_label_to_id_map(dataset_dir / "entity_ids.del")
    relation_map = load_label_to_id_map(dataset_dir / "relation_ids.del")
    if not test_path.exists():
        return {}, {}
    with test_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            s = token_to_id(str(parts[0]), entity_map)
            r = token_to_id(str(parts[1]), relation_map)
            o = token_to_id(str(parts[2]), entity_map)
            if s is None or r is None or o is None:
                continue
            sp_gold[(s, r)].add(o)
            po_gold[(r, o)].add(s)
    return sp_gold, po_gold


def compute_gold_qc_effective_metrics(exp_dir: Path, dataset_dir: Path, delta: float = 0.01) -> Dict[str, Optional[float]]:
    application_dir = dataset_dir / "application"
    sp_path = application_dir / "processed_sp_test.pkl"
    po_path = application_dir / "processed_po_test.pkl"
    if not sp_path.exists() or not po_path.exists():
        return {
            "gold_avg_effective_rule_count": None,
            "gold_avg_effective_dep_count": None,
            "gold_top3_rule_share": None,
            "gold_top3_dep_share": None,
            "gold_qc_count": None,
        }

    # relation -> rule_id -> trained weight
    rule_weight_by_rel: Dict[int, Dict[int, float]] = {}
    for path in sorted(exp_dir.glob("weight-*.csv")):
        rel = relation_id_from_filename(path, "weight-")
        if rel is None:
            continue
        rel_map: Dict[int, float] = {}
        for parts in parse_simple_csv(path):
            if len(parts) < 3:
                continue
            try:
                rid = int(parts[0])
                trained = float(parts[2])
            except ValueError:
                continue
            rel_map[rid] = trained
        rule_weight_by_rel[rel] = rel_map

    # relation -> adjacency over dep edges with effective_trained > delta
    dep_adj_by_rel: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}
    for path in sorted(exp_dir.glob("dependency-final-*.csv")):
        rel = relation_id_from_filename(path, "dependency-final-")
        if rel is None:
            continue
        adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for parts in parse_simple_csv(path):
            if len(parts) < 7:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                w = float(parts[6])
            except ValueError:
                continue
            if w <= delta:
                continue
            adj[u].append((v, w))
            adj[v].append((u, w))
        dep_adj_by_rel[rel] = dict(adj)

    sp_gold, po_gold = load_test_gold_maps(dataset_dir)

    processed_sp = pickle.load(open(sp_path, "rb"))
    processed_po = pickle.load(open(po_path, "rb"))

    gold_qc_count = 0
    sum_rule_cnt = 0.0
    sum_dep_cnt = 0.0
    sum_rule_top1_share = 0.0
    sum_rule_top3_share = 0.0
    sum_dep_top1_share = 0.0
    sum_dep_top3_share = 0.0
    n_rule_share = 0
    n_dep_share = 0

    def consume(processed: Dict[Tuple[int, int], Dict[str, object]], gold_map: Dict[Tuple[int, int], set], relation_from_key) -> None:
        nonlocal gold_qc_count, sum_rule_cnt, sum_dep_cnt, sum_rule_top1_share, sum_rule_top3_share, sum_dep_top1_share, sum_dep_top3_share, n_rule_share, n_dep_share
        for key, entry in processed.items():
            gold_set = gold_map.get((int(key[0]), int(key[1])))
            if not gold_set:
                continue
            relation = relation_from_key(key)
            rule_weight_map = rule_weight_by_rel.get(relation, {})
            dep_adj = dep_adj_by_rel.get(relation, {})
            candidates = entry.get("candidates") or []
            rules_by_cand = entry.get("rules") or []
            if not isinstance(candidates, list) or not isinstance(rules_by_cand, list):
                continue
            for idx, cand in enumerate(candidates):
                try:
                    cand_int = int(cand)
                except (TypeError, ValueError):
                    continue
                if cand_int not in gold_set:
                    continue
                if idx >= len(rules_by_cand):
                    continue
                rule_ids = rules_by_cand[idx]
                if not isinstance(rule_ids, list):
                    continue

                effective_rule_weights: List[float] = []
                active_rules_set = set()
                for rid in rule_ids:
                    try:
                        rid_int = int(rid)
                    except (TypeError, ValueError):
                        continue
                    w = float(rule_weight_map.get(rid_int, 0.0))
                    if w > delta:
                        effective_rule_weights.append(w)
                        active_rules_set.add(rid_int)

                rule_cnt = len(effective_rule_weights)
                sum_rule_cnt += float(rule_cnt)

                if effective_rule_weights:
                    total_rule = sum(effective_rule_weights)
                    if total_rule > 0.0:
                        top1_rule = max(effective_rule_weights)
                        top3_rule = sum(sorted(effective_rule_weights, reverse=True)[:3])
                        sum_rule_top1_share += top1_rule / total_rule
                        sum_rule_top3_share += top3_rule / total_rule
                        n_rule_share += 1

                dep_weights: List[float] = []
                if len(active_rules_set) >= 2 and dep_adj:
                    # undirected unique pairs
                    for u in active_rules_set:
                        for v, w in dep_adj.get(u, []):
                            if v in active_rules_set and u < v:
                                dep_weights.append(w)

                dep_cnt = len(dep_weights)
                sum_dep_cnt += float(dep_cnt)

                if dep_weights:
                    total_dep = sum(dep_weights)
                    if total_dep > 0.0:
                        top1_dep = max(dep_weights)
                        top3_dep = sum(sorted(dep_weights, reverse=True)[:3])
                        sum_dep_top1_share += top1_dep / total_dep
                        sum_dep_top3_share += top3_dep / total_dep
                        n_dep_share += 1

                gold_qc_count += 1

    consume(processed_sp, sp_gold, lambda key: int(key[1]))
    consume(processed_po, po_gold, lambda key: int(key[0]))

    if gold_qc_count <= 0:
        return {
            "gold_avg_effective_rule_count": None,
            "gold_avg_effective_dep_count": None,
            "gold_top1_rule_share": None,
            "gold_top1_dep_share": None,
            "gold_top3_rule_share": None,
            "gold_top3_dep_share": None,
            "gold_qc_count": 0.0,
        }

    return {
        "gold_avg_effective_rule_count": sum_rule_cnt / float(gold_qc_count),
        "gold_avg_effective_dep_count": sum_dep_cnt / float(gold_qc_count),
        "gold_top1_rule_share": None if n_rule_share <= 0 else sum_rule_top1_share / float(n_rule_share),
        "gold_top1_dep_share": None if n_dep_share <= 0 else sum_dep_top1_share / float(n_dep_share),
        "gold_top3_rule_share": None if n_rule_share <= 0 else sum_rule_top3_share / float(n_rule_share),
        "gold_top3_dep_share": None if n_dep_share <= 0 else sum_dep_top3_share / float(n_dep_share),
        "gold_qc_count": float(gold_qc_count),
    }


def build_weight_analysis(data_root: Path, best_configs: Dict[str, str]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    dep_type_rows: List[Dict[str, object]] = []

    for dataset, aggregation in sorted(best_configs.items(), key=lambda item: dataset_sort_key(item[0])):
        exp_dir = data_root / dataset / "aggregation" / aggregation
        dataset_dir = data_root / dataset
        rule_stats = RunningStats()
        dep_trial_stats = RunningStats()
        dep_final_stats = RunningStats()
        dep_type_stats: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
        gold_delta = 0.01

        for path in sorted(exp_dir.glob("weight-*.csv")):
            for parts in parse_simple_csv(path):
                if len(parts) < 3:
                    continue
                try:
                    original = float(parts[1])
                    trained = float(parts[2])
                except ValueError:
                    continue
                rule_stats.add(original, trained)

        for stage_name, glob_pattern, stage_stats in [
            ("dependency_trial", "dependency-trial-*.csv", dep_trial_stats),
            ("dependency_final", "dependency-final-*.csv", dep_final_stats),
        ]:
            for path in sorted(exp_dir.glob(glob_pattern)):
                for parts in parse_simple_csv(path):
                    if len(parts) < 7:
                        continue
                    dep_type = parts[2]
                    try:
                        effective_original = float(parts[5])
                        effective_trained = float(parts[6])
                    except ValueError:
                        continue
                    stage_stats.add(effective_original, effective_trained)
                    dep_type_stats[(stage_name, dep_type)].add(effective_original, effective_trained)

        gold_qc = compute_gold_qc_effective_metrics(exp_dir, dataset_dir, delta=gold_delta)

        for component, stats in [
            ("rule", rule_stats),
            ("dependency_trial", dep_trial_stats),
            ("dependency_final", dep_final_stats),
        ]:
            row = stats.as_row(dataset, aggregation, component)
            row["gold_delta"] = gold_delta
            row["gold_avg_effective_count"] = None
            row["gold_top1_share"] = None
            row["gold_top3_share"] = None
            row["gold_qc_count"] = gold_qc.get("gold_qc_count")
            if component == "rule":
                row["gold_avg_effective_count"] = gold_qc.get("gold_avg_effective_rule_count")
                row["gold_top1_share"] = gold_qc.get("gold_top1_rule_share")
                row["gold_top3_share"] = gold_qc.get("gold_top3_rule_share")
            elif component == "dependency_final":
                row["gold_avg_effective_count"] = gold_qc.get("gold_avg_effective_dep_count")
                row["gold_top1_share"] = gold_qc.get("gold_top1_dep_share")
                row["gold_top3_share"] = gold_qc.get("gold_top3_dep_share")
            summary_rows.append(row)

        for (stage_name, dep_type), stats in sorted(dep_type_stats.items()):
            row = stats.as_row(dataset, aggregation, stage_name)
            row["dep_type"] = dep_type
            dep_type_rows.append(row)

    return summary_rows, dep_type_rows


def plot_weight_near_zero(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    datasets = sorted({row["dataset"] for row in summary_rows}, key=dataset_sort_key)
    components = ["rule", "dependency_trial", "dependency_final"]
    color_map = {"rule": COLOR_PALETTE[0], "dependency_trial": COLOR_PALETTE[1], "dependency_final": COLOR_PALETTE[6]}
    x = np.arange(len(datasets))
    width = 0.25
    plt.figure(figsize=(10, 4))
    for idx, component in enumerate(components):
        vals = []
        for dataset in datasets:
            match = next((row for row in summary_rows if row["dataset"] == dataset and row["component"] == component), None)
            vals.append(100.0 * float(match["near_zero_ratio_0.01"] or 0.0) if match else 0.0)
        plt.bar(x + (idx - 1) * width, vals, width=width, label=component, color=color_map[component])
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Ratio with |trained weight| < 0.01 (%)")
    plt.title("Near-zero Weight Ratio by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weight_max_abs(summary_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    datasets = sorted({row["dataset"] for row in summary_rows}, key=dataset_sort_key)
    components = ["rule", "dependency_trial", "dependency_final"]
    color_map = {"rule": COLOR_PALETTE[0], "dependency_trial": COLOR_PALETTE[1], "dependency_final": COLOR_PALETTE[6]}
    x = np.arange(len(datasets))
    width = 0.25
    plt.figure(figsize=(10, 4))
    for idx, component in enumerate(components):
        vals = []
        for dataset in datasets:
            match = next((row for row in summary_rows if row["dataset"] == dataset and row["component"] == component), None)
            vals.append(float(match["max_abs_trained"] or 0.0) if match else 0.0)
        plt.bar(x + (idx - 1) * width, vals, width=width, label=component, color=color_map[component])
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Max absolute trained weight")
    plt.title("Largest Learned Weight by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dependency_sign_by_type(dep_type_rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    labels = ["dependency_trial:synergy", "dependency_trial:redundancy", "dependency_final:synergy", "dependency_final:redundancy"]
    values = []
    for stage, dep_type in [("dependency_trial", "synergy"), ("dependency_trial", "redundancy"), ("dependency_final", "synergy"), ("dependency_final", "redundancy")]:
        subset = [
            row
            for row in dep_type_rows
            if row["component"] == stage and row["dep_type"] == dep_type and float(row["count"] or 0.0) > 0.0
        ]
        values.append(100.0 * float(mean_or_none(row["positive_ratio"] for row in subset) or 0.0))
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color=[COLOR_PALETTE[1], COLOR_PALETTE[6], COLOR_PALETTE[3], COLOR_PALETTE[7]])
    plt.ylabel("Positive trained-weight ratio (%)")
    plt.title("Dependency Weight Sign by Type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_type_weight_md(
    dataset_summary_rows: List[Dict[str, object]],
    relation_rows: List[Dict[str, object]],
    global_rows: List[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# 0407 Type-weight Analysis")
    lines.append("")
    lines.append("这一节不再把 type weight 直接做成单个实验级平均值，而是回到 relation 粒度来问：在每个数据集最优的 typed 实验里，究竟是哪一类 rule type 或 dependency interaction 真正在起作用。")
    lines.append("")
    lines.append("这里将某个 type 在一个 relation 上的重要性定义为：`support x trained_weight`。")
    lines.append("")
    lines.append("- `trained_weight` 是最终学到的乘性系数，因此权重越大，说明模型越愿意放大该 type 的贡献。")
    lines.append("- `support` 反映这个 type 在该 relation 上覆盖了多少规则或 rule pair。")
    lines.append("- 因而 `support x trained_weight` 可以理解为该 type 在 relation 上的总“有效质量”；若某个 type 的权重小于 `1`，它的相对重要性也会自然下降。")
    lines.append("")
    lines.append("相关表格：")
    lines.append("")
    lines.append("- `best_typed_experiment_by_dataset.csv`")
    lines.append("- `relation_type_weight_importance.csv`")
    lines.append("- `dataset_type_weight_summary.csv`")
    lines.append("- `global_type_weight_summary.csv`")
    lines.append("")
    lines.extend(image_block("plot_dataset_rule_type_dominance.png", "Dataset Rule Type Dominance", "Figure 1: share of relations whose dominant rule type is B / U / Uc / Ud in each dataset."))
    lines.extend(image_block("plot_dataset_dependency_type_dominance.png", "Dataset Dependency Type Dominance", "Figure 2: share of relations whose dominant dependency interaction type differs across datasets."))
    lines.extend(image_block("plot_dataset_rule_type_impact_heatmap.png", "Dataset Rule Type Impact Heatmap", "Figure 3: median support-weighted importance of each rule type in each dataset."))

    lines.append("## Best Typed Experiment Per Dataset")
    lines.append("")
    lines.append("为了避免把“是否使用 type weight”与“type weight 学成什么样”混在一起，这里先对每个数据集只在显式带 type weight 的实验中选一个最优配置，即 `r2d3` 与 `r3d6` 二选一。")
    lines.append("")
    lines.append("| Dataset | Best typed experiment | Grouping | Test MRR | Top rule type | Top dependency type |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for row in dataset_summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['aggregation']} | {row['type_grouping']} | {fmt_float_short(row['typed_test_mrr'])} | "
            f"{row['top_rule_type'] or '-'} | {row['top_dep_type'] or '-'} |"
        )
    lines.append("")

    typed_grouping_counter = Counter(str(row["type_grouping"]) for row in dataset_summary_rows)
    lines.append("## Dataset-level Pattern")
    lines.append("")
    lines.append(
        f"在这批结果中，最佳 typed 实验的 grouping 分布为 "
        f"`r2d3 = {typed_grouping_counter.get('r2d3', 0)}`，`r3d6 = {typed_grouping_counter.get('r3d6', 0)}`。"
        " 但这一步只是选择分析入口，真正关键的是进入该实验后，不同 relation 对不同 type 的偏好是否一致。"
    )
    lines.append("")

    overall_rule_counter = Counter(str(row["dominant_rule_type"]) for row in relation_rows if row["dominant_rule_type"])
    overall_dep_counter = Counter(str(row["dominant_dep_type"]) for row in relation_rows if row["dominant_dep_type"])
    top_rule = overall_rule_counter.most_common(1)[0] if overall_rule_counter else ("", 0)
    top_dep = overall_dep_counter.most_common(1)[0] if overall_dep_counter else ("", 0)
    lines.append(
        f"按 relation 计数，整体上最常成为主导项的 rule type 是 `{top_rule[0]}`，最常成为主导项的 dependency interaction 是 `{top_dep[0]}`。"
    )
    lines.append("")

    ordering_row = next((row for row in global_rows if row["scope"] == "r3d6_relation_ordering"), None)
    if ordering_row and ordering_row.get("value") not in (None, ""):
        lines.append(
            f"如果只在能观测到 `B / Uc / Ud` 三类权重的 relation 上检查，满足 `Ud < B < Uc` 的比例为 `{fmt_float_short(float(ordering_row['value']) * 100.0)}%`。"
            " 这比“直接看全局平均值”更合理，因为它保留了 relation 之间的差异。"
        )
        lines.append("")

    lines.append("## Within-dataset Heterogeneity")
    lines.append("")
    lines.append("下面的统计更能说明问题：如果某个数据集所有 relation 都偏好同一种 type，那么它的 dominant-type entropy 会很低；反过来，如果不同 relation 各自依赖不同的 type，entropy 就会更高。")
    lines.append("")
    for row in dataset_summary_rows:
        lines.append(
            f"- `{row['dataset']}`: dominant rule type 最常见的是 `{row['top_rule_type'] or '-'}`，占 `{fmt_float_short(100.0 * float(row['top_rule_type_share'] or 0.0))}%`；"
            f"rule-type entropy 为 `{fmt_float_short(row['rule_type_entropy'])}`，dependency-type entropy 为 `{fmt_float_short(row['dep_type_entropy'])}`。"
        )
    lines.append("")

    lines.append("## What Is Important In Each Dataset")
    lines.append("")
    lines.append("如果把“更重要”理解为 `support x trained_weight` 更高，那么不同数据集的主导 type 确实明显不同。")
    lines.append("")
    for row in dataset_summary_rows:
        rule_candidates = [(key, row.get(f"median_rule_impact_{key}")) for key in RULE_WEIGHT_KEYS if row.get(f"median_rule_impact_{key}") not in (None, "")]
        dep_candidates = [(key, row.get(f"median_dep_impact_{key}")) for key in DEP_WEIGHT_KEYS if row.get(f"median_dep_impact_{key}") not in (None, "")]
        best_rule = max(rule_candidates, key=lambda item: float(item[1])) if rule_candidates else ("-", None)
        best_dep = max(dep_candidates, key=lambda item: float(item[1])) if dep_candidates else ("-", None)
        lines.append(
            f"- `{row['dataset']}`: rule 侧最重要的 type 是 `{best_rule[0]}`，median importance `{fmt_float_short(best_rule[1])}`；"
            f"dependency 侧最重要的 interaction 是 `{best_dep[0]}`，median importance `{fmt_float_short(best_dep[1])}`。"
        )
    lines.append("")

    lines.append("## Representative Relation-level Diversity")
    lines.append("")
    lines.append("relation 级别的差异并不会被数据集平均值完全解释。下面列出若干代表性 relation，展示同一数据集内部也会出现不同的主导 type。")
    lines.append("")
    shown = 0
    seen_pairs = set()
    for row in sorted(relation_rows, key=lambda item: float(item["dominant_rule_impact"] or 0.0), reverse=True):
        key = (str(row["dataset"]), str(row["dominant_rule_type"]))
        if not row["dominant_rule_type"] or key in seen_pairs:
            continue
        seen_pairs.add(key)
        lines.append(
            f"- `{row['dataset']}` / `{row['relation_name'] or row['relation']}`: dominant rule type = `{row['dominant_rule_type']}`, "
            f"weight `{fmt_float_short(row['dominant_rule_weight'])}`, support `{fmt_float_short(row['dominant_rule_support'])}`, "
            f"importance `{fmt_float_short(row['dominant_rule_impact'])}`; dominant dependency type = `{row['dominant_dep_type'] or '-'}`."
        )
        shown += 1
        if shown >= 12:
            break
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("这次按 relation 粒度重做后，可以看到 type weight 的作用并不是“整个实验学到一组固定的全局偏好”，而更像是模型针对不同 relation 的局部结构自适应地调整不同 type。")
    lines.append("")
    lines.append("因此，第 4 部分更合理的结论不该是“某个 grouping 的全局平均权重大小关系如何”，而应当是：")
    lines.append("")
    lines.append("- 不同数据集的主导 rule type 和主导 dependency interaction 确实不同。")
    lines.append("- 同一数据集内部，不同 relation 的 dominant type 也会显著变化。")
    lines.append("- `Ud < B < Uc` 若要分析，也应在 relation 级别或数据集级别检查其成立比例，而不是只看被平均之后的一组数字。")
    lines.append("")
    return "\n".join(lines)


def build_rule_dependency_weight_md(summary_rows: List[Dict[str, object]], dep_type_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# 0407 Rule / Dependency Weight Analysis")
    lines.append("")
    lines.append("本节沿用第 2 部分的 best config，仅使用 `dependency_final`，考察 rule 与 dependency 的参数变化，以及最终被保留的 dependency 权重分布。")
    lines.append("")
    lines.append("相关表格：")
    lines.append("")
    lines.append("- `best_config_weight_summary.csv`")
    lines.append("- `dependency_sign_by_type.csv`")
    lines.append("")
    lines.extend(image_block("plot_weight_near_zero_ratio.png", "Weight Near-zero Ratio", "Figure 1: near-zero ratio of learned rule and dependency weights."))
    lines.extend(image_block("plot_weight_max_abs.png", "Weight Max Abs", "Figure 2: maximum absolute value of learned rule and dependency weights."))
    lines.extend(image_block("plot_dependency_sign_by_type.png", "Dependency Sign by Type", "Figure 3: positive-weight ratio of synergy and redundancy dependencies in dependency_final."))
    lines.append("## Definitions (gold-effective metrics)")
    lines.append("")
    lines.append("下面新增 4 个统计，统一基于 `dependency_final` 与阈值 `delta=0.01`，并且都在 gold `(q,c)` 粒度上计算：")
    lines.append("")
    lines.append("- gold 平均有效 Rule 数量：对每个 gold `(q,c)`，统计该候选上 `rule_weight > delta` 的规则条目数，再对所有 gold `(q,c)` 求平均。")
    lines.append("- gold 平均有效 Dep 数量：对每个 gold `(q,c)`，统计由激活规则诱导、且 `dep_weight > delta` 的 dependency 边数，再对所有 gold `(q,c)` 求平均。")
    lines.append("- gold top3 Rule 占比：对每个 gold `(q,c)`，`sum(top3(rule_weight>delta)) / sum(rule_weight>delta)`，再求平均。")
    lines.append("- gold top3 Dep 占比：对每个 gold `(q,c)`，`sum(top3(dep_weight>delta)) / sum(dep_weight>delta)`，再求平均。")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Config | Rule near-zero | Dep final near-zero | Rule max abs | Dep final max abs | Gold avg eff Rule per (q,c) | Gold avg eff Dep per (q,c) | Gold top1 Rule share | Gold top1 Dep share | Gold top3 Rule share | Gold top3 Dep share |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    grouped = {(row["dataset"], row["component"]): row for row in summary_rows}
    dataset_vals: List[Dict[str, Optional[float]]] = []
    for dataset in sorted({row["dataset"] for row in summary_rows}, key=dataset_sort_key):
        rule = grouped.get((dataset, "rule"), {})
        dep_final = grouped.get((dataset, "dependency_final"), {})
        lines.append(
            f"| {dataset} | {rule.get('aggregation','')} | "
            f"{fmt_float_short(100.0 * float(rule.get('near_zero_ratio_0.01') or 0.0))} | "
            f"{fmt_float_short(100.0 * float(dep_final.get('near_zero_ratio_0.01') or 0.0))} | "
            f"{fmt_float_short(rule.get('max_abs_trained'))} | "
            f"{fmt_float_short(dep_final.get('max_abs_trained'))} | "
            f"{fmt_float_short(rule.get('gold_avg_effective_count'))} | "
            f"{fmt_float_short(dep_final.get('gold_avg_effective_count'))} | "
            f"{fmt_float_short(100.0 * float(rule.get('gold_top1_share') or 0.0))}% | "
            f"{fmt_float_short(100.0 * float(dep_final.get('gold_top1_share') or 0.0))}% | "
            f"{fmt_float_short(100.0 * float(rule.get('gold_top3_share') or 0.0))}% | "
            f"{fmt_float_short(100.0 * float(dep_final.get('gold_top3_share') or 0.0))}% |"
        )
        dataset_vals.append(
            {
                "rule_near_zero": to_float(rule.get("near_zero_ratio_0.01")),
                "dep_near_zero": to_float(dep_final.get("near_zero_ratio_0.01")),
                "rule_max_abs": to_float(rule.get("max_abs_trained")),
                "dep_max_abs": to_float(dep_final.get("max_abs_trained")),
                "rule_eff": to_float(rule.get("gold_avg_effective_count")),
                "dep_eff": to_float(dep_final.get("gold_avg_effective_count")),
                "rule_top1": to_float(rule.get("gold_top1_share")),
                "dep_top1": to_float(dep_final.get("gold_top1_share")),
                "rule_top3": to_float(rule.get("gold_top3_share")),
                "dep_top3": to_float(dep_final.get("gold_top3_share")),
            }
        )
    lines.append(
        "| Average | - | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['rule_near_zero'] for r in dataset_vals) or 0.0))} | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['dep_near_zero'] for r in dataset_vals) or 0.0))} | "
        f"{fmt_float_short(mean_or_none(r['rule_max_abs'] for r in dataset_vals))} | "
        f"{fmt_float_short(mean_or_none(r['dep_max_abs'] for r in dataset_vals))} | "
        f"{fmt_float_short(mean_or_none(r['rule_eff'] for r in dataset_vals))} | "
        f"{fmt_float_short(mean_or_none(r['dep_eff'] for r in dataset_vals))} | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['rule_top1'] for r in dataset_vals) or 0.0))}% | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['dep_top1'] for r in dataset_vals) or 0.0))}% | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['rule_top3'] for r in dataset_vals) or 0.0))}% | "
        f"{fmt_float_short(100.0 * float(mean_or_none(r['dep_top3'] for r in dataset_vals) or 0.0))}% |"
    )
    lines.append("")

    final_synergy = [row for row in dep_type_rows if row["component"] == "dependency_final" and row["dep_type"] == "synergy"]
    final_redundancy = [row for row in dep_type_rows if row["component"] == "dependency_final" and row["dep_type"] == "redundancy"]

    lines.append("## Dependency Sign vs Type")
    lines.append("")
    lines.append(
        f"在 `dependency_final` 中，`synergy` 的平均正权重比例为 `{fmt_float_short(100.0 * float(mean_or_none(row['positive_ratio'] for row in final_synergy) or 0.0))}%`，"
        f"`redundancy` 为 `{fmt_float_short(100.0 * float(mean_or_none(row['positive_ratio'] for row in final_redundancy) or 0.0))}%`。"
    )
    lines.append("")

    rule_rows = [row for row in summary_rows if row["component"] == "rule"]
    dep_final_rows = [row for row in summary_rows if row["component"] == "dependency_final"]
    lines.append("## Global View")
    lines.append("")
    lines.append(f"rule 权重的平均绝对变化为 `{fmt_float_short(mean_or_none(row['mean_abs_delta'] for row in rule_rows))}`，dependency(final) 为 `{fmt_float_short(mean_or_none(row['mean_abs_delta'] for row in dep_final_rows))}`。")
    lines.append(
        f"近零比例方面，rule 权重均值为 `{fmt_float_short(100.0 * float(mean_or_none(row['near_zero_ratio_0.01'] for row in rule_rows) or 0.0))}%`，"
        f"dependency(final) 为 `{fmt_float_short(100.0 * float(mean_or_none(row['near_zero_ratio_0.01'] for row in dep_final_rows) or 0.0))}%`。"
    )
    lines.append(
        f"gold 口径下，平均有效 Rule 数量为 `{fmt_float_short(mean_or_none(row['gold_avg_effective_count'] for row in rule_rows))}`，"
        f"平均有效 Dep 数量为 `{fmt_float_short(mean_or_none(row['gold_avg_effective_count'] for row in dep_final_rows))}`。"
    )
    lines.append(
        f"用于统计的 gold `(q,c)` 数量均值为 `{fmt_float_short(mean_or_none(row['gold_qc_count'] for row in rule_rows))}`。"
    )
    lines.append(
        f"gold top1 占比方面，Rule 为 `{fmt_float_short(100.0 * float(mean_or_none(row['gold_top1_share'] for row in rule_rows) or 0.0))}%`，"
        f"Dep 为 `{fmt_float_short(100.0 * float(mean_or_none(row['gold_top1_share'] for row in dep_final_rows) or 0.0))}%`。"
    )
    lines.append(
        f"gold top3 占比方面，Rule 为 `{fmt_float_short(100.0 * float(mean_or_none(row['gold_top3_share'] for row in rule_rows) or 0.0))}%`，"
        f"Dep 为 `{fmt_float_short(100.0 * float(mean_or_none(row['gold_top3_share'] for row in dep_final_rows) or 0.0))}%`。"
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("这些结果共同说明，模型确实会主动稀疏化大量 dependency 边；在最终被保留的 `dependency_final` 中，依然只有少数边具有显著权重，因此它们更像是稀疏但强烈的修正项，而不是均匀分布在所有规则对上的微弱偏置。")
    lines.append("")
    lines.append("从符号分布看，`synergy` 更容易获得正权重，而 `redundancy` 通常更保守，这与依赖类型本身的语义方向基本一致，但并不是绝对的一一对应关系。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve() if args.data_root else root / "data"
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / "reports" / "0407"
    report_dir.mkdir(parents=True, exist_ok=True)

    best_configs = load_best_configs(report_dir)
    datasets = sorted(best_configs.keys(), key=dataset_sort_key)

    type_weight_rows = load_type_weight_rows(data_root, datasets)
    if type_weight_rows:
        type_weight_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in type_weight_rows]
        write_csv(report_dir / "type_grouping_weight_summary.csv", type_weight_csv_rows, list(type_weight_rows[0].keys()))

        type_grouping_summary = build_type_grouping_summary(type_weight_rows)
        type_grouping_summary_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in type_grouping_summary]
        write_csv(report_dir / "type_grouping_summary.csv", type_grouping_summary_csv_rows, list(type_grouping_summary[0].keys()))

        type_global_summary_legacy = build_type_grouping_global_summary(type_weight_rows)
        type_global_summary_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in type_global_summary_legacy]
        write_csv(report_dir / "type_grouping_global_summary.csv", type_global_summary_csv_rows, list(type_global_summary_legacy[0].keys()))

    best_typed_rows, relation_type_rows = build_relation_type_weight_rows(data_root, datasets)
    dataset_type_summary = build_dataset_type_weight_summary(best_typed_rows, relation_type_rows)
    global_type_summary = build_global_type_weight_summary(best_typed_rows, relation_type_rows)

    if best_typed_rows:
        best_typed_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in best_typed_rows]
        write_csv(report_dir / "best_typed_experiment_by_dataset.csv", best_typed_csv_rows, list(best_typed_rows[0].keys()))
    if relation_type_rows:
        relation_type_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in relation_type_rows]
        write_csv(report_dir / "relation_type_weight_importance.csv", relation_type_csv_rows, list(relation_type_rows[0].keys()))
    if dataset_type_summary:
        dataset_type_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in dataset_type_summary]
        write_csv(report_dir / "dataset_type_weight_summary.csv", dataset_type_csv_rows, list(dataset_type_summary[0].keys()))
    if global_type_summary:
        global_type_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in global_type_summary]
        write_csv(report_dir / "global_type_weight_summary.csv", global_type_csv_rows, list(global_type_summary[0].keys()))

    plot_dataset_rule_type_dominance(dataset_type_summary, report_dir / "plot_dataset_rule_type_dominance.png")
    plot_dataset_dependency_type_dominance(dataset_type_summary, report_dir / "plot_dataset_dependency_type_dominance.png")
    plot_dataset_rule_type_impact_heatmap(dataset_type_summary, report_dir / "plot_dataset_rule_type_impact_heatmap.png")
    (report_dir / "type_weight_analysis.md").write_text(
        build_type_weight_md(dataset_type_summary, relation_type_rows, global_type_summary),
        encoding="utf-8",
    )

    weight_summary_rows, dep_type_rows = build_weight_analysis(data_root, best_configs)
    weight_summary_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in weight_summary_rows]
    write_csv(report_dir / "best_config_weight_summary.csv", weight_summary_csv_rows, list(weight_summary_rows[0].keys()))
    dep_type_csv_rows = [{k: fmt_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in dep_type_rows]
    write_csv(report_dir / "dependency_sign_by_type.csv", dep_type_csv_rows, list(dep_type_rows[0].keys()))

    plot_weight_near_zero(weight_summary_rows, report_dir / "plot_weight_near_zero_ratio.png")
    plot_weight_max_abs(weight_summary_rows, report_dir / "plot_weight_max_abs.png")
    plot_dependency_sign_by_type(dep_type_rows, report_dir / "plot_dependency_sign_by_type.png")
    (report_dir / "rule_dependency_weight_analysis.md").write_text(build_rule_dependency_weight_md(weight_summary_rows, dep_type_rows), encoding="utf-8")

    print(f"Wrote {report_dir / 'best_typed_experiment_by_dataset.csv'}")
    print(f"Wrote {report_dir / 'relation_type_weight_importance.csv'}")
    print(f"Wrote {report_dir / 'dataset_type_weight_summary.csv'}")
    print(f"Wrote {report_dir / 'best_config_weight_summary.csv'}")
    print(f"Wrote {report_dir / 'type_weight_analysis.md'}")
    print(f"Wrote {report_dir / 'rule_dependency_weight_analysis.md'}")


if __name__ == "__main__":
    main()
