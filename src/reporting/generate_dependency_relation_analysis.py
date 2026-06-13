#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports"
DATA_ROOT = ROOT / "data"
STRUCTURAL_COMPARISON = REPORT_DIR / "structural_filtered_comparison.csv"


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
                rel = int(parts[1])
            except ValueError:
                continue
            counts[rel] += 1
    return counts


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


def coarse_rule_type(t: str) -> str:
    return "B" if t == "B" else "U"


def parse_rules_by_relation(ds_dir: Path, relation_map: Dict[int, str]):
    relname_to_id = {name: idx for idx, name in relation_map.items()}
    per_relation_rule_counts: Dict[int, Counter] = defaultdict(Counter)
    rule_index_meta: Dict[int, Tuple[int, str]] = {}
    rule_path = ds_dir / "rules" / "rule.txt"
    if not rule_path.exists():
        return per_relation_rule_counts, rule_index_meta

    valid_rule_idx = 0
    with rule_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            rule = parts[3]
            head = rule.split("<=", 1)[0].strip()
            rel_name = head.split("(", 1)[0]
            rel = relname_to_id.get(rel_name)
            if rel is None:
                valid_rule_idx += 1
                continue
            t = rule_type(rule)
            per_relation_rule_counts[rel][t] += 1
            valid_rule_idx += 1
            rule_index_meta[valid_rule_idx] = (rel, t)  # dependency files are 1-based
    return per_relation_rule_counts, rule_index_meta


def parse_filtered_dep_counts(ds_dir: Path, rule_index_meta: Dict[int, Tuple[int, str]]):
    result: Dict[int, Counter] = defaultdict(Counter)
    rules_dir = ds_dir / "rules"
    dep_files = [
        ("synergy_filtered.txt", "synergy"),
        ("redundancy_filtered.txt", "redundancy"),
    ]
    for filename, kind in dep_files:
        path = rules_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
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
                if rel_a != rel_b:
                    continue
                rel = rel_a
                result[rel][kind] += 1
                coarse_a = coarse_rule_type(type_a)
                coarse_b = coarse_rule_type(type_b)
                coarse_dep = "".join(sorted([coarse_a, coarse_b]))
                result[rel][f"{kind}_{coarse_dep}"] += 1
                fine_dep = "__".join(sorted([type_a, type_b]))
                result[rel][f"{kind}_{fine_dep}"] += 1
    return result


def best_structural_config_by_dataset() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with STRUCTURAL_COMPARISON.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["dataset"]] = row["best_by_mrr"]
    return mapping


def parse_weight_rows(rows: List[List[object]]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for item in rows or []:
        if len(item) < 4:
            continue
        key = str(item[0])
        support = float(item[1])
        trained = float(item[3])
        out[key] = (support, trained)
    return out


def assign_gain_bucket(rel_gain_pct: float) -> str:
    if rel_gain_pct > 3.0:
        return "positive"
    if rel_gain_pct < -3.0:
        return "negative"
    return "neutral"


def safe_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if isinstance(x, str):
        return x
    return f"{float(x):.5f}"


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


def stage1_bucket(x: float) -> str:
    if x < 0.2:
        return "[0.0,0.2)"
    if x < 0.4:
        return "[0.2,0.4)"
    if x < 0.6:
        return "[0.4,0.6)"
    return "[0.6,1.0]"


def weighted_average(rows: List[Dict[str, object]], key: str, weight_key: str = "test_triple_count") -> float | None:
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


def mean_or_none(values: List[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median_or_none(values: List[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(median(vals))


def extract_relation_rows() -> List[Dict[str, object]]:
    best_cfg = best_structural_config_by_dataset()
    rows: List[Dict[str, object]] = []

    for dataset, aggregation in sorted(best_cfg.items()):
        ds_dir = DATA_ROOT / dataset
        relation_map = load_relation_map(ds_dir)
        train_counts = count_split_by_relation(ds_dir, "train")
        valid_counts = count_split_by_relation(ds_dir, "valid")
        test_counts = count_split_by_relation(ds_dir, "test")
        rule_counts, rule_index_meta = parse_rules_by_relation(ds_dir, relation_map)
        filtered_dep_counts = parse_filtered_dep_counts(ds_dir, rule_index_meta)

        agg_dir = ds_dir / "aggregation" / aggregation
        if not agg_dir.exists():
            continue

        for metric_path in sorted(agg_dir.glob("metric-*.json")):
            if metric_path.name == "metrics-final.json":
                continue
            obj = json.loads(metric_path.read_text())
            relation = int(obj["relation"])
            relation_name = relation_map.get(relation, "")
            stage1 = obj.get("test_after_stage1") or {}
            final_test = obj.get("test") or obj.get("test_after_stage2") or {}
            best_valid_stage1 = obj.get("best_valid_stage1") or {}
            best_valid_stage2 = obj.get("best_valid_stage2") or {}
            model_selection = obj.get("model_selection") or {}
            params = obj.get("params") or {}

            stage1_mrr = safe_float(stage1.get("mrr"))
            final_mrr = safe_float(final_test.get("mrr"))
            if stage1_mrr is None or final_mrr is None:
                continue
            abs_gain = final_mrr - stage1_mrr
            rel_gain_pct = (abs_gain / max(stage1_mrr, 1e-12)) * 100.0

            valid_stage1_mrr = safe_float(best_valid_stage1.get("mrr"))
            valid_stage2_mrr = safe_float(best_valid_stage2.get("mrr"))
            valid_gain = None
            if valid_stage1_mrr is not None and valid_stage2_mrr is not None:
                valid_gain = valid_stage2_mrr - valid_stage1_mrr

            rc = rule_counts.get(relation, Counter())
            total_rule_count = int(sum(rc.values()))
            b_count = int(rc.get("B", 0))
            uc_count = int(rc.get("Uc", 0))
            ud_count = int(rc.get("Ud", 0))
            z_count = int(rc.get("Z", 0))

            depc = filtered_dep_counts.get(relation, Counter())

            rule_weight_map = parse_weight_rows(params.get("rule_type_weights") or [])
            dep_final_weight_map = parse_weight_rows(params.get("dependency_type_weights_final") or [])
            dep_trial_weight_map = parse_weight_rows(params.get("dependency_type_weights_trial") or [])

            row = {
                "dataset": dataset,
                "aggregation": aggregation,
                "relation": relation,
                "relation_name": relation_name,
                "train_triple_count": int(train_counts.get(relation, 0)),
                "valid_triple_count": int(valid_counts.get(relation, 0)),
                "test_triple_count": int(test_counts.get(relation, obj.get("num_test_samples", 0))),
                "stage1_mrr": stage1_mrr,
                "final_test_mrr": final_mrr,
                "abs_gain": abs_gain,
                "rel_gain_pct": rel_gain_pct,
                "gain_bucket": assign_gain_bucket(rel_gain_pct),
                "selected_stage": model_selection.get("selected_stage", ""),
                "dependency_stage_attempted": bool(model_selection.get("dependency_stage_attempted", False)),
                "dependency_stage_accepted": bool(model_selection.get("dependency_stage_accepted", False)),
                "valid_stage1_mrr": valid_stage1_mrr,
                "valid_stage2_mrr": valid_stage2_mrr,
                "valid_gain": valid_gain,
                "num_relation_rules": int(obj.get("num_relation_rules", 0)),
                "num_relation_dependencies": int(obj.get("num_relation_dependencies", 0)),
                "num_relation_rule_types": int(obj.get("num_relation_rule_types", 0)),
                "num_relation_dependency_types": int(obj.get("num_relation_dependency_types", 0)),
                "num_relation_dependency_type_source_pairs": int(obj.get("num_relation_dependency_type_source_pairs", 0)),
                "parsed_total_rules": total_rule_count,
                "B_rule_count": b_count,
                "Uc_rule_count": uc_count,
                "Ud_rule_count": ud_count,
                "Z_rule_count": z_count,
                "B_ratio": (b_count / total_rule_count) if total_rule_count else None,
                "Uc_ratio": (uc_count / total_rule_count) if total_rule_count else None,
                "Ud_ratio": (ud_count / total_rule_count) if total_rule_count else None,
                "filtered_synergy_count": int(depc.get("synergy", 0)),
                "filtered_redundancy_count": int(depc.get("redundancy", 0)),
                "filtered_dep_total": int(depc.get("synergy", 0) + depc.get("redundancy", 0)),
                "filtered_dep_per_rule": ((depc.get("synergy", 0) + depc.get("redundancy", 0)) / total_rule_count)
                if total_rule_count
                else None,
                "dep_per_rule": (int(obj.get("num_relation_dependencies", 0)) / max(int(obj.get("num_relation_rules", 0)), 1)),
                "stage1_headroom": 1.0 - stage1_mrr,
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

            dep_weight_source = dep_final_weight_map or dep_trial_weight_map
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

    dep_density_values = [float(r["dep_per_rule"]) for r in rows if r.get("dep_per_rule") is not None]
    for row in rows:
        row["stage1_bucket"] = stage1_bucket(float(row["stage1_mrr"]))
        row["dep_density_bucket"] = quartile_bucket(dep_density_values, float(row["dep_per_rule"]))
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fieldnames})


def build_group_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for group in ["positive", "neutral", "negative"]:
        subset = [r for r in rows if r["gain_bucket"] == group]
        out.append(
            {
                "group": group,
                "num_relations": len(subset),
                "selected_dependency_ratio": mean_or_none([1.0 if r["selected_stage"] == "dependency" else 0.0 for r in subset]),
                "avg_stage1_mrr": mean_or_none([r["stage1_mrr"] for r in subset]),
                "median_stage1_mrr": median_or_none([r["stage1_mrr"] for r in subset]),
                "avg_rel_gain_pct": mean_or_none([r["rel_gain_pct"] for r in subset]),
                "median_rel_gain_pct": median_or_none([r["rel_gain_pct"] for r in subset]),
                "avg_dep_per_rule": mean_or_none([r["dep_per_rule"] for r in subset]),
                "median_dep_per_rule": median_or_none([r["dep_per_rule"] for r in subset]),
                "avg_filtered_dep_per_rule": mean_or_none([r["filtered_dep_per_rule"] for r in subset if r["filtered_dep_per_rule"] is not None]),
                "avg_B_ratio": mean_or_none([r["B_ratio"] for r in subset if r["B_ratio"] is not None]),
                "avg_Uc_ratio": mean_or_none([r["Uc_ratio"] for r in subset if r["Uc_ratio"] is not None]),
                "avg_Ud_ratio": mean_or_none([r["Ud_ratio"] for r in subset if r["Ud_ratio"] is not None]),
                "avg_test_triple_count": mean_or_none([r["test_triple_count"] for r in subset]),
            }
        )
    return out


def build_dataset_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, subset in sorted(by_dataset.items()):
        counts = Counter(r["gain_bucket"] for r in subset)
        out.append(
            {
                "dataset": dataset,
                "aggregation": subset[0]["aggregation"],
                "num_relations": len(subset),
                "positive_relations": counts.get("positive", 0),
                "neutral_relations": counts.get("neutral", 0),
                "negative_relations": counts.get("negative", 0),
                "avg_rel_gain_pct": mean_or_none([r["rel_gain_pct"] for r in subset]),
                "avg_stage1_mrr": mean_or_none([r["stage1_mrr"] for r in subset]),
                "avg_dep_per_rule": mean_or_none([r["dep_per_rule"] for r in subset]),
                "avg_B_ratio": mean_or_none([r["B_ratio"] for r in subset if r["B_ratio"] is not None]),
                "avg_Uc_ratio": mean_or_none([r["Uc_ratio"] for r in subset if r["Uc_ratio"] is not None]),
                "avg_Ud_ratio": mean_or_none([r["Ud_ratio"] for r in subset if r["Ud_ratio"] is not None]),
            }
        )
    return out


def build_bucket_summary(rows: List[Dict[str, object]], bucket_key: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[bucket_key]].append(row)
    out = []
    for bucket, subset in sorted(groups.items()):
        counts = Counter(r["gain_bucket"] for r in subset)
        out.append(
            {
                "bucket": bucket,
                "num_relations": len(subset),
                "positive_ratio": counts.get("positive", 0) / len(subset) if subset else None,
                "negative_ratio": counts.get("negative", 0) / len(subset) if subset else None,
                "avg_rel_gain_pct": mean_or_none([r["rel_gain_pct"] for r in subset]),
                "median_rel_gain_pct": median_or_none([r["rel_gain_pct"] for r in subset]),
                "avg_stage1_mrr": mean_or_none([r["stage1_mrr"] for r in subset]),
                "avg_dep_per_rule": mean_or_none([r["dep_per_rule"] for r in subset]),
            }
        )
    return out


def build_type_weight_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    by_dataset: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
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
    for dataset, subset in sorted(by_dataset.items()):
        row = {"dataset": dataset, "aggregation": subset[0]["aggregation"]}
        for key in keys:
            row[key] = weighted_average(subset, key)
        out.append(row)
    return out


def plot_gain_vs_stage1(rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    datasets = sorted({r["dataset"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for idx, dataset in enumerate(datasets):
        subset = [r for r in rows if r["dataset"] == dataset]
        plt.scatter(
            [r["stage1_mrr"] for r in subset],
            [r["rel_gain_pct"] for r in subset],
            s=[max(20, math.sqrt(max(r["test_triple_count"], 1)) * 3) for r in subset],
            alpha=0.7,
            label=dataset,
            color=cmap(idx % 10),
        )
    plt.axhline(3.0, color="green", linestyle="--", linewidth=1)
    plt.axhline(-3.0, color="red", linestyle="--", linewidth=1)
    plt.axhline(0.0, color="black", linestyle=":", linewidth=1)
    plt.xlabel("Stage-1 MRR")
    plt.ylabel("Relative Gain on Final Test (%)")
    plt.title("Dependency Gain vs Stage-1 Strength")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gain_vs_dep_density(rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(8, 5))
    datasets = sorted({r["dataset"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for idx, dataset in enumerate(datasets):
        subset = [r for r in rows if r["dataset"] == dataset]
        xs = [math.log10(1.0 + float(r["dep_per_rule"])) for r in subset]
        ys = [r["rel_gain_pct"] for r in subset]
        plt.scatter(xs, ys, alpha=0.7, label=dataset, color=cmap(idx % 10))
    plt.axhline(3.0, color="green", linestyle="--", linewidth=1)
    plt.axhline(-3.0, color="red", linestyle="--", linewidth=1)
    plt.axhline(0.0, color="black", linestyle=":", linewidth=1)
    plt.xlabel("log10(1 + dependency per rule)")
    plt.ylabel("Relative Gain on Final Test (%)")
    plt.title("Dependency Gain vs Dependency Density")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bucket_summary(summary_rows: List[Dict[str, object]], title: str, out_path: Path) -> None:
    if plt is None:
        return
    labels = [row["bucket"] for row in summary_rows]
    positive = [float(row["positive_ratio"]) * 100 if row["positive_ratio"] is not None else 0.0 for row in summary_rows]
    negative = [float(row["negative_ratio"]) * 100 if row["negative_ratio"] is not None else 0.0 for row in summary_rows]
    avg_gain = [float(row["avg_rel_gain_pct"]) if row["avg_rel_gain_pct"] is not None else 0.0 for row in summary_rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, positive, width=w, label="positive ratio (%)", color="#5B8FF9")
    ax1.bar(x + w / 2, negative, width=w, label="negative ratio (%)", color="#E8684A")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Ratio (%)")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, avg_gain, color="#222222", marker="o", label="avg gain (%)")
    ax2.set_ylabel("Average Gain (%)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dataset_gain_mix(dataset_summary: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None:
        return
    labels = [row["dataset"] for row in dataset_summary]
    pos = [int(row["positive_relations"]) for row in dataset_summary]
    neu = [int(row["neutral_relations"]) for row in dataset_summary]
    neg = [int(row["negative_relations"]) for row in dataset_summary]
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
    if plt is None:
        return
    datasets = [row["dataset"] for row in type_rows if row["aggregation"] != "structural_none"]
    rule_keys = ["rule_weight_B", "rule_weight_U", "rule_weight_Uc", "rule_weight_Ud"]
    filtered_rows = [row for row in type_rows if row["aggregation"] != "structural_none"]
    if not filtered_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(datasets))
    width = 0.18
    for idx, key in enumerate(rule_keys):
        vals = [row.get(key) if row.get(key) is not None else np.nan for row in filtered_rows]
        axes[0].bar(x + (idx - 1.5) * width, vals, width=width, label=key.replace("rule_weight_", ""))
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=25, ha="right")
    axes[0].set_title("Average Learned Rule-Type Weights")
    axes[0].legend(fontsize=8)

    dep_keys = ["dep_weight_BB", "dep_weight_BU", "dep_weight_UU", "dep_weight_Uc_Ud"]
    width2 = 0.2
    for idx, key in enumerate(dep_keys):
        vals = [row.get(key) if row.get(key) is not None else np.nan for row in filtered_rows]
        axes[1].bar(x + (idx - 1.5) * width2, vals, width=width2, label=key.replace("dep_weight_", ""))
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=25, ha="right")
    axes[1].set_title("Average Learned Dependency-Type Weights")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = extract_relation_rows()
    rows.sort(key=lambda r: (r["dataset"], int(r["relation"])))

    relation_csv = REPORT_DIR / "relation_dependency_analysis.csv"
    relation_fields = list(rows[0].keys())
    write_csv(relation_csv, rows, relation_fields)

    group_rows = build_group_summary(rows)
    write_csv(REPORT_DIR / "relation_gain_group_summary.csv", group_rows, list(group_rows[0].keys()))

    dataset_rows = build_dataset_summary(rows)
    write_csv(REPORT_DIR / "relation_gain_dataset_summary.csv", dataset_rows, list(dataset_rows[0].keys()))

    stage1_bucket_rows = build_bucket_summary(rows, "stage1_bucket")
    write_csv(REPORT_DIR / "relation_gain_stage1_bucket_summary.csv", stage1_bucket_rows, list(stage1_bucket_rows[0].keys()))

    dep_bucket_rows = build_bucket_summary(rows, "dep_density_bucket")
    write_csv(REPORT_DIR / "relation_gain_dep_density_bucket_summary.csv", dep_bucket_rows, list(dep_bucket_rows[0].keys()))

    type_rows = build_type_weight_summary(rows)
    write_csv(REPORT_DIR / "relation_type_weight_summary.csv", type_rows, list(type_rows[0].keys()))

    plot_gain_vs_stage1(rows, REPORT_DIR / "plot_gain_vs_stage1.png")
    plot_gain_vs_dep_density(rows, REPORT_DIR / "plot_gain_vs_dep_density.png")
    plot_bucket_summary(stage1_bucket_rows, "Gain by Stage-1 Bucket", REPORT_DIR / "plot_stage1_bucket_summary.png")
    plot_bucket_summary(dep_bucket_rows, "Gain by Dependency Density Bucket", REPORT_DIR / "plot_dep_density_bucket_summary.png")
    plot_dataset_gain_mix(dataset_rows, REPORT_DIR / "plot_dataset_gain_mix.png")
    plot_type_weight_summary(type_rows, REPORT_DIR / "plot_type_weight_summary.png")

    print(f"Wrote relation analysis table to {relation_csv}")
    print(f"Rows: {len(rows)}")
    if plt is None:
        print("matplotlib not available; skipped plot generation")


if __name__ == "__main__":
    main()
