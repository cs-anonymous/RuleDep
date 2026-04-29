#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean


DATA_ROOT = "/home/sy/RuleDep/data"
DEFAULT_EXAMPLE_ROOT = "/home/sy/RuleDep/RuleDepDemo/frontend/public/example"


def read_entity_count(dataset: str) -> int:
    path = os.path.join(DATA_ROOT, dataset, "entity_ids.del")
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_test_gold_index(dataset: str):
    tail_golds = defaultdict(set)
    head_golds = defaultdict(set)
    path = os.path.join(DATA_ROOT, dataset, "test.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head, relation, tail = parts[:3]
            tail_golds[(head, relation)].add(tail)
            head_golds[(relation, tail)].add(head)
    return tail_golds, head_golds


def read_relation_names(dataset: str) -> list[str]:
    path = os.path.join(DATA_ROOT, dataset, "relation_ids.del")
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def max_conf(candidate: dict) -> float:
    if "maxConf" in candidate:
        return float(candidate.get("maxConf") or 0.0)
    max_safe = float(candidate.get("max", 0.0) or 0.0)
    return 1.0 - math.exp(-max_safe) if max_safe < 50 else 1.0


def official_score(candidate: dict, stage_key: str) -> float:
    official_key = f"{stage_key}Official"
    if official_key in candidate:
        return float(candidate.get(official_key) or 0.0)
    return sigmoid(float(candidate.get(stage_key, 0.0) or 0.0)) * max_conf(candidate)


def official_rank(candidates: list[dict], score_key: str, gt_entity: str, gt_entities: set[str], num_entities: int) -> float:
    scores = {str(c["name"]): official_score(c, score_key) for c in candidates}
    non_gold_candidates = set(scores).difference(gt_entities)
    zero_count = max(int(num_entities) - len(non_gold_candidates), 0)
    gold_score = scores.get(gt_entity, 0.0)

    higher = sum(1 for entity in non_gold_candidates if scores[entity] > gold_score)
    equal = sum(1 for entity in non_gold_candidates if scores[entity] == gold_score)
    if 0.0 > gold_score:
        higher += zero_count - 1
    elif 0.0 == gold_score:
        equal += zero_count
    else:
        equal += 1
    return higher + (equal + 1.0) / 2.0


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def topk_stats(values: list[float], prefix: str, ks=(1, 3, 5, 10)) -> dict[str, float]:
    ordered = sorted((float(v) for v in values), reverse=True)
    out = {}
    for k in ks:
        top = ordered[:k]
        out[f"{prefix}_top{k}_sum"] = sum(top)
        out[f"{prefix}_top{k}_mean"] = safe_mean(top)
    out[f"{prefix}_max"] = ordered[0] if ordered else 0.0
    out[f"{prefix}_mean"] = safe_mean(ordered)
    return out


def load_dependency_weights(dataset: str):
    weights = {}
    for kind, filename in [("synergy", "synergy_filtered.txt"), ("redundancy", "redundancy_filtered.txt")]:
        path = os.path.join(DATA_ROOT, dataset, "rules", filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    a, b = int(parts[0]), int(parts[1])
                    weight = float(parts[2])
                except ValueError:
                    continue
                weights[tuple(sorted((a, b)))] = (kind, weight)
    return weights


def build_query_features(cands: list[dict], q: dict, dep_weights: dict[tuple[int, int], tuple[str, float]]) -> dict[str, float]:
    rule_ids = set()
    dep_pairs = set()
    rule_counts = []
    pos_counts = []
    neg_counts = []
    dep_scores = []
    stage1_scores = []
    stage2_scores = []
    rule_values = []
    synergy_values = []
    redundancy_values = []
    candidates_with_dep = 0
    candidates_with_rules = 0

    for cand in cands:
        rules = [int(r) for r in cand.get("rules", [])]
        if rules:
            candidates_with_rules += 1
        rule_ids.update(rules)
        rule_counts.append(int(cand.get("scoredRuleCount", len(rules)) or 0))
        pos = int(cand.get("positiveDep", 0) or 0)
        neg = int(cand.get("negativeDep", 0) or 0)
        pos_counts.append(pos)
        neg_counts.append(neg)
        if pos + neg > 0:
            candidates_with_dep += 1
        dep_scores.append(float(cand.get("dependencyScore", 0.0) or 0.0))
        stage1_scores.append(official_score(cand, "stage1"))
        stage2_scores.append(official_score(cand, "stage2"))
        rule_values.extend(float(v) for v in cand.get("maxplus", []) if isinstance(v, (int, float)))

        for pair in cand.get("displayedDependencyPairs", []) or []:
            if len(pair) < 2:
                continue
            try:
                key = tuple(sorted((int(pair[0]), int(pair[1]))))
            except (TypeError, ValueError):
                continue
            dep_pairs.add(key)

    for pair in dep_pairs:
        kind_weight = dep_weights.get(pair)
        if not kind_weight:
            continue
        kind, weight = kind_weight
        if kind == "synergy":
            synergy_values.append(abs(float(weight)))
        elif kind == "redundancy":
            redundancy_values.append(abs(float(weight)))

    top_stage1 = sorted(stage1_scores, reverse=True)
    top_stage2 = sorted(stage2_scores, reverse=True)
    stage1_top_margin = top_stage1[0] - top_stage1[1] if len(top_stage1) > 1 else 0.0
    stage2_top_margin = top_stage2[0] - top_stage2[1] if len(top_stage2) > 1 else 0.0

    num_rules = len(rule_ids)
    num_dependencies = len(dep_pairs)
    dep_density = (2.0 * num_dependencies / (num_rules * (num_rules - 1))) if num_rules > 1 else 0.0

    num_pos_dep = len(synergy_values)
    num_neg_dep = len(redundancy_values)
    dep_count_total = num_pos_dep + num_neg_dep
    pos_dep_ratio = (num_pos_dep / dep_count_total) if dep_count_total > 0 else 0.0
    neg_dep_ratio = (num_neg_dep / dep_count_total) if dep_count_total > 0 else 0.0

    pos_mass = sum(synergy_values)
    neg_mass = sum(redundancy_values)
    net_dep_mass = pos_mass - neg_mass
    abs_dep_mass = pos_mass + neg_mass

    rule_mass = sum(float(v) for v in rule_values)
    rule_dominance_ratio = (max(rule_values) / (rule_mass + 1e-12)) if rule_values else 0.0
    weak_rule_score = 1.0 - (max(rule_values) if rule_values else 0.0)

    dep_rule_ratio = abs_dep_mass / (rule_mass + 1e-12)
    syn_rule_ratio = pos_mass / (rule_mass + 1e-12)
    red_rule_ratio = neg_mass / (rule_mass + 1e-12)

    if stage1_scores:
        smax = max(stage1_scores)
        exps = [math.exp(s - smax) for s in stage1_scores]
        z = sum(exps)
        probs = [(e / z) if z > 0 else 0.0 for e in exps]
        s1_entropy = -sum((p * math.log(p)) for p in probs if p > 0)
    else:
        s1_entropy = 0.0
    effective_candidates = math.exp(s1_entropy)
    s1_top1 = top_stage1[0] if top_stage1 else 0.0
    s1_margin = stage1_top_margin
    s1_norm_margin = s1_margin / (s1_top1 + 1e-12)

    features = {
        "num_candidates": len(cands),
        "num_candidate_rule_edges": sum(rule_counts),
        "num_rule_nodes": num_rules,
        "num_rules": num_rules,
        "num_dependency_edges": num_dependencies,
        "num_dependencies": num_dependencies,
        "dep_density": dep_density,
        "query_num_nodes": int(q.get("numNodes", 0) or 0),
        "query_num_edges": int(q.get("numEdges", 0) or 0),
        "avg_rules_per_candidate": safe_mean(rule_counts),
        "max_rules_per_candidate": max(rule_counts) if rule_counts else 0,
        "candidate_rule_coverage": candidates_with_rules / len(cands) if cands else 0.0,
        "candidate_dep_coverage": candidates_with_dep / len(cands) if cands else 0.0,
        "dep_candidate_ratio": candidates_with_dep / len(cands) if cands else 0.0,
        "sum_positive_dep": sum(pos_counts),
        "sum_negative_dep": sum(neg_counts),
        "avg_positive_dep": safe_mean(pos_counts),
        "avg_negative_dep": safe_mean(neg_counts),
        "max_positive_dep": max(pos_counts) if pos_counts else 0,
        "max_negative_dep": max(neg_counts) if neg_counts else 0,
        "avg_candidate_dep_score": safe_mean(dep_scores),
        "max_candidate_dep_score": max(dep_scores) if dep_scores else 0.0,
        "avg_stage1_score": safe_mean(stage1_scores),
        "max_stage1_score": max(stage1_scores) if stage1_scores else 0.0,
        "stage1_top_margin": stage1_top_margin,
        "s1_top1": s1_top1,
        "s1_margin": s1_margin,
        "s1_norm_margin": s1_norm_margin,
        "s1_entropy": s1_entropy,
        "effective_candidates": effective_candidates,
        "avg_stage2_score": safe_mean(stage2_scores),
        "max_stage2_score": max(stage2_scores) if stage2_scores else 0.0,
        "stage2_top_margin": stage2_top_margin,
        "unique_synergy_edges": num_pos_dep,
        "unique_redundancy_edges": num_neg_dep,
        "num_pos_dep": num_pos_dep,
        "num_neg_dep": num_neg_dep,
        "pos_dep_ratio": pos_dep_ratio,
        "neg_dep_ratio": neg_dep_ratio,
        "pos_mass": pos_mass,
        "neg_mass": neg_mass,
        "net_dep_mass": net_dep_mass,
        "abs_dep_mass": abs_dep_mass,
        "rule_mass": rule_mass,
        "top1_rule_weight": max(rule_values) if rule_values else 0.0,
        "topk_rule_weight": safe_mean(sorted(rule_values, reverse=True)[:3]) if rule_values else 0.0,
        "rule_dominance_ratio": rule_dominance_ratio,
        "weak_rule_score": weak_rule_score,
        "dep_rule_ratio": dep_rule_ratio,
        "syn_rule_ratio": syn_rule_ratio,
        "red_rule_ratio": red_rule_ratio,
    }
    features.update(topk_stats(rule_values, "rule_weight"))
    features.update(topk_stats(synergy_values, "synergy_weight"))
    features.update(topk_stats(redundancy_values, "redundancy_weight"))
    features["topk_synergy"] = features["synergy_weight_top3_mean"]
    features["topk_synergy_sum"] = features["synergy_weight_top3_sum"]
    features["topk_redundancy"] = features["redundancy_weight_top3_mean"]
    return features


def load_official_relation_deltas(rows: list[dict]) -> dict[tuple[str, str, str], float]:
    needed = {(row["dataset"], row["experiment"]) for row in rows}
    out = {}
    for dataset, experiment in needed:
        relation_names = read_relation_names(dataset)
        exp_dir = os.path.join(DATA_ROOT, dataset, "aggregation", experiment)
        for metric_path in glob.glob(os.path.join(exp_dir, "metric-*.json")):
            metric = json.load(open(metric_path, encoding="utf-8"))
            relation_id = int(metric["relation"])
            if relation_id >= len(relation_names):
                continue
            stage1 = metric.get("test_after_stage1") or {}
            stage2 = metric.get("test_after_stage2") or stage1
            if "mrr" not in stage1 or "mrr" not in stage2:
                continue
            out[(dataset, experiment, relation_names[relation_id])] = float(stage2["mrr"]) - float(stage1["mrr"])
    return out


def calibrate_relation_delta(rows: list[dict]) -> list[dict]:
    official_delta = load_official_relation_deltas(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["experiment"], row["relation"])].append(row)

    for key, group in grouped.items():
        target = official_delta.get(key)
        raw_mean_delta = safe_mean([float(row["raw_delta_rr"]) for row in group])
        offset = (target - raw_mean_delta) if target is not None else 0.0
        for row in group:
            row["official_relation_delta_mrr"] = target if target is not None else ""
            row["calibration_offset"] = offset
            row["delta_rr"] = float(row["raw_delta_rr"]) + offset
            row["rr_stage1"] = float(row["raw_rr_stage1"])
            row["rr_stage2"] = float(row["rr_stage1"]) + float(row["delta_rr"])
            row["gain_pt"] = (float(row["rr_stage2"]) / float(row["rr_stage1"]) - 1.0) if float(row["rr_stage1"]) > 0 else ""
    return rows


def compute_rows(example_root: str) -> list[dict]:
    rows = []
    dataset_cache = {}
    dep_weight_cache = {}
    feature_cache = {}

    for dataset_dir in sorted(glob.glob(os.path.join(example_root, "*"))):
        if not os.path.isdir(dataset_dir):
            continue
        dataset = os.path.basename(dataset_dir)
        if dataset not in dataset_cache:
            dataset_cache[dataset] = (read_entity_count(dataset), *read_test_gold_index(dataset))
        if dataset not in dep_weight_cache:
            dep_weight_cache[dataset] = load_dependency_weights(dataset)
        num_entities, tail_golds, head_golds = dataset_cache[dataset]
        dep_weights = dep_weight_cache[dataset]

        for exp_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
            if not os.path.isdir(exp_dir):
                continue
            experiment = os.path.basename(exp_dir)
            rels_path = os.path.join(exp_dir, "relations.json")
            if not os.path.exists(rels_path):
                continue
            relations = json.load(open(rels_path, encoding="utf-8")).get("relations", [])

            for relation in relations:
                rel_dir = os.path.join(exp_dir, relation.replace("/", "_"))
                qpath = os.path.join(rel_dir, "queries.json")
                if not os.path.exists(qpath):
                    continue
                qitems = json.load(open(qpath, encoding="utf-8")).get("queries", [])
                for q in qitems:
                    fpath = os.path.join(rel_dir, q["filename"])
                    if not os.path.exists(fpath):
                        continue
                    feature_key = (dataset, fpath)
                    cands = None

                    direction = q.get("direction", "")
                    entity = q.get("entity", "")
                    if direction == "head":
                        all_gt = set(q.get("allGtEntities") or q.get("gtEntities") or head_golds.get((relation, entity), set()))
                    else:
                        all_gt = set(q.get("allGtEntities") or q.get("gtEntities") or tail_golds.get((entity, relation), set()))
                    target_gt = q.get("targetGtEntity") or (q.get("gtEntities") or [None])[0]
                    if not target_gt:
                        continue
                    if target_gt not in all_gt:
                        all_gt.add(target_gt)

                    if q.get("gtRankStage1") and q.get("gtRankStage2"):
                        r1 = float(q.get("gtRankStage1"))
                        r2 = float(q.get("gtRankStage2"))
                    else:
                        cands = json.load(open(fpath, encoding="utf-8"))
                        if not cands:
                            continue
                        r1 = float(official_rank(cands, "stage1", target_gt, all_gt, num_entities))
                        r2 = float(official_rank(cands, "stage2", target_gt, all_gt, num_entities))
                    rr1 = (1.0 / r1) if r1 > 0 else 0.0
                    rr2 = (1.0 / r2) if r2 > 0 else 0.0

                    row = {
                        "dataset": dataset,
                        "experiment": experiment,
                        "relation": relation,
                        "direction": direction,
                        "query": q.get("query", ""),
                        "filename": q.get("filename", ""),
                        "case_id": q.get("caseId", f"{q.get('filename', '')}::{target_gt}"),
                        "target_gt_entity": target_gt,
                        "raw_rr_stage1": rr1,
                        "raw_rr_stage2": rr2,
                        "raw_delta_rr": rr2 - rr1,
                    }
                    if feature_key not in feature_cache:
                        if cands is None:
                            cands = json.load(open(fpath, encoding="utf-8"))
                        if not cands:
                            continue
                        feature_cache[feature_key] = build_query_features(cands, q, dep_weights)
                    row.update(feature_cache[feature_key])
                    rows.append(row)
    return calibrate_relation_delta(rows)


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile_ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [1.0 for _ in values]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        rank = ((i + j - 1) / 2.0) / (n - 1)
        for pos in range(i, j):
            ranks[order[pos]] = rank
        i = j
    return ranks


def add_composite_features(rows: list[dict]) -> list[dict]:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    high_features = [
        "num_candidates",
        "num_dependency_edges",
        "candidate_dep_coverage",
        "sum_positive_dep",
        "sum_negative_dep",
        "unique_synergy_edges",
        "unique_redundancy_edges",
        "avg_rules_per_candidate",
        "rule_weight_mean",
    ]
    low_features = [
        "max_stage1_score",
        "stage1_top_margin",
        "rule_weight_max",
    ]

    for dataset_rows in by_dataset.values():
        ranks = {}
        for feature in high_features + low_features:
            ranks[feature] = percentile_ranks([float(row.get(feature, 0.0)) for row in dataset_rows])

        for idx, row in enumerate(dataset_rows):
            dep_parts = [
                ranks["num_dependency_edges"][idx],
                ranks["candidate_dep_coverage"][idx],
                ranks["sum_positive_dep"][idx],
                ranks["sum_negative_dep"][idx],
                ranks["unique_synergy_edges"][idx],
                ranks["unique_redundancy_edges"][idx],
            ]
            complexity_parts = [
                ranks["num_candidates"][idx],
                ranks["avg_rules_per_candidate"][idx],
                ranks["rule_weight_mean"][idx],
            ]
            uncertainty_parts = [
                1.0 - ranks["max_stage1_score"][idx],
                1.0 - ranks["stage1_top_margin"][idx],
            ]
            low_rule_conf_parts = [1.0 - ranks["rule_weight_max"][idx]]

            row["combo_dependency_activity"] = safe_mean(dep_parts)
            row["combo_candidate_complexity"] = safe_mean(complexity_parts)
            row["combo_stage1_uncertainty"] = safe_mean(uncertainty_parts)
            row["combo_low_rule_confidence"] = safe_mean(low_rule_conf_parts)
            row["combo_dep_activity_x_uncertainty"] = row["combo_dependency_activity"] * row["combo_stage1_uncertainty"]
            row["combo_complex_dep_low_conf"] = safe_mean([
                row["combo_candidate_complexity"],
                row["combo_dependency_activity"],
                row["combo_low_rule_confidence"],
            ])
    return rows


def numeric_feature_names(rows: list[dict]) -> list[str]:
    excluded = {
        "raw_rr_stage1",
        "raw_rr_stage2",
        "raw_delta_rr",
        "rr_stage1",
        "rr_stage2",
        "delta_rr",
        "gain_pt",
        "official_relation_delta_mrr",
        "calibration_offset",
    }
    names = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            names.append(key)
    return names


def build_threshold_curves(rows: list[dict], coverage_steps: list[float]) -> list[dict]:
    curves = []
    features = numeric_feature_names(rows)
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset, dataset_rows in sorted(by_dataset.items()):
        n_total = len(dataset_rows)
        for feature in features:
            for direction in ("desc", "asc"):
                ordered = sorted(dataset_rows, key=lambda r: float(r.get(feature, 0.0)), reverse=(direction == "desc"))
                for coverage in coverage_steps:
                    n = max(1, min(n_total, int(round(n_total * coverage))))
                    subset = ordered[:n]
                    mrr_s1 = safe_mean([float(r["rr_stage1"]) for r in subset])
                    mrr_s2 = safe_mean([float(r["rr_stage2"]) for r in subset])
                    gain_pt = (mrr_s2 / mrr_s1 - 1.0) if mrr_s1 > 0 else 0.0
                    curves.append({
                        "dataset": dataset,
                        "feature": feature,
                        "sort_direction": direction,
                        "coverage": coverage,
                        "n": n,
                        "threshold": float(subset[-1].get(feature, 0.0)),
                        "mrr_stage1": mrr_s1,
                        "mrr_stage2": mrr_s2,
                        "delta_mrr": mrr_s2 - mrr_s1,
                        "gain_pt": gain_pt,
                    })
    return curves


def build_best_summary(curves: list[dict]) -> list[dict]:
    grouped = {}
    for row in curves:
        key = (row["dataset"], row["feature"], row["sort_direction"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (dataset, feature, direction), rows in grouped.items():
        full = min(rows, key=lambda r: abs(float(r["coverage"]) - 1.0))
        eligible = [r for r in rows if float(r["coverage"]) >= 0.2]
        best = max(eligible, key=lambda r: float(r["gain_pt"])) if eligible else max(rows, key=lambda r: float(r["gain_pt"]))
        summaries.append({
            "dataset": dataset,
            "feature": feature,
            "sort_direction": direction,
            "full_gain_pt": full["gain_pt"],
            "best_coverage_ge_20": best["coverage"],
            "best_n": best["n"],
            "best_threshold": best["threshold"],
            "best_mrr_stage1": best["mrr_stage1"],
            "best_mrr_stage2": best["mrr_stage2"],
            "best_delta_mrr": best["delta_mrr"],
            "best_gain_pt": best["gain_pt"],
            "gain_lift_vs_full": float(best["gain_pt"]) - float(full["gain_pt"]),
        })
    return sorted(summaries, key=lambda r: (r["dataset"], -float(r["best_gain_pt"])))


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_feature_curves(curves: list[dict], out_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = os.path.join(out_dir, "feature_plots")
    os.makedirs(plot_root, exist_ok=True)

    by_key = defaultdict(list)
    for row in curves:
        by_key[(row["feature"], row["sort_direction"])].append(row)

    def find_coverage_key(keys: list[float], target: float, tol: float = 1e-9) -> float | None:
        if not keys:
            return None
        best = min(keys, key=lambda k: abs(k - target))
        return best if abs(best - target) <= tol else None

    for (feature, direction), rows in sorted(by_key.items()):
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        by_dataset = defaultdict(list)
        for row in rows:
            by_dataset[row["dataset"]].append(row)

        macro_points = defaultdict(list)
        for row in rows:
            macro_points[float(row["coverage"])].append(float(row["gain_pt"]))
        macro_curves = []
        for coverage in sorted(macro_points):
            xs = coverage * 100.0
            ys = safe_mean(macro_points[coverage]) * 100.0
            macro_curves.append((xs, ys))

        dataset_names = sorted(by_dataset.keys())
        cmap = plt.get_cmap("tab10")
        color_map = {name: cmap(i % 10) for i, name in enumerate(dataset_names)}

        for dataset, dataset_rows in sorted(by_dataset.items()):
            points = sorted(dataset_rows, key=lambda r: float(r["coverage"]))
            xs = [float(r["coverage"]) * 100.0 for r in points]
            ys = [float(r["gain_pt"]) * 100.0 for r in points]
            ax.plot(xs, ys, color=color_map[dataset], alpha=0.45, linewidth=1.1, label=dataset)

        macro_xs = [x for x, _ in macro_curves]
        macro_ys = [y for _, y in macro_curves]
        ax.plot(
            macro_xs,
            macro_ys,
            color="#d9480f",
            marker="o",
            markersize=4.6,
            linewidth=2.8,
            label="macro ave gain_pt",
            zorder=5,
        )

        cov_keys = list(macro_points.keys())
        for target_cov_frac in (0.10, 0.20):
            matched = find_coverage_key(cov_keys, target_cov_frac)
            if matched is not None:
                x = matched * 100.0
                y = safe_mean(macro_points[matched]) * 100.0
                ax.scatter([x], [y], color="#d9480f", s=38, zorder=6)
                ax.annotate(
                    f"{int(round(target_cov_frac * 100))}%: {y:.2f}%",
                    xy=(x, y),
                    xytext=(8, 10 if target_cov_frac == 0.10 else -16),
                    textcoords="offset points",
                    fontsize=8,
                    color="#8a2f06",
                    arrowprops=dict(arrowstyle="->", color="#8a2f06", lw=0.8),
                )

        if macro_ys:
            y_min = min(macro_ys)
            y_max = max(macro_ys)
            y_span = max(y_max - y_min, 0.25)
            pad = max(0.20 * y_span, 0.15)
            ax.set_ylim(y_min - pad, y_max + pad)

        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"Data coverage (%) ranked by {feature} ({direction})")
        ax.set_ylabel("gain_pt (%)")
        ax.set_title(f"{feature} ({direction})")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_root, f"{slugify(feature)}__{direction}.png"))
        plt.close(fig)


def write_markdown(path: str, summary_rows: list[dict], curves: list[dict] | None = None, topn: int = 8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    by_dataset = defaultdict(list)
    for row in summary_rows:
        by_dataset[row["dataset"]].append(row)

    lines = [
        "# Official-aligned Query Subset Feature Search",
        "",
        "Metric: `gain_pt = MRR_stage2 / MRR_stage1 - 1`, using official-aligned per-test-triple RR.",
        "",
        "Feature policy: all ranking features are query/candidate-set aggregates. GT-specific fields are not used for threshold selection.",
        "",
        "Plots: `feature_plots/<feature>__desc.png` keeps the highest feature values first; `feature_plots/<feature>__asc.png` keeps the lowest feature values first. Each plot contains one curve per dataset.",
        "",
    ]

    if curves:
        target_coverages = [0.20, 0.10, 0.05]
        lines.extend([
            "## Top gain_pt at fixed coverage (20% / 10% / 5%)",
            "",
            "### Macro average (mean across datasets)",
            "",
            "| coverage | feature | order | macro gain_pt |",
            "| ---: | --- | --- | ---: |",
        ])

        for cov in target_coverages:
            cov_rows = [r for r in curves if abs(float(r["coverage"]) - cov) < 1e-9]
            grouped = defaultdict(list)
            for r in cov_rows:
                grouped[(r["feature"], r["sort_direction"])].append(float(r["gain_pt"]))
            if not grouped:
                continue
            best_feature, gains = max(grouped.items(), key=lambda kv: safe_mean(kv[1]))
            lines.append(
                f"| {int(cov * 100)}% | {best_feature[0]} | {best_feature[1]} | {safe_mean(gains):.4f} |"
            )

        lines.extend([
            "",
            "### Per-dataset top feature at fixed coverage",
            "",
            "| dataset | coverage | feature | order | gain_pt | threshold |",
            "| --- | ---: | --- | --- | ---: | ---: |",
        ])

        datasets = sorted({str(r["dataset"]) for r in curves})
        for dataset in datasets:
            for cov in target_coverages:
                cand = [
                    r for r in curves
                    if str(r["dataset"]) == dataset and abs(float(r["coverage"]) - cov) < 1e-9
                ]
                if not cand:
                    continue
                best = max(cand, key=lambda r: float(r["gain_pt"]))
                lines.append(
                    "| {dataset} | {cov}% | {feature} | {sort_direction} | {gain_pt:.4f} | {threshold:.6g} |".format(
                        dataset=dataset,
                        cov=int(cov * 100),
                        feature=best["feature"],
                        sort_direction=best["sort_direction"],
                        gain_pt=float(best["gain_pt"]),
                        threshold=float(best["threshold"]),
                    )
                )
        lines.append("")

    for dataset, rows in sorted(by_dataset.items()):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| feature | order | full gain | best coverage >=20% | best gain | delta MRR | threshold |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows[:topn]:
            lines.append(
                "| {feature} | {sort_direction} | {full_gain_pt:.4f} | {best_coverage_ge_20:.2f} | "
                "{best_gain_pt:.4f} | {best_delta_mrr:.6f} | {best_threshold:.6g} |".format(**row)
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example-root", default=DEFAULT_EXAMPLE_ROOT)
    parser.add_argument("--out-dir", default="/home/sy/RuleDep/reports/0421/official_query_subset")
    parser.add_argument("--data-only", action="store_true", help="only rebuild feature CSV, skip curves and plots")
    parser.add_argument("--plots-only", action="store_true", help="reuse existing feature CSV and only rebuild curves/plots")
    args = parser.parse_args()

    features_csv = os.path.join(args.out_dir, "official_query_triple_features.csv")
    if args.plots_only:
        with open(features_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key, value in list(row.items()):
                try:
                    if value != "":
                        row[key] = float(value)
                except Exception:
                    pass
    else:
        rows = compute_rows(args.example_root)
        rows = add_composite_features(rows)
        write_csv(features_csv, rows)

    if args.data_only:
        print(f"rows={len(rows)}")
        print(f"features={len(numeric_feature_names(rows))}")
        return

    coverage_steps = [round(x / 100.0, 2) for x in range(100, 0, -1)]
    curves = build_threshold_curves(rows, coverage_steps)
    write_csv(os.path.join(args.out_dir, "feature_threshold_curves.csv"), curves)
    plot_feature_curves(curves, args.out_dir)

    summary = build_best_summary(curves)
    write_csv(os.path.join(args.out_dir, "best_feature_threshold_summary.csv"), summary)
    write_markdown(os.path.join(args.out_dir, "README.md"), summary, curves)

    print(f"rows={len(rows)}")
    print(f"features={len(numeric_feature_names(rows))}")
    print(f"curves={len(curves)}")


if __name__ == "__main__":
    main()
