#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path("/home/sy/RuleDep")
DATA_ROOT = ROOT / "data"
DEFAULT_DATASETS = ["KG20C", "codex-m", "WN18RR", "FB15k-237", "codex-l", "YAGO3-10"]


def parse_relation_map(dataset_dir: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for candidate in ["relation_ids.del", "relations.dict", "relation2id.txt"]:
        path = dataset_dir / candidate
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
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


def parse_rule_relation_index(rule_path: Path, relation_name_to_id: Dict[str, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns:
    - rule_to_relation: 1-based rule id -> relation id
    - total_rules_by_relation: relation id -> number of rules
    """
    rule_to_relation: Dict[int, int] = {}
    total_rules_by_relation: Dict[int, int] = defaultdict(int)

    if not rule_path.exists():
        return rule_to_relation, total_rules_by_relation

    idx = 0
    with rule_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            idx += 1
            rule = parts[3]
            head = rule.split("<=", 1)[0].strip()
            relation_name = head.split("(", 1)[0].strip()
            relation_id = relation_name_to_id.get(relation_name)
            if relation_id is None:
                continue
            rule_to_relation[idx] = relation_id
            total_rules_by_relation[relation_id] += 1

    return rule_to_relation, total_rules_by_relation


def parse_dependency_edges(
    dep_path: Path,
    rule_to_relation: Dict[int, int],
    dep_pairs_by_relation: Dict[int, Set[Tuple[int, int]]],
    connected_rules_by_relation: Dict[int, Set[int]],
) -> None:
    if not dep_path.exists():
        return

    with dep_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue
            ra = rule_to_relation.get(a)
            rb = rule_to_relation.get(b)
            if ra is None or rb is None or ra != rb:
                continue
            if a == b:
                continue
            x, y = (a, b) if a < b else (b, a)
            dep_pairs_by_relation[ra].add((x, y))
            connected_rules_by_relation[ra].add(x)
            connected_rules_by_relation[ra].add(y)


def build_adjacency(dep_pairs_by_relation: Dict[int, Set[Tuple[int, int]]]) -> Dict[int, Dict[int, Set[int]]]:
    adjacency_by_relation: Dict[int, Dict[int, Set[int]]] = {}
    for relation, pairs in dep_pairs_by_relation.items():
        adj: Dict[int, Set[int]] = defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
        adjacency_by_relation[relation] = adj
    return adjacency_by_relation


def count_dependencies_in_active_set(active_rules: Set[int], adjacency: Dict[int, Set[int]]) -> int:
    if not active_rules or not adjacency:
        return 0
    total = 0
    for rid in active_rules:
        neigh = adjacency.get(rid)
        if not neigh:
            continue
        total += len(neigh.intersection(active_rules))
    return total // 2


def iter_processed_split_queries(path: Path) -> Iterable[Tuple[Tuple[int, int], Dict[str, List[List[int]]]]]:
    import pickle

    with path.open("rb") as handle:
        data = pickle.load(handle)
    for key, payload in data.items():
        yield key, payload


def collect_query_rows(
    dataset: str,
    split: str,
    side: str,
    processed_path: Path,
    relation_map: Dict[int, str],
    connected_rules_by_relation: Dict[int, Set[int]],
    adjacency_by_relation: Dict[int, Dict[int, Set[int]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rel_idx = 1 if side == "sp" else 0

    for key, payload in iter_processed_split_queries(processed_path):
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        if not isinstance(payload, dict):
            continue
        rule_lists = payload.get("rules")
        if not isinstance(rule_lists, list):
            continue

        relation_id = int(key[rel_idx])
        relation_name = relation_map.get(relation_id, str(relation_id))

        active_rules: Set[int] = set()
        for rule_ids in rule_lists:
            if not isinstance(rule_ids, list):
                continue
            for rid in rule_ids:
                if isinstance(rid, int) and rid > 0:
                    active_rules.add(rid)

        connected_set = connected_rules_by_relation.get(relation_id, set())
        connected_active = active_rules.intersection(connected_set)
        dep_count = count_dependencies_in_active_set(connected_active, adjacency_by_relation.get(relation_id, {}))

        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "side": side,
                "query_key_0": int(key[0]),
                "query_key_1": int(key[1]),
                "query_id": f"{split}:{side}:{int(key[0])}|{int(key[1])}",
                "relation_id": relation_id,
                "relation_name": relation_name,
                "rules": len(active_rules),
                "connected_rules": len(connected_active),
                "dependencies": dep_count,
            }
        )
    return rows


def plot_relation_top20(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    if plt is None or df.empty:
        return

    top = df.sort_values("total_rules", ascending=False).head(20).copy()
    x = list(range(len(top)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar([i - 0.2 for i in x], top["total_rules"], width=0.4, label="Total Rules", color="#8ecae6")
    ax1.bar([i + 0.2 for i in x], top["connected_rules"], width=0.4, label="Connected Rules", color="#3a7ca5")
    ax1.set_ylabel("Number of Rules", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top["relation_name"], rotation=40, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, top["dependencies"], color="red", marker="o", label="Dependencies")
    ax2.set_ylabel("Number of Dependencies", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title(f"{dataset}: Rule and Dependency Distribution by Relation (Top 20)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_query_top20(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    if plt is None or df.empty:
        return

    top = df.sort_values("rules", ascending=False).head(20).copy()
    labels = top["query_id"].tolist()
    x = list(range(len(top)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar([i - 0.2 for i in x], top["rules"], width=0.4, label="Rules", color="#8ecae6")
    ax1.bar([i + 0.2 for i in x], top["connected_rules"], width=0.4, label="Connected Rules", color="#3a7ca5")
    ax1.set_ylabel("Number of Rules", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, top["dependencies"], color="red", marker="o", label="Dependencies")
    ax2.set_ylabel("Number of Dependencies", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title(f"{dataset}: Rule and Dependency Distribution by Query (Top 20 by Rule Count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_relation_rows(
    dataset: str,
    relation_map: Dict[int, str],
    total_rules_by_relation: Dict[int, int],
    connected_rules_by_relation: Dict[int, Set[int]],
    dep_pairs_by_relation: Dict[int, Set[Tuple[int, int]]],
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    relation_ids = set(total_rules_by_relation.keys()) | set(connected_rules_by_relation.keys()) | set(dep_pairs_by_relation.keys())
    if not query_df.empty:
        relation_ids |= set(query_df["relation_id"].astype(int).tolist())

    rows: List[Dict[str, object]] = []
    grouped = query_df.groupby("relation_id") if not query_df.empty else None

    for rel in sorted(relation_ids):
        query_count = 0
        avg_rules = 0.0
        avg_connected = 0.0
        avg_deps = 0.0
        if grouped is not None and rel in grouped.groups:
            sub = grouped.get_group(rel)
            query_count = int(len(sub))
            avg_rules = float(sub["rules"].mean())
            avg_connected = float(sub["connected_rules"].mean())
            avg_deps = float(sub["dependencies"].mean())

        rows.append(
            {
                "dataset": dataset,
                "relation_id": int(rel),
                "relation_name": relation_map.get(int(rel), str(rel)),
                "total_rules": int(total_rules_by_relation.get(int(rel), 0)),
                "connected_rules": int(len(connected_rules_by_relation.get(int(rel), set()))),
                "dependencies": int(len(dep_pairs_by_relation.get(int(rel), set()))),
                "query_count": query_count,
                "avg_rules_per_query": avg_rules,
                "avg_connected_rules_per_query": avg_connected,
                "avg_dependencies_per_query": avg_deps,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 0407 coverage stats by relation and query for all datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names. Default: KG20C,codex-m,WN18RR,FB15k-237,codex-l,YAGO3-10",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test",
        help="Comma-separated splits for query statistics. Default: test",
    )
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    out_dirs = [ROOT / "reports" / "0407" / "coverage", ROOT / "report" / "0407" / "coverage"]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_relation_frames: List[pd.DataFrame] = []
    all_query_frames: List[pd.DataFrame] = []

    for dataset in datasets:
        dataset_dir = DATA_ROOT / dataset
        rules_dir = dataset_dir / "rules"
        application_dir = dataset_dir / "application"
        if not dataset_dir.exists() or not rules_dir.exists() or not application_dir.exists():
            print(f"[SKIP] {dataset}: missing dataset/rules/application directory")
            continue

        print(f"[INFO] Processing dataset: {dataset}")
        relation_map = parse_relation_map(dataset_dir)
        relation_name_to_id = {name: rid for rid, name in relation_map.items()}

        rule_path = rules_dir / "rule.txt"
        rule_to_relation, total_rules_by_relation = parse_rule_relation_index(rule_path, relation_name_to_id)

        dep_pairs_by_relation: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        connected_rules_by_relation: Dict[int, Set[int]] = defaultdict(set)
        parse_dependency_edges(rules_dir / "synergy_filtered.txt", rule_to_relation, dep_pairs_by_relation, connected_rules_by_relation)
        parse_dependency_edges(rules_dir / "redundancy_filtered.txt", rule_to_relation, dep_pairs_by_relation, connected_rules_by_relation)
        adjacency_by_relation = build_adjacency(dep_pairs_by_relation)

        query_rows: List[Dict[str, object]] = []
        for split in splits:
            for side in ["sp", "po"]:
                processed_path = application_dir / f"processed_{side}_{split}.pkl"
                if not processed_path.exists():
                    continue
                print(f"  [INFO] Query stats from {processed_path.name}")
                query_rows.extend(
                    collect_query_rows(
                        dataset=dataset,
                        split=split,
                        side=side,
                        processed_path=processed_path,
                        relation_map=relation_map,
                        connected_rules_by_relation=connected_rules_by_relation,
                        adjacency_by_relation=adjacency_by_relation,
                    )
                )

        query_df = pd.DataFrame(query_rows)
        relation_df = summarize_relation_rows(
            dataset=dataset,
            relation_map=relation_map,
            total_rules_by_relation=total_rules_by_relation,
            connected_rules_by_relation=connected_rules_by_relation,
            dep_pairs_by_relation=dep_pairs_by_relation,
            query_df=query_df,
        )

        all_relation_frames.append(relation_df)
        all_query_frames.append(query_df)

        for out_dir in out_dirs:
            relation_df.sort_values(["total_rules", "dependencies"], ascending=[False, False]).to_csv(
                out_dir / f"relation_stats_{dataset}.csv", index=False
            )
            query_df.to_csv(out_dir / f"query_stats_{dataset}.csv", index=False)
            plot_relation_top20(relation_df, dataset, out_dir / f"plot_relation_top20_{dataset}.png")
            plot_query_top20(query_df, dataset, out_dir / f"plot_query_top20_{dataset}.png")

    relation_all = pd.concat(all_relation_frames, ignore_index=True) if all_relation_frames else pd.DataFrame()
    query_all = pd.concat(all_query_frames, ignore_index=True) if all_query_frames else pd.DataFrame()

    if not relation_all.empty:
        relation_all.sort_values(["dataset", "total_rules"], ascending=[True, False], inplace=True)
    if not query_all.empty:
        query_all.sort_values(["dataset", "split", "side", "rules"], ascending=[True, True, True, False], inplace=True)

    for out_dir in out_dirs:
        relation_all.to_csv(out_dir / "relation_stats_all_datasets.csv", index=False)
        query_all.to_csv(out_dir / "query_stats_all_datasets.csv", index=False)

    print("[DONE] Coverage report artifacts generated.")
    for out_dir in out_dirs:
        print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
