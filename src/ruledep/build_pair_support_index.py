#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from filter_dependency import load_split_targets, read_ids


def collect_positive_rule_sets(processed, split_targets, direction, output):
    for key, golds in split_targets.items():
        bucket = processed.get(key)
        if bucket is None:
            continue
        relation = int(key[1] if direction == "sp" else key[0])
        gold_set = set(int(value) for value in golds)
        for prediction, rule_ids in zip(bucket.get("candidates", []), bucket.get("rules", [])):
            if int(prediction) in gold_set:
                rules = sorted(set(int(rule_id) for rule_id in rule_ids))
                if len(rules) >= 2:
                    output[relation].append(rules)


def build_relation_index(rule_sets, min_support):
    rule_ids = np.array(sorted({rule_id for rules in rule_sets for rule_id in rules}), dtype=np.int64)
    if rule_ids.size < 2:
        return np.empty(0, dtype=np.uint64), 0

    local_by_global = {int(rule_id): index for index, rule_id in enumerate(rule_ids)}
    indptr = np.empty(len(rule_sets) + 1, dtype=np.int64)
    indptr[0] = 0
    indices = []
    for row, rules in enumerate(rule_sets, start=1):
        indices.extend(local_by_global[rule_id] for rule_id in rules)
        indptr[row] = len(indices)
    indices = np.asarray(indices, dtype=np.int32)
    matrix = csr_matrix(
        (np.ones(indices.size, dtype=np.int32), indices, indptr),
        shape=(len(rule_sets), rule_ids.size),
    )
    cooccurrence = (matrix.T @ matrix).tocoo()
    mask = (cooccurrence.row < cooccurrence.col) & (cooccurrence.data >= int(min_support))
    left = rule_ids[cooccurrence.row[mask]].astype(np.uint64)
    right = rule_ids[cooccurrence.col[mask]].astype(np.uint64)
    packed = np.sort((left << np.uint64(32)) | right)
    return packed, int(matrix.nnz)


def main():
    parser = argparse.ArgumentParser(description="Build reusable positive pair-support indexes.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--min_supp", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    start = perf_counter()
    dataset_dir = Path(args.data_root) / args.dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    relation_ids = read_ids(dataset_dir / "relation_ids.del")
    split_sp, split_po = load_split_targets(dataset_dir / "train.del")
    rule_sets_by_relation = defaultdict(list)

    for direction, split_targets in (("sp", split_sp), ("po", split_po)):
        processed_path = dataset_dir / "application" / f"processed_{direction}_train.pkl"
        with open(processed_path, "rb") as handle:
            processed = pickle.load(handle)
        collect_positive_rule_sets(processed, split_targets, direction, rule_sets_by_relation)
        del processed
        gc.collect()

    relation_rows = []
    total_pairs = 0
    for relation in range(len(relation_ids)):
        relation_start = perf_counter()
        rule_sets = rule_sets_by_relation.get(relation, [])
        packed, active_rule_occurrences = build_relation_index(rule_sets, args.min_supp)
        np.save(output_dir / f"relation_{relation}.npy", packed, allow_pickle=False)
        total_pairs += int(packed.size)
        row = {
            "relation": relation,
            "positive_instances": len(rule_sets),
            "active_rule_occurrences": active_rule_occurrences,
            "qualified_pairs": int(packed.size),
            "seconds": perf_counter() - relation_start,
        }
        relation_rows.append(row)
        print(
            f"relation={relation}/{len(relation_ids) - 1} instances={len(rule_sets)} "
            f"qualified_pairs={packed.size} seconds={row['seconds']:.2f}",
            flush=True,
        )
        del packed
        gc.collect()

    summary = {
        "dataset": args.dataset,
        "min_support": args.min_supp,
        "qualified_pairs": total_pairs,
        "seconds": perf_counter() - start,
        "relations": relation_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    Path(output_dir / "complete").touch()
    print(json.dumps({key: value for key, value in summary.items() if key != "relations"}), flush=True)


if __name__ == "__main__":
    main()
