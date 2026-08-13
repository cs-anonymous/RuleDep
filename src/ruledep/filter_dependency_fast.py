#!/usr/bin/env python3
import argparse
import heapq
import os
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from filter_dependency import parse_rule_file_metadata, read_ids


def is_supported(index, keys):
    if index.size == 0 or keys.size == 0:
        return np.zeros(keys.size, dtype=bool)
    positions = np.searchsorted(index, keys)
    valid = positions < index.size
    result = np.zeros(keys.size, dtype=bool)
    result[valid] = index[positions[valid]] == keys[valid]
    return result


def update_heaps(rows, support_indexes, relation_by_rule, limits, heaps):
    if not rows:
        return 0
    left = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
    right = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
    lifts = np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows))
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    relations = relation_by_rule[low]
    kept = 0

    for relation in np.unique(relations):
        if relation < 0 or relation not in support_indexes:
            continue
        selected = np.flatnonzero(relations == relation)
        keys = (low[selected].astype(np.uint64) << np.uint64(32)) | high[selected].astype(np.uint64)
        qualified = selected[is_supported(support_indexes[relation], keys)]
        heap = heaps[relation]
        limit = limits.get(int(relation), 0)
        for index in qualified:
            a = int(low[index])
            b = int(high[index])
            lift = float(lifts[index])
            item = (abs(lift), -a, -b, a, b, lift)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:3] > heap[0][:3]:
                heapq.heapreplace(heap, item)
            kept += 1
    return kept


def filter_file(input_path, output_path, support_indexes, relation_by_rule, limits, batch_size):
    start = perf_counter()
    heaps = defaultdict(list)
    rows = []
    parsed = 0
    qualified = 0

    with open(input_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 5:
                parts = re.split(r"\s+", raw_line.strip())
            if len(parts) < 5:
                continue
            rows.append((int(parts[3]), int(parts[4]), float(parts[2])))
            parsed += 1
            if len(rows) >= batch_size:
                qualified += update_heaps(rows, support_indexes, relation_by_rule, limits, heaps)
                rows.clear()
                if parsed % (batch_size * 10) == 0:
                    print(f"{Path(input_path).name}: parsed={parsed} qualified={qualified}", flush=True)
        qualified += update_heaps(rows, support_indexes, relation_by_rule, limits, heaps)

    selected = []
    for heap in heaps.values():
        selected.extend((item[3], item[4], item[5]) for item in heap)
    selected.sort(key=lambda item: (-abs(item[2]), item[0], item[1]))
    with open(output_path, "w", encoding="utf-8") as handle:
        for a, b, lift in selected:
            handle.write(f"{a}\t{b}\t{lift:.10g}\n")
    print(
        f"{Path(input_path).name}: parsed={parsed} support_qualified={qualified} "
        f"selected={len(selected)} seconds={perf_counter() - start:.2f} -> {output_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Stream dependency filtering using a reusable support index.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rule_file", default="")
    parser.add_argument("--support_index", required=True)
    parser.add_argument("--synergy_file", required=True)
    parser.add_argument("--redundancy_file", required=True)
    parser.add_argument("--dep_per_rule_multiplier", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=250000)
    args = parser.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    rule_file = args.rule_file or dataset_dir / "rules" / "rule.txt"
    relation_ids = read_ids(dataset_dir / "relation_ids.del")
    rule_relation_by_id, relation_rule_counts = parse_rule_file_metadata(rule_file, relation_ids)
    max_rule_id = max(rule_relation_by_id, default=0)
    relation_by_rule = np.full(max_rule_id + 1, -1, dtype=np.int32)
    for rule_id, relation in rule_relation_by_id.items():
        relation_by_rule[rule_id] = relation
    limits = {
        relation: count * max(int(args.dep_per_rule_multiplier), 0)
        for relation, count in relation_rule_counts.items()
    }
    support_dir = Path(args.support_index)
    support_indexes = {
        relation: np.load(support_dir / f"relation_{relation}.npy", mmap_mode="r")
        for relation in range(len(relation_ids))
    }

    for input_path in (args.synergy_file, args.redundancy_file):
        output_path = os.path.splitext(input_path)[0] + "_filtered.txt"
        filter_file(
            input_path,
            output_path,
            support_indexes,
            relation_by_rule,
            limits,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
