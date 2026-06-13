#!/usr/bin/env python3
import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import re
from collections import defaultdict

try:
    import ctypes
except Exception:
    ctypes = None


WORKER_ACTIVE_RULE_SETS_BY_RELATION = None
WORKER_NEGATIVE_RULE_SETS_BY_RELATION = None
WORKER_MIN_SUPP = 0
DEFAULT_FILTER_JOBS = max(1, (os.cpu_count() or 1) * 3 // 4)


def _init_filter_worker(active_rule_sets_by_relation, negative_rule_sets_by_relation, min_supp):
    global WORKER_ACTIVE_RULE_SETS_BY_RELATION
    global WORKER_NEGATIVE_RULE_SETS_BY_RELATION
    global WORKER_MIN_SUPP
    WORKER_ACTIVE_RULE_SETS_BY_RELATION = active_rule_sets_by_relation
    WORKER_NEGATIVE_RULE_SETS_BY_RELATION = negative_rule_sets_by_relation
    WORKER_MIN_SUPP = int(min_supp)

def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def _split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def extract_head_relation(rule_body: str):
    head = rule_body.split("<=")[0].strip()
    match = re.match(r"^\s*([^\(]+)\(", head)
    if not match:
        return None
    return match.group(1).strip()


def parse_rule_file_metadata(rule_file, relation_ids):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_relation_by_id = {}
    relation_rule_count_by_id = defaultdict(int)

    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is not None:
                rule_relation_by_id[int(line_no)] = int(rel_id)
                relation_rule_count_by_id[int(rel_id)] += 1
    return rule_relation_by_id, dict(relation_rule_count_by_id)


def collect_labeled_active_rule_sets_by_relation(split_to_targets, processed, direction="o"):
    positive_rule_sets_by_relation = defaultdict(list)
    negative_rule_sets_by_relation = defaultdict(list)
    for key, golds in split_to_targets.items():
        if direction == "o":
            _e, relation = key
        else:
            relation, _e = key

        if key not in processed:
            continue

        if hasattr(golds, "tolist"):
            gold_iter = golds.tolist()
        else:
            gold_iter = golds
        gold_set = set(int(x) for x in gold_iter)
        candidates = processed[key].get("candidates", [])
        rules_per_candidate = processed[key].get("rules", [])
        for prediction, rule_ids in zip(candidates, rules_per_candidate):
            active_rule_set = set(int(rid) for rid in rule_ids)
            if int(prediction) not in gold_set:
                negative_rule_sets_by_relation[int(relation)].append(active_rule_set)
                continue
            positive_rule_sets_by_relation[int(relation)].append(active_rule_set)
    return positive_rule_sets_by_relation, negative_rule_sets_by_relation


def prefilter_candidates_by_support(active_rule_sets, candidates, min_count):
    min_count = int(min_count)
    if min_count <= 0:
        return list(range(len(candidates)))
    if len(candidates) == 0 or len(active_rule_sets) == 0:
        return []

    adj = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        a, b = int(candidate[0]), int(candidate[1])
        adj[a].append((b, idx))

    counts = [0] * len(candidates)
    keep = [False] * len(candidates)
    remaining = len(candidates)

    for rs in active_rule_sets:
        if remaining <= 0:
            break
        if len(rs) < 2:
            continue
        for a in rs:
            if a not in adj:
                continue
            for b, idx in adj[a]:
                if keep[idx]:
                    continue
                if b in rs:
                    counts[idx] += 1
                    if counts[idx] >= min_count:
                        keep[idx] = True
                        remaining -= 1

    return [i for i, k in enumerate(keep) if k]


def count_pair_occurrences(active_rule_sets, candidates):
    if len(candidates) == 0 or len(active_rule_sets) == 0:
        return [0] * len(candidates)

    adj = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        a, b = int(candidate[0]), int(candidate[1])
        adj[a].append((b, idx))

    counts = [0] * len(candidates)
    for rs in active_rule_sets:
        if len(rs) < 2:
            continue
        for a in rs:
            if a not in adj:
                continue
            for b, idx in adj[a]:
                if b in rs:
                    counts[idx] += 1
    return counts


def load_split_targets(split_path):
    split_sp_to_o = defaultdict(list)
    split_po_to_s = defaultdict(list)
    with open(split_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            s, p, o = int(parts[0]), int(parts[1]), int(parts[2])
            split_sp_to_o[(s, p)].append(o)
            split_po_to_s[(p, o)].append(s)
    return dict(split_sp_to_o), dict(split_po_to_s)


def load_applied_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_processed_from_applied(applied_rules, entity_id_to_idx, relation_id_to_idx):
    processed_sp = {}
    processed_po = {}

    tail_applied = applied_rules.get("tail", {})
    for rel_raw, source_map in tail_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for s_raw, target_map in source_map.items():
            if s_raw not in entity_id_to_idx:
                continue
            s_idx = entity_id_to_idx[s_raw]
            key = (s_idx, p_idx)
            bucket = processed_sp.setdefault(key, {"candidates": [], "rules": []})

            for o_raw, rule_ids in target_map.items():
                if o_raw not in entity_id_to_idx:
                    continue
                o_idx = entity_id_to_idx[o_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(o_idx)
                bucket["rules"].append(ids)

    head_applied = applied_rules.get("head", {})
    for rel_raw, source_map in head_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for o_raw, target_map in source_map.items():
            if o_raw not in entity_id_to_idx:
                continue
            o_idx = entity_id_to_idx[o_raw]
            key = (p_idx, o_idx)
            bucket = processed_po.setdefault(key, {"candidates": [], "rules": []})

            for s_raw, rule_ids in target_map.items():
                if s_raw not in entity_id_to_idx:
                    continue
                s_idx = entity_id_to_idx[s_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(s_idx)
                bucket["rules"].append(ids)

    return processed_sp, processed_po


def load_processed_split(application_dir, entity_ids, relation_ids, split_name):
    sp_path = os.path.join(application_dir, f"processed_sp_{split_name}.pkl")
    po_path = os.path.join(application_dir, f"processed_po_{split_name}.pkl")
    if os.path.exists(sp_path) and os.path.exists(po_path):
        return pickle.load(open(sp_path, "rb")), pickle.load(open(po_path, "rb"))

    applied_path = os.path.join(application_dir, f"applied_rules_{split_name}.json")
    if not os.path.exists(applied_path):
        raise FileNotFoundError(
            f"Missing processed {split_name} explanations ({sp_path}, {po_path}) and fallback source {applied_path}"
        )

    entity_id_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}
    relation_id_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}
    applied_rules = load_applied_rules(applied_path)
    return build_processed_from_applied(applied_rules, entity_id_to_idx, relation_id_to_idx)


def build_labeled_rule_sets_by_relation(split_to_targets, processed_sp, processed_po):
    positive_rule_sets_by_relation = defaultdict(list)
    negative_rule_sets_by_relation = defaultdict(list)

    pos_sp, neg_sp = collect_labeled_active_rule_sets_by_relation(split_to_targets[0], processed_sp, "o")
    pos_po, neg_po = collect_labeled_active_rule_sets_by_relation(split_to_targets[1], processed_po, "s")

    for relation, sets_ in pos_sp.items():
        positive_rule_sets_by_relation[int(relation)].extend(sets_)
    for relation, sets_ in pos_po.items():
        positive_rule_sets_by_relation[int(relation)].extend(sets_)
    for relation, sets_ in neg_sp.items():
        negative_rule_sets_by_relation[int(relation)].extend(sets_)
    for relation, sets_ in neg_po.items():
        negative_rule_sets_by_relation[int(relation)].extend(sets_)

    return positive_rule_sets_by_relation, negative_rule_sets_by_relation


def parse_raw_dependency_file(path, rule_relation_by_id):
    pairs_by_relation = defaultdict(list)
    seen_pairs = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            try:
                lift = float(parts[2])
                id1 = int(parts[3])
                id2 = int(parts[4])
            except Exception:
                continue

            rel1 = rule_relation_by_id.get(id1)
            rel2 = rule_relation_by_id.get(id2)
            if rel1 is None or rel2 is None or rel1 != rel2:
                continue

            a, b = (id1, id2) if id1 <= id2 else (id2, id1)
            if (a, b) in seen_pairs[rel1]:
                continue
            seen_pairs[rel1].add((a, b))
            pairs_by_relation[int(rel1)].append((a, b, float(lift)))

    return dict(pairs_by_relation)


def interleave_rankings(primary, secondary, limit):
    selected = []
    seen = set()
    primary_idx = 0
    secondary_idx = 0

    while len(selected) < limit and (primary_idx < len(primary) or secondary_idx < len(secondary)):
        if primary_idx < len(primary):
            item = primary[primary_idx]
            primary_idx += 1
            key = (int(item[0]), int(item[1]))
            if key not in seen:
                selected.append(item)
                seen.add(key)
                if len(selected) >= limit:
                    break
        if secondary_idx < len(secondary):
            item = secondary[secondary_idx]
            secondary_idx += 1
            key = (int(item[0]), int(item[1]))
            if key not in seen:
                selected.append(item)
                seen.add(key)
                if len(selected) >= limit:
                    break

    return selected


def rank_candidates_for_mode(candidates, ranking_mode, limit):
    ranking_mode = str(ranking_mode).lower()
    if ranking_mode == "lift":
        ranked = sorted(candidates, key=lambda x: (-abs(float(x[2])), int(x[0]), int(x[1])))
    elif ranking_mode == "ratio":
        ranked = sorted(candidates, key=lambda x: (-float(x[5]), -abs(float(x[2])), int(x[0]), int(x[1])))
    elif ranking_mode == "mix":
        ranked_lift = sorted(candidates, key=lambda x: (-abs(float(x[2])), int(x[0]), int(x[1])))
        ranked_ratio = sorted(candidates, key=lambda x: (-float(x[5]), -abs(float(x[2])), int(x[0]), int(x[1])))
        ranked = interleave_rankings(ranked_lift, ranked_ratio, limit if limit > 0 else len(candidates))
    else:
        raise ValueError(f"Unknown ranking_mode: {ranking_mode}")
    if limit > 0:
        return ranked[:limit]
    return []


def rank_and_limit_kept_pairs_by_relation(
    kept_pairs_by_relation, relation_rule_count_by_id, dep_per_rule_multiplier, ranking_mode
):
    final_pairs = []
    kept_before_limit = 0
    dep_per_rule_multiplier = max(int(dep_per_rule_multiplier), 0)

    for relation, candidates in sorted(kept_pairs_by_relation.items(), key=lambda x: x[0]):
        limit = max(int(relation_rule_count_by_id.get(int(relation), 0)), 0) * dep_per_rule_multiplier
        ranked = rank_candidates_for_mode(candidates, ranking_mode, limit)
        kept_before_limit += len(ranked)
        final_pairs.extend((int(a), int(b), float(lift)) for a, b, lift, *_rest in ranked)

    return final_pairs, kept_before_limit


def default_variant_specs():
    specs = []
    for ranking_mode in ("lift", "ratio", "mix"):
        for multiplier in (1, 2, 4):
            specs.append((ranking_mode, multiplier, f"_{ranking_mode}_k{multiplier}"))
    return specs


def parse_variant_specs(args):
    if args.variant_sweep == "default_9":
        return default_variant_specs()
    output_suffix = str(args.output_suffix)
    if output_suffix and not output_suffix.startswith("_"):
        output_suffix = "_" + output_suffix
    return [(str(args.ranking_mode), int(args.dep_per_rule_multiplier), output_suffix)]


def write_filtered_dependency_file(output_path, kept_pairs):
    with open(output_path, "w", encoding="utf-8") as f:
        for a, b, lift in kept_pairs:
            f.write(f"{int(a)}\t{int(b)}\t{float(lift):.10g}\n")


def release_process_memory():
    gc.collect()
    if ctypes is None:
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except Exception:
        pass


def filter_relation_candidates(task):
    relation, chunk_id, candidates = task
    keep_idx = set(range(len(candidates)))

    if WORKER_MIN_SUPP > 0:
        active_sets = WORKER_ACTIVE_RULE_SETS_BY_RELATION.get(int(relation), [])
        keep_idx &= set(prefilter_candidates_by_support(active_sets, candidates, WORKER_MIN_SUPP))

    keep_idx = sorted(keep_idx)
    kept_candidates = [candidates[i] for i in keep_idx]
    positive_sets = WORKER_ACTIVE_RULE_SETS_BY_RELATION.get(int(relation), [])
    negative_sets = WORKER_NEGATIVE_RULE_SETS_BY_RELATION.get(int(relation), [])
    pos_counts = count_pair_occurrences(positive_sets, kept_candidates)
    neg_counts = count_pair_occurrences(negative_sets, kept_candidates)

    kept_pairs = []
    for (a, b, lift), pos_count, neg_count in zip(kept_candidates, pos_counts, neg_counts):
        ratio_score = math.log((float(pos_count) + 1.0) / (float(neg_count) + 1.0))
        kept_pairs.append((int(a), int(b), float(lift), int(pos_count), int(neg_count), float(ratio_score)))
    return int(relation), int(chunk_id), int(len(candidates)), kept_pairs


def build_filter_tasks(pairs_by_relation, jobs, chunk_candidates):
    relation_items = sorted(pairs_by_relation.items(), key=lambda x: x[0])
    tasks = []
    for relation, candidates in relation_items:
        candidate_count = len(candidates)
        if candidate_count == 0:
            continue

        # Small relations stay as a single task to avoid extra overhead.
        if jobs <= 1 or candidate_count <= chunk_candidates:
            tasks.append((int(relation), 0, candidates))
            continue

        for chunk_id, start in enumerate(range(0, candidate_count, chunk_candidates)):
            end = min(start + chunk_candidates, candidate_count)
            tasks.append((int(relation), int(chunk_id), candidates[start:end]))
    return tasks


def filter_dependency_file(
    input_path,
    positive_rule_sets_by_relation,
    negative_rule_sets_by_relation,
    rule_relation_by_id,
    relation_rule_count_by_id,
    min_train,
    variant_specs,
    jobs=1,
    progress_every=10,
    chunk_candidates=10000,
):
    pairs_by_relation = parse_raw_dependency_file(input_path, rule_relation_by_id)

    kept_pairs_by_relation = defaultdict(list)
    raw_total = sum(len(v) for v in pairs_by_relation.values())
    relation_items = sorted(pairs_by_relation.items(), key=lambda x: x[0])
    num_relations = len(relation_items)
    jobs = max(int(jobs), 1)
    progress_every = max(int(progress_every), 1)
    chunk_candidates = max(int(chunk_candidates), 1)
    tasks = build_filter_tasks(pairs_by_relation, jobs=jobs, chunk_candidates=chunk_candidates)
    num_tasks = len(tasks)

    print(
        f"{os.path.basename(input_path)}: filtering {raw_total} dependencies across {num_relations} relations "
        f"using {num_tasks} task(s) (jobs={jobs}, chunk_candidates={chunk_candidates}, "
        f"min_train={int(min_train)}, variants={len(variant_specs)})"
    )

    processed_tasks = 0
    processed_candidates = 0

    if jobs == 1:
        _init_filter_worker(positive_rule_sets_by_relation, negative_rule_sets_by_relation, min_train)
        for item in tasks:
            _relation, _chunk_id, candidate_count, kept_pairs_rel = filter_relation_candidates(item)
            kept_pairs_by_relation[int(_relation)].extend(kept_pairs_rel)
            processed_tasks += 1
            processed_candidates += candidate_count
            if processed_tasks % progress_every == 0 or processed_tasks == num_tasks:
                print(
                    f"{os.path.basename(input_path)}: progress {processed_tasks}/{num_tasks} tasks, "
                    f"{processed_candidates}/{raw_total} candidates examined, "
                    f"kept={sum(len(v) for v in kept_pairs_by_relation.values())}"
                )
    else:
        with mp.Pool(
            processes=jobs,
            initializer=_init_filter_worker,
            initargs=(positive_rule_sets_by_relation, negative_rule_sets_by_relation, min_train),
        ) as pool:
            for _relation, _chunk_id, candidate_count, kept_pairs_rel in pool.imap_unordered(
                filter_relation_candidates, tasks, chunksize=1
            ):
                kept_pairs_by_relation[int(_relation)].extend(kept_pairs_rel)
                processed_tasks += 1
                processed_candidates += candidate_count
                if processed_tasks % progress_every == 0 or processed_tasks == num_tasks:
                    print(
                        f"{os.path.basename(input_path)}: progress {processed_tasks}/{num_tasks} tasks, "
                        f"{processed_candidates}/{raw_total} candidates examined, "
                        f"kept={sum(len(v) for v in kept_pairs_by_relation.values())}"
                    )

    base_path = os.path.splitext(input_path)[0]
    for ranking_mode, dep_per_rule_multiplier, output_suffix in variant_specs:
        kept_pairs, kept_before_limit = rank_and_limit_kept_pairs_by_relation(
            kept_pairs_by_relation,
            relation_rule_count_by_id,
            dep_per_rule_multiplier,
            ranking_mode,
        )
        output_path = base_path + f"_filtered{output_suffix}.txt"
        write_filtered_dependency_file(output_path, kept_pairs)
        print(
            f"{os.path.basename(input_path)}[{ranking_mode},k={dep_per_rule_multiplier}]: "
            f"kept {len(kept_pairs)} / {raw_total} dependencies "
            f"(before relation cap: {kept_before_limit}) -> {output_path}"
        )
        del kept_pairs

    del kept_pairs_by_relation
    del tasks
    del pairs_by_relation
    release_process_memory()

def main():
    parser = argparse.ArgumentParser(description="Filter dependency files by split support.")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rule_file", default="")
    parser.add_argument("--synergy_file", default="")
    parser.add_argument("--redundancy_file", default="")
    parser.add_argument("--target_split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--min_supp", type=int, default=5)
    parser.add_argument("--min_train", dest="min_supp", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dep_per_rule_multiplier", type=int, default=2, help="Keep at most this many dependencies per rule for each relation after ranking by |lift|.")
    parser.add_argument("--ranking_mode", choices=["lift", "ratio", "mix"], default="lift")
    parser.add_argument("--output_suffix", default="", help="Suffix appended to filtered dependency filenames before .txt, e.g. _ratio_k2.")
    parser.add_argument("--variant_sweep", choices=["none", "default_9"], default="none", help="When set, generate multiple ranking/k variants from a single filtering pass.")
    parser.add_argument("--jobs", type=int, default=DEFAULT_FILTER_JOBS, help="Number of worker processes for relation-level filtering.")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress after every N filtering tasks per dependency file.",
    )
    parser.add_argument(
        "--chunk_candidates",
        type=int,
        default=10000,
        help="Split very large relations into chunks of this many candidate pairs before dispatching work.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    rules_dir = os.path.join(dataset_dir, "rules")
    application_dir = os.path.join(dataset_dir, "application")

    rule_file = args.rule_file or os.path.join(rules_dir, "rule.txt")
    synergy_file = args.synergy_file or os.path.join(rules_dir, "synergy.txt")
    redundancy_file = args.redundancy_file or os.path.join(rules_dir, "redundancy.txt")

    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    rule_relation_by_id, relation_rule_count_by_id = parse_rule_file_metadata(rule_file, relation_ids)

    positive_rule_sets_by_relation = defaultdict(list)
    negative_rule_sets_by_relation = defaultdict(list)
    if int(args.min_supp) > 0:
        entity_ids = read_ids(os.path.join(dataset_dir, "entity_ids.del"))
        split_sp_to_o, split_po_to_s = load_split_targets(os.path.join(dataset_dir, f"{args.target_split}.del"))
        processed_sp_split, processed_po_split = load_processed_split(application_dir, entity_ids, relation_ids, args.target_split)
        positive_rule_sets_by_relation, negative_rule_sets_by_relation = build_labeled_rule_sets_by_relation(
            (split_sp_to_o, split_po_to_s),
            processed_sp_split,
            processed_po_split,
        )
    if args.dataset in ["hetionet", "YAGO3-10"]:
        args.jobs = min(args.jobs, 18)  # Avoid OOM

    variant_specs = parse_variant_specs(args)

    if os.path.exists(synergy_file):
        filter_dependency_file(
            synergy_file,
            positive_rule_sets_by_relation,
            negative_rule_sets_by_relation,
            rule_relation_by_id,
            relation_rule_count_by_id,
            min_train=args.min_supp,
            variant_specs=variant_specs,
            jobs=args.jobs,
            progress_every=args.progress_every,
            chunk_candidates=args.chunk_candidates,
        )
        release_process_memory()

    if os.path.exists(redundancy_file):
        filter_dependency_file(
            redundancy_file,
            positive_rule_sets_by_relation,
            negative_rule_sets_by_relation,
            rule_relation_by_id,
            relation_rule_count_by_id,
            min_train=args.min_supp,
            variant_specs=variant_specs,
            jobs=args.jobs,
            progress_every=args.progress_every,
            chunk_candidates=args.chunk_candidates,
        )


if __name__ == "__main__":
    main()
