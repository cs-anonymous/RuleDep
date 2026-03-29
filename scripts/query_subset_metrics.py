#!/home/sy/anaconda3/bin/python
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
import pickle
import sys
import tempfile

import torch


def build_loader_argv(config, loader_experiment_dir, device_override=None):
    argv = [
        "aggregation.py",
        "-d",
        str(config["dataset"]),
        "--data_root",
        str(config.get("data_root", "data")),
        "--device",
        str(device_override or config.get("device", "cuda")),
        "--model",
        str(config.get("model", "LinearAggregator")),
        "--batch_size",
        str(config.get("batch_size", 4096)),
        "--lr",
        str(config.get("lr", "0.01,0.005,0.001")),
        "--max_epoch",
        str(config.get("max_epoch", 60)),
        "--evaluate_every",
        str(config.get("evaluate_every", "4,2,1")),
        "--early_stopping",
        str(config.get("early_stopping", 3)),
        "--pos",
        str(config.get("pos", "auto_sqrt")),
        "--relation",
        str(config.get("relation", -1)),
        "--multiprocess",
        "0",
        "--rule_file",
        str(config["rule_file"]),
    ]
    if bool(config.get("shuffle_train", False)):
        argv.append("--shuffle_train")
    if bool(config.get("sign_constraint", True)):
        pass
    else:
        argv.append("--no_sign_constraint")
    if bool(config.get("sign_constraint_dependency", False)):
        argv.append("--sign_constraint_dependency")
    else:
        argv.append("--no_sign_constraint_dependency")
    if bool(config.get("synergy", False)):
        argv.append("--synergy")
    if bool(config.get("redundancy", False)):
        argv.append("--redundancy")
    if bool(config.get("init_dep_with_lift", False)):
        argv.append("--init_dep_with_lift")
    if bool(config.get("train_rule_in_dependency_stage", False)):
        argv.append("--train_rule_in_dependency_stage")
    if bool(config.get("dependency_pairs", True)):
        argv.append("--dependency_pairs")
    else:
        argv.append("--no_dependency_pairs")
    if bool(config.get("dependency_types", True)):
        argv.append("--dependency_types")
    else:
        argv.append("--no_dependency_types")

    optional_keys = [
        "dependency_lr",
        "dependency_evaluate_every",
        "dependency_max_epoch",
        "dependency_early_stopping",
        "dependency_min_epochs_before_stop",
        "dependency_checkpoint_selection",
        "dependency_accept_min_mrr_delta",
        "dependency_candidate_selection_epsilon",
        "dependency_inherited_lr_scale",
        "dependency_pairwise_finetune_epochs",
        "dependency_pairwise_topk_neg",
        "dependency_pairwise_query_batch_size",
        "dependency_pairwise_margin",
        "dependency_pairwise_pos_limit",
        "dependency_pair_overlap_normalization",
        "dependency_pair_position_chunk",
        "dependency_candidate_variants",
        "eval_key_batch_size",
        "dependency_chunk_size",
        "max_worker_dataloader",
    ]
    for key in optional_keys:
        if key in config and config[key] not in (None, ""):
            argv.extend([f"--{key}", str(config[key])])

    if bool(config.get("dependency_skip_init_checkpoint", False)):
        argv.append("--dependency_skip_init_checkpoint")
    if bool(config.get("dependency_pairwise_train_rules", False)):
        argv.append("--dependency_pairwise_train_rules")
    if bool(config.get("collect_train_hit_counts", True)):
        argv.append("--collect_train_hit_counts")
    else:
        argv.append("--no_collect_train_hit_counts")

    os.environ["EXPERIMENT_DIR"] = loader_experiment_dir
    return argv


def load_aggregation_module(repo_root, stage2_config, device_override=None):
    temp_root = os.path.join(repo_root, "tmp")
    os.makedirs(temp_root, exist_ok=True)
    loader_dir = tempfile.mkdtemp(prefix="query_subset_loader_", dir=temp_root)

    old_argv = sys.argv[:]
    old_exp = os.environ.get("EXPERIMENT_DIR")
    try:
        sys.argv = build_loader_argv(stage2_config, loader_dir, device_override=device_override)
        module_path = os.path.join(repo_root, "aggregation.py")
        spec = importlib.util.spec_from_file_location("aggregation_query_subset_loader", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules["__mp_main__"] = module
        return module, loader_dir
    finally:
        sys.argv = old_argv
        if old_exp is None:
            os.environ.pop("EXPERIMENT_DIR", None)
        else:
            os.environ["EXPERIMENT_DIR"] = old_exp


def load_mrr_pair(module, mrr_path):
    sys.modules["__mp_main__"] = module
    with open(mrr_path, "rb") as f:
        return pickle.load(f)


def load_metric_json(experiment_dir, relation):
    with open(os.path.join(experiment_dir, f"metric-{int(relation)}.json"), "r") as f:
        return json.load(f)


def resolve_selected_candidate(metric_json, stage2_config=None):
    model_selection = metric_json.get("model_selection", {}) or {}
    if str(model_selection.get("selected_stage", "rule_only")) != "dependency":
        return None

    stage2_metrics = (metric_json.get("train", {}) or {}).get("stage2_dependency_only", {}) or {}
    selected_label = model_selection.get("dependency_candidate_label") or stage2_metrics.get("candidate_label")
    if selected_label is None:
        # Older experiments did not persist explicit candidate metadata. Fall back to
        # the experiment-level dependency settings so we can reconstruct the stage-2 model.
        allowed_kinds = None
        if stage2_config is not None:
            allowed_kinds = []
            if bool(stage2_config.get("synergy", False)):
                allowed_kinds.append("synergy")
            if bool(stage2_config.get("redundancy", False)):
                allowed_kinds.append("redundancy")
            if len(allowed_kinds) == 0:
                allowed_kinds = None
            else:
                allowed_kinds = tuple(allowed_kinds)
        return {
            "label": "legacy_default",
            "allowed_kinds": allowed_kinds,
            "use_pairs": True if stage2_config is None else bool(stage2_config.get("dependency_pairs", True)),
            "use_types": True if stage2_config is None else bool(stage2_config.get("dependency_types", True)),
            "train_rules": False
            if stage2_config is None
            else bool(stage2_config.get("train_rule_in_dependency_stage", False)),
        }

    for candidate in stage2_metrics.get("candidate_variants", []) or []:
        if str(candidate.get("label")) == str(selected_label):
            return {
                "label": str(candidate.get("label")),
                "allowed_kinds": None
                if candidate.get("allowed_kinds") is None
                else tuple(str(kind) for kind in candidate.get("allowed_kinds")),
                "use_pairs": bool(candidate.get("use_pairs", True)),
                "use_types": bool(candidate.get("use_types", True)),
                "train_rules": bool(candidate.get("train_rules", True)),
            }

    candidate_allowed_kinds = stage2_metrics.get("candidate_allowed_kinds")
    return {
        "label": str(selected_label),
        "allowed_kinds": None if candidate_allowed_kinds is None else tuple(str(kind) for kind in candidate_allowed_kinds),
        "use_pairs": bool(stage2_metrics.get("candidate_use_pairs", True)),
        "use_types": bool(stage2_metrics.get("candidate_use_types", True)),
        "train_rules": bool(stage2_metrics.get("candidate_train_rules", True)),
    }


def build_model(module, relation, state_dict, candidate=None):
    if candidate is None:
        builder = module.build_rule_only_model_for_relation
        model = builder(int(relation)).to(module.args.device)
    else:
        builder = lambda rel: module.build_dependency_model_for_relation(rel, candidate=candidate)
        model = builder(int(relation)).to(module.args.device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "dependency_pair_keys_local_sorted",
        "dependency_pair_indices_sorted",
        "dependency_rule_degree_local",
        "dependency_pair_global_max_scale",
        "dependency_pair_global_sqrt_scale",
    }
    allowed_unexpected = {
        "dependency_pair_keys_local_sorted",
        "dependency_pair_indices_sorted",
        "dependency_rule_degree_local",
        "dependency_pair_global_max_scale",
        "dependency_pair_global_sqrt_scale",
    }
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    unexpected = [key for key in incompatible.unexpected_keys if key not in allowed_unexpected]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for relation={relation}, candidate={None if candidate is None else candidate.get('label')}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def ensure_query_tensors(module, processed_entry):
    if "candidates_tensor_gpu" not in processed_entry:
        processed_entry["candidates_tensor_gpu"] = torch.as_tensor(
            processed_entry.get("candidates", []), dtype=torch.long, device=module.EVAL_DEVICE
        )
    if "rules_padded_tensor" not in processed_entry:
        rule_lists = processed_entry.get("rules", [])
        if len(rule_lists) > 0:
            processed_entry["rules_padded_tensor"] = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor([torch.tensor(x) for x in rule_lists]),
                padding=module.PAD_TOK,
            ).long()
        else:
            processed_entry["rules_padded_tensor"] = torch.empty((0, 0), dtype=torch.long)
    if "rules_padded_tensor_gpu" not in processed_entry:
        processed_entry["rules_padded_tensor_gpu"] = processed_entry["rules_padded_tensor"].to(
            module.EVAL_DEVICE, non_blocking=True
        )
    return processed_entry["candidates_tensor_gpu"], processed_entry["rules_padded_tensor_gpu"]


def build_dependency_adjacency(module, relation):
    deps = module.dependency_map.get(int(relation), [])
    adjacency = defaultdict(list)
    for a, b, *_rest in deps:
        aa, bb = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        adjacency[aa].append(bb)
    return adjacency


def compute_query_activity(processed_entry, adjacency):
    candidate_count = int(len(processed_entry.get("candidates", [])))
    active_candidate_count = 0
    total_active_pairs = 0
    max_active_pairs = 0
    hub_degree_sum = 0

    for rule_ids in processed_entry.get("rules", []):
        if len(rule_ids) <= 1:
            continue
        active_rules = sorted(set(int(rid) for rid in rule_ids))
        active_rule_set = set(active_rules)
        pair_count = 0
        endpoint_degree = Counter()
        for a in active_rules:
            for b in adjacency.get(a, []):
                if b in active_rule_set:
                    pair_count += 1
                    endpoint_degree[a] += 1
                    endpoint_degree[b] += 1
        if pair_count > 0:
            active_candidate_count += 1
            total_active_pairs += pair_count
            max_active_pairs = max(max_active_pairs, pair_count)
            if len(endpoint_degree) > 0:
                hub_degree_sum += max(endpoint_degree.values())

    return {
        "candidate_count": candidate_count,
        "active_candidate_count": int(active_candidate_count),
        "total_active_pairs": int(total_active_pairs),
        "max_active_pairs": int(max_active_pairs),
        "hub_degree_sum": int(hub_degree_sum),
    }


def query_mrr_from_rank_tensor(rank_tensor):
    if int(rank_tensor.numel()) == 0:
        return 0.0
    return float((1.0 / rank_tensor).mean().item())


def gather_query_rows(module, relation, direction, stage1_model, stage2_model, adjacency):
    if direction == "o":
        processed = module.processed_sp_test
        split_to_targets = module.test_sp_to_o
        relation_key_name = "test_o"
    else:
        processed = module.processed_po_test
        split_to_targets = module.test_po_to_s
        relation_key_name = "test_s"

    keys = module.relation_keys[relation_key_name].get(int(relation), [])
    if len(keys) == 0:
        return []

    rows = []
    batch_items = []
    batch_meta = []
    key_batch_size = max(int(module.args.eval_key_batch_size), 1)

    for key in keys:
        processed_entry = processed.get(key)
        golds = split_to_targets.get(key)
        if processed_entry is None or golds is None:
            continue

        candidates_t, rules_t = ensure_query_tensors(module, processed_entry)
        golds_t = golds.long().to(module.EVAL_DEVICE, non_blocking=True)
        batch_items.append((golds_t, candidates_t, rules_t, None))
        batch_meta.append(
            {
                "relation": int(relation),
                "direction": str(direction),
                "key": [int(key[0]), int(key[1])],
                **compute_query_activity(processed_entry, adjacency),
            }
        )

        if len(batch_items) >= key_batch_size:
            stage1_results = module.rank_batch_group(stage1_model, batch_items)
            stage2_results = module.rank_batch_group(stage2_model, batch_items)
            for meta, res1, res2 in zip(batch_meta, stage1_results, stage2_results):
                rank1, _rank1_raw, n1 = res1
                rank2, _rank2_raw, n2 = res2
                meta["num_golds"] = int(n1)
                meta["stage1_query_mrr"] = query_mrr_from_rank_tensor(rank1)
                meta["stage2_query_mrr"] = query_mrr_from_rank_tensor(rank2)
                rows.append(meta)
            batch_items = []
            batch_meta = []

    if len(batch_items) > 0:
        stage1_results = module.rank_batch_group(stage1_model, batch_items)
        stage2_results = module.rank_batch_group(stage2_model, batch_items)
        for meta, res1, res2 in zip(batch_meta, stage1_results, stage2_results):
            rank1, _rank1_raw, n1 = res1
            rank2, _rank2_raw, n2 = res2
            meta["num_golds"] = int(n1)
            meta["stage1_query_mrr"] = query_mrr_from_rank_tensor(rank1)
            meta["stage2_query_mrr"] = query_mrr_from_rank_tensor(rank2)
            rows.append(meta)

    return rows


def summarize_subset(rows, selector_name, predicate):
    selected = [row for row in rows if predicate(row)]
    if len(selected) == 0:
        return None

    stage1 = sum(row["stage1_query_mrr"] for row in selected) / len(selected)
    stage2 = sum(row["stage2_query_mrr"] for row in selected) / len(selected)
    gain_abs = stage2 - stage1
    gain_rel = 0.0 if stage1 == 0.0 else gain_abs / stage1 * 100.0
    relation_counter = Counter(row["relation"] for row in selected)
    return {
        "selector": selector_name,
        "num_queries": int(len(selected)),
        "stage1_query_mrr": float(stage1),
        "stage2_query_mrr": float(stage2),
        "gain_abs": float(gain_abs),
        "gain_rel_percent": float(gain_rel),
        "top_relations": relation_counter.most_common(15),
    }


def search_thresholds(rows, feature_name, min_gain_percent):
    values = sorted({int(row[feature_name]) for row in rows if int(row[feature_name]) > 0}, reverse=True)
    summaries = []
    for value in values:
        summary = summarize_subset(rows, f"{feature_name}>={value}", lambda row, v=value: int(row[feature_name]) >= v)
        if summary is not None:
            summaries.append(summary)
    best = None
    for summary in summaries:
        if summary["gain_rel_percent"] < float(min_gain_percent):
            continue
        if best is None:
            best = summary
            continue
        if summary["num_queries"] > best["num_queries"]:
            best = summary
            continue
        if summary["num_queries"] == best["num_queries"] and summary["gain_rel_percent"] > best["gain_rel_percent"]:
            best = summary
    return best, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_experiment", required=True, help="Rule-only experiment directory")
    parser.add_argument("--stage2_experiment", required=True, help="Dependency experiment directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_gain_percent", type=float, default=5.0)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--rows_output",
        default="",
        help="Optional path to dump per-query rows as JSON for downstream subset search.",
    )
    args = parser.parse_args()

    stage1_experiment = os.path.abspath(args.stage1_experiment)
    stage2_experiment = os.path.abspath(args.stage2_experiment)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    with open(os.path.join(stage2_experiment, "config.json"), "r") as f:
        stage2_config = json.load(f)

    module, loader_dir = load_aggregation_module(repo_root, stage2_config, device_override=args.device)

    rows = []
    relation_files = sorted(
        int(name.split("-")[1].split(".")[0])
        for name in os.listdir(stage2_experiment)
        if name.startswith("mrr-") and name.endswith(".pkl")
    )

    for relation in relation_files:
        print(f"[query_subset] relation={relation}", flush=True)
        stage1_mrr_path = os.path.join(stage1_experiment, f"mrr-{relation}.pkl")
        stage2_mrr_path = os.path.join(stage2_experiment, f"mrr-{relation}.pkl")
        if not os.path.exists(stage1_mrr_path) or not os.path.exists(stage2_mrr_path):
            print(f"[query_subset] skip relation={relation} missing_mrr", flush=True)
            continue

        stage1_metric_json = load_metric_json(stage1_experiment, relation)
        stage2_metric_json = load_metric_json(stage2_experiment, relation)

        stage1_head, _stage1_tail = load_mrr_pair(module, stage1_mrr_path)
        stage2_head, _stage2_tail = load_mrr_pair(module, stage2_mrr_path)

        stage1_model = build_model(module, relation, stage1_head.nnm, candidate=None)
        stage2_candidate = resolve_selected_candidate(stage2_metric_json, stage2_config=stage2_config)
        stage2_model = build_model(module, relation, stage2_head.nnm, candidate=stage2_candidate)
        adjacency = build_dependency_adjacency(module, relation)

        rows.extend(gather_query_rows(module, relation, "o", stage1_model, stage2_model, adjacency))
        rows.extend(gather_query_rows(module, relation, "s", stage1_model, stage2_model, adjacency))
        print(f"[query_subset] relation={relation} rows_so_far={len(rows)}", flush=True)

        del stage1_model
        del stage2_model
        torch.cuda.empty_cache()

    total_queries = len(rows)
    overall = summarize_subset(rows, "all_queries", lambda row: True)
    active = summarize_subset(rows, "active_candidate_count>=1", lambda row: row["active_candidate_count"] >= 1)
    hard_active = summarize_subset(rows, "active_candidate_count>=2", lambda row: row["active_candidate_count"] >= 2)

    best_total_pairs, total_pairs_scan = search_thresholds(rows, "total_active_pairs", args.min_gain_percent)
    best_active_candidates, active_candidates_scan = search_thresholds(rows, "active_candidate_count", args.min_gain_percent)
    best_max_pairs, max_pairs_scan = search_thresholds(rows, "max_active_pairs", args.min_gain_percent)
    best_hub_degree, hub_degree_scan = search_thresholds(rows, "hub_degree_sum", args.min_gain_percent)

    summary = {
        "stage1_experiment": stage1_experiment,
        "stage2_experiment": stage2_experiment,
        "loader_experiment_dir": loader_dir,
        "total_queries": int(total_queries),
        "overall": None if overall is None else {
            **overall,
            "coverage": 1.0,
        },
        "active_candidate_count_ge_1": None if active is None else {
            **active,
            "coverage": float(active["num_queries"] / total_queries) if total_queries else 0.0,
        },
        "active_candidate_count_ge_2": None if hard_active is None else {
            **hard_active,
            "coverage": float(hard_active["num_queries"] / total_queries) if total_queries else 0.0,
        },
        "best_thresholds_for_gain": {
            "min_gain_percent": float(args.min_gain_percent),
            "total_active_pairs": None if best_total_pairs is None else {
                **best_total_pairs,
                "coverage": float(best_total_pairs["num_queries"] / total_queries),
            },
            "active_candidate_count": None if best_active_candidates is None else {
                **best_active_candidates,
                "coverage": float(best_active_candidates["num_queries"] / total_queries),
            },
            "max_active_pairs": None if best_max_pairs is None else {
                **best_max_pairs,
                "coverage": float(best_max_pairs["num_queries"] / total_queries),
            },
            "hub_degree_sum": None if best_hub_degree is None else {
                **best_hub_degree,
                "coverage": float(best_hub_degree["num_queries"] / total_queries),
            },
        },
        "scan_preview": {
            "total_active_pairs": [
                {**item, "coverage": float(item["num_queries"] / total_queries)} for item in total_pairs_scan[:20]
            ],
            "active_candidate_count": [
                {**item, "coverage": float(item["num_queries"] / total_queries)} for item in active_candidates_scan[:20]
            ],
            "max_active_pairs": [
                {**item, "coverage": float(item["num_queries"] / total_queries)} for item in max_pairs_scan[:20]
            ],
            "hub_degree_sum": [
                {**item, "coverage": float(item["num_queries"] / total_queries)} for item in hub_degree_scan[:20]
            ],
        },
    }

    output_path = args.output.strip()
    if output_path == "":
        output_path = os.path.join(stage2_experiment, "query_subset_metrics.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    rows_output_path = args.rows_output.strip()
    if rows_output_path != "":
        with open(rows_output_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
