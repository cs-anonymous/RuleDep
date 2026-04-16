import json
import math
import os
import re
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
import yaml

import script.generate_case_study_examples as casegen

ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "0407"
CASE_CSV = REPORT_DIR / "relation_case_study_examples.csv"
OUT_DIR = REPORT_DIR / "case"


class FlowList(list):
    """YAML helper: keep short scalar arrays compact as [a, b, c]."""


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


yaml.SafeDumper.add_representer(FlowList, _flow_list_representer)


def compact_lists(obj):
    if isinstance(obj, dict):
        return {k: compact_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [compact_lists(v) for v in obj]
        if all(not isinstance(v, (dict, list)) for v in items):
            return FlowList(items)
        return items
    return obj


def safe_id(s):
    s = str(s).strip().replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.>-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] or "case"


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    if x is None:
        return None
    return float(x)


def round6(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return round(x, 6)


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


def normalize_digits_key(value):
    s = str(value)
    if s.isdigit():
        return str(int(s))
    return s


def label_from_info(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("label") or value.get("id")
    if isinstance(value, str):
        return value
    return None


def load_label_maps(dataset):
    data_dir = ROOT / "data" / dataset
    entity_labels = {}
    relation_labels = {}

    # KG20C.
    info_path = data_dir / "all_entity_info.txt"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            header = next(f, "")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    entity_labels[parts[0]] = parts[1] or parts[0]

    # FB15k/YAGO/WN style metadata.
    for path in [data_dir / "entity.json", data_dir / "mid.json", data_dir / "entities.json"]:
        obj = load_json(path)
        if isinstance(obj, dict):
            for key, value in obj.items():
                label = label_from_info(value)
                if label:
                    entity_labels[str(key)] = str(label)
                    entity_labels[normalize_digits_key(key)] = str(label)

    # Dictionaries are a fallback; JSON metadata above is usually more readable.
    dict_path = data_dir / "entities.dict"
    if dict_path.exists():
        with open(dict_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1] not in entity_labels:
                    entity_labels[parts[1]] = parts[1]

    # Hetionet IDs are already structured; strip namespace into a readable fallback.
    ids_path = data_dir / "entity_ids.del"
    if ids_path.exists():
        with open(ids_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1] not in entity_labels:
                    entity_labels[parts[1]] = parts[1].split("::")[-1]

    for path in [data_dir / "relation_name.json", data_dir / "relations.json", data_dir / "relation.json"]:
        obj = load_json(path)
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    label = value.get("label") or value.get("name") or value.get("id")
                else:
                    label = value
                if label:
                    relation_labels[str(key)] = str(label)

    rel_info = data_dir / "all_relation_info.txt"
    if rel_info.exists():
        with open(rel_info, encoding="utf-8") as f:
            next(f, "")
            for line in f:
                rid = line.rstrip("\n").split("\t")[0]
                if rid:
                    relation_labels.setdefault(rid, rid)

    return entity_labels, relation_labels


def entity_label(label_map, entity_id):
    if entity_id is None or str(entity_id) == "":
        return None
    s = str(entity_id)
    return label_map.get(s) or label_map.get(normalize_digits_key(s)) or s.split("::")[-1]


def relation_label(label_map, relation_id):
    s = str(relation_id)
    return label_map.get(s) or s


def load_rule_texts(rule_file):
    texts = {}
    confs = {}
    with open(rule_file, encoding="utf-8") as f:
        # Rule IDs are 1-based (line_no) across training/aggregation artifacts.
        # Keep the same base here to avoid off-by-one text lookup in case reports.
        for rid, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            texts[rid] = parts[3] if len(parts) >= 4 else line.strip()
            try:
                confs[rid] = float(parts[2]) if len(parts) >= 3 else None
            except Exception:
                confs[rid] = None
    return texts, confs


def type_key_to_str(k):
    if isinstance(k, (tuple, list)):
        return "_".join(str(x) for x in k)
    return str(k)


def get_rule_type_weight(model, local_idx):
    if not hasattr(model, "rule_types") or getattr(model, "num_relation_rule_types", 0) <= 0:
        return 1.0, None
    type_local = int(model.rule_local_to_type_local[int(local_idx)].detach().cpu().item())
    if type_local == int(model.pad_rule_type_tok):
        return 1.0, None
    return scalar(model.rule_types.weight[type_local, 0]), type_key_to_str(model.rule_type_keys[type_local])


def get_rule_base_weight(model, local_idx):
    raw = model.rules.weight[int(local_idx), 0]
    if getattr(model, "sign_constraint", False):
        raw = raw**2
    return scalar(raw)


def get_rule_weight_record(model, global_rule_id, mod):
    local = int(model.global_to_local[int(global_rule_id)].detach().cpu().item())
    if local == int(model.pad_local_tok):
        return {"base_weight": None, "type_weight": None, "effective_weight": None, "rule_type": None}
    base = get_rule_base_weight(model, local)
    type_w, type_name = get_rule_type_weight(model, local)
    type_name = type_name or mod.rule_type_r3_by_id.get(int(global_rule_id), "unknown")
    return {
        "base_weight": round6(base),
        "type_weight": round6(type_w),
        "effective_weight": round6(base * type_w),
        "rule_type": str(type_name),
    }


def get_dep_base_weight(model, dep_idx):
    raw = model.dependencies.weight[int(dep_idx), 0]
    if getattr(model, "dependency_sign_constraint", False):
        raw = (raw**2) * model.dependency_pair_sign[int(dep_idx)]
    dependency_mask = getattr(model, "trainable_dependency_grad_mask", None)
    if dependency_mask is not None:
        raw = raw * dependency_mask[int(dep_idx)].to(raw.device)
    return scalar(raw)


def get_dep_type_weight(model, dep_idx):
    if not hasattr(model, "dependency_types") or getattr(model, "num_relation_dependency_types", 0) <= 0:
        return 1.0, None
    type_local = int(model.dependency_local_to_type_local[int(dep_idx)].detach().cpu().item())
    if type_local == int(model.pad_dependency_type_tok):
        return 1.0, None
    return scalar(model.dependency_types.weight[type_local, 0]), type_key_to_str(model.dependency_type_keys[type_local])


def get_component_scales(model):
    rule_scale = 1.0
    dep_scale = 1.0
    if getattr(model, "use_global_score_scales", False):
        rule_scale = scalar(model.rule_component_scale_raw) ** 2
        dep_scale = scalar(model.dependency_component_scale_raw) ** 2
    return rule_scale, dep_scale


def build_dep_lookup(model):
    cache_name = "_case_detail_dep_lookup"
    if hasattr(model, cache_name):
        return getattr(model, cache_name)
    lookup = defaultdict(list)
    if hasattr(model, "relation_dependency_pairs_global") and hasattr(model, "dependencies"):
        for dep_idx, dep in enumerate(model.relation_dependency_pairs_global):
            a, b, kind = int(dep[0]), int(dep[1]), str(dep[2])
            la = int(model.global_to_local[a].detach().cpu().item())
            lb = int(model.global_to_local[b].detach().cpu().item())
            if la == int(model.pad_local_tok) or lb == int(model.pad_local_tok):
                continue
            key = tuple(sorted((la, lb)))
            lookup[key].append((int(dep_idx), a, b, kind))
    setattr(model, cache_name, lookup)
    return lookup


def decompose_candidate(mod, model, active_rule_ids):
    active_rule_ids = [int(r) for r in active_rule_ids]
    local_rule_ids = []
    for rid in active_rule_ids:
        local = int(model.global_to_local[int(rid)].detach().cpu().item())
        if local != int(model.pad_local_tok):
            local_rule_ids.append((rid, local))
    active_local_set = {local for _rid, local in local_rule_ids}
    rule_scale, dep_global_scale = get_component_scales(model)

    rule_components = defaultdict(float)
    rule_values = {}
    for rid, local in local_rule_ids:
        base = get_rule_base_weight(model, local)
        type_w, type_name = get_rule_type_weight(model, local)
        type_name = type_name or mod.rule_type_r3_by_id.get(rid, "unknown")
        val = base * type_w * rule_scale
        rule_components[str(type_name)] += val
        rule_values[rid] = val
    rule_total = sum(rule_values.values())

    dep_components = defaultdict(float)
    dep_signs = defaultdict(float)
    active_deps = []
    raw_dep_values = []
    if len(active_local_set) >= 2 and hasattr(model, "dependencies"):
        lookup = build_dep_lookup(model)
        for la, lb in combinations(sorted(active_local_set), 2):
            for dep_idx, a, b, kind in lookup.get((la, lb), []):
                base = get_dep_base_weight(model, dep_idx)
                type_w, dep_type = get_dep_type_weight(model, dep_idx)
                dep_type = dep_type or kind
                raw_dep_values.append((dep_idx, a, b, kind, dep_type, base * type_w))

    den = 1.0
    scale_mode = str(getattr(model, "dependency_scale_mode", "none")).lower()
    if scale_mode == "sqrt_active":
        den = math.sqrt(max(len(raw_dep_values), 1))
    elif scale_mode == "log1p_active":
        den = max(math.log1p(len(raw_dep_values)), 1.0)

    for dep_idx, a, b, kind, dep_type, val in raw_dep_values:
        val = (val / den) * dep_global_scale
        dep_components[str(dep_type)] += val
        sign_name = "positive" if kind == "synergy" else "negative"
        dep_signs[sign_name] += val
        active_deps.append(
            {
                "dep_idx": int(dep_idx),
                "left_rule_id": f"R{a}",
                "right_rule_id": f"R{b}",
                "dependency_type": str(dep_type),
                "sign": sign_name,
                "contribution": round6(val),
            }
        )

    dep_total = sum(dep_components.values())
    bias = scalar(model.bias) if hasattr(model, "bias") else 0.0
    return {
        "intercept": bias,
        "rule_total": rule_total,
        "dependency_total": dep_total,
        "total_linear": bias + rule_total + dep_total,
        "rule_components": {k: round6(v) for k, v in sorted(rule_components.items())},
        "dependency_components": {k: round6(v) for k, v in sorted(dep_components.items())},
        "dependency_by_sign": {k: round6(v) for k, v in sorted(dep_signs.items())},
        "active_rule_ids": [rid for rid, _local in local_rule_ids],
        "active_dependencies": active_deps,
    }


def rank_lookup(scores, candidate_ids):
    if not candidate_ids:
        return {}
    values = torch.sort(scores.detach()).values
    cand_scores = scores[torch.as_tensor(candidate_ids, dtype=torch.long, device=scores.device)].detach()
    left = torch.searchsorted(values, cand_scores, right=False)
    right = torch.searchsorted(values, cand_scores, right=True)
    n = int(scores.numel())
    higher = n - right
    equal = right - left
    ranks = higher.float() + (equal.float() + 1.0) / 2.0
    return {int(cid): round6(rank) for cid, rank in zip(candidate_ids, ranks.detach().cpu().tolist())}


def score_lookup(scores, candidate_ids):
    if not candidate_ids:
        return {}
    vals = scores[torch.as_tensor(candidate_ids, dtype=torch.long, device=scores.device)].detach().cpu().tolist()
    return {int(cid): round6(v) for cid, v in zip(candidate_ids, vals)}


def find_entry_for_case(mod, row):
    entity_to_id = {v: i for i, v in enumerate(mod.dataset.entity_ids())}
    rel = int(row["relation"])
    direction = str(row["direction"])
    if direction == "tail":
        head = entity_to_id[str(row["query_entity"])]
        key = (head, rel)
        processed = mod.load_relation_processed(rel, "test", "o")
    else:
        tail = entity_to_id[str(row["fixed_entity"])]
        key = (rel, tail)
        processed = mod.load_relation_processed(rel, "test", "s")
    return key, processed[key]


def rule_records(mod, rule_ids, rule_texts, stage1_model, final_model):
    out = []
    for rid in sorted(set(int(r) for r in rule_ids)):
        stage1 = get_rule_weight_record(stage1_model, rid, mod)
        stage2 = get_rule_weight_record(final_model, rid, mod)
        out.append(
            {
                "rule_id": f"R{rid}",
                "rule_type": str(stage2.get("rule_type") or stage1.get("rule_type") or mod.rule_type_r3_by_id.get(int(rid), "unknown")),
                "rule_text": rule_texts.get(int(rid), ""),
                "weight": {
                    "stage1": stage1,
                    "stage2": stage2,
                },
            }
        )
    return out


def dependency_record(final_model, dep_idx, dep):
    base = get_dep_base_weight(final_model, dep_idx)
    type_w, dep_type = get_dep_type_weight(final_model, dep_idx)
    dep_type = dep_type or dep["dependency_type"]
    _, dep_global_scale = get_component_scales(final_model)
    return {
        "dep_id": f"D{dep_idx}",
        "left_rule_id": dep["left_rule_id"],
        "right_rule_id": dep["right_rule_id"],
        "dependency_type": str(dep_type),
        "sign": dep["sign"],
        "dep_text": f"{dep['left_rule_id']} + {dep['right_rule_id']}",
        "weight": {
            "stage1": {
                "base_weight": 0.0,
                "type_weight": None,
                "effective_weight": 0.0,
                "note": "dependency absent in stage1",
            },
            "stage2": {
                "base_weight": round6(base),
                "type_weight": round6(type_w),
                "effective_weight_without_active_scaling": round6(base * type_w * dep_global_scale),
            },
        },
    }


def type_weight_dict(model, kind):
    out = {}
    if kind == "rule" and hasattr(model, "rule_types") and getattr(model, "num_relation_rule_types", 0) > 0:
        for i, key in enumerate(model.rule_type_keys):
            out[type_key_to_str(key)] = round6(scalar(model.rule_types.weight[i, 0]))
    elif kind == "dependency" and hasattr(model, "dependency_types") and getattr(model, "num_relation_dependency_types", 0) > 0:
        for i, key in enumerate(model.dependency_type_keys):
            out[type_key_to_str(key)] = round6(scalar(model.dependency_types.weight[i, 0]))
    if not out:
        out["none"] = 1.0
    return out


def make_case_id(row, idx):
    relation = str(row["relation_name"])
    rel = safe_id(relation.split("/")[-1] if "/" in relation else relation)
    ent = safe_id(row["query_entity"] if str(row.get("query_entity", "")) else row.get("fixed_entity", "entity"))
    return f"{safe_id(row['dataset'])}_{rel}_{ent}_{idx:04d}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CASE_CSV).fillna("")
    only_dataset = str(os.environ.get("ONLY_DATASET", "")).strip()
    limit_cases = int(os.environ.get("LIMIT_CASES", "0") or "0")
    if only_dataset:
        df = df[df["dataset"] == only_dataset].copy()
    if limit_cases > 0:
        df = df.head(limit_cases).copy()
    for path in OUT_DIR.glob("*.yml"):
        path.unlink()

    label_maps = {dataset: load_label_maps(dataset) for dataset in sorted(df["dataset"].unique())}

    grouped = df.groupby(["dataset", "best_config"], sort=False)
    written = 0
    for (dataset, best_config), group in grouped:
        first = group.iloc[0].to_dict()
        cfg, mod = casegen.build_group_runtime(first)
        rule_texts, _rule_confs = load_rule_texts(cfg["rule_file"])
        entity_labels, relation_labels = label_maps[dataset]

        for rel, rel_group in group.groupby("relation", sort=False):
            rel = int(rel)
            print(f"[case-detail] dataset={dataset} config={best_config} relation={rel} cases={len(rel_group)}", flush=True)
            mod.args.relation = rel
            mod.clear_relation_processed_cache(rel)
            mod.args.synergy = False
            mod.args.redundancy = False
            stage1_model = casegen.train_stage1_only(mod, rel)
            final_head, final_tail = casegen.load_final_directional_states(mod, rel, ROOT / "data" / dataset / "aggregation" / best_config)

            entity_ids = mod.dataset.entity_ids()
            entity_to_id = {v: i for i, v in enumerate(entity_ids)}

            for _row_idx, row_series in rel_group.iterrows():
                row = row_series.to_dict()
                _key, processed_entry = find_entry_for_case(mod, row)
                final_model = final_tail if row["direction"] == "tail" else final_head

                candidate_ids = [int(c) for c in list(processed_entry["candidates"])]
                candidate_to_rules = {
                    int(cand): [int(r) for r in rules]
                    for cand, rules in zip(list(processed_entry["candidates"]), list(processed_entry["rules"]))
                }
                before_scores, _candidate_count, _active_rule_count = casegen.score_key(mod, stage1_model, processed_entry)
                after_scores, _, _ = casegen.score_key(mod, final_model, processed_entry)
                before_ranks = rank_lookup(before_scores, candidate_ids)
                after_ranks = rank_lookup(after_scores, candidate_ids)
                before_rank_scores = score_lookup(before_scores, candidate_ids)
                after_rank_scores = score_lookup(after_scores, candidate_ids)

                gold_id = entity_to_id[str(row["gold_entity"])]
                before_top1_id = entity_to_id[str(row["stage1_top1"])]
                after_top1_id = entity_to_id[str(row["final_top1"])]

                all_rule_ids = set()
                all_deps = {}
                candidate_records = []
                for cid in candidate_ids:
                    active_rules = candidate_to_rules.get(int(cid), [])
                    before = decompose_candidate(mod, stage1_model, active_rules)
                    after = decompose_candidate(mod, final_model, active_rules)
                    all_rule_ids.update(before["active_rule_ids"])
                    all_rule_ids.update(after["active_rule_ids"])
                    for dep in after["active_dependencies"]:
                        all_deps[dep["dep_idx"]] = dep

                    eid = entity_ids[int(cid)]
                    candidate_records.append(
                        {
                            "entity_id": eid,
                            "entity_label": entity_label(entity_labels, eid),
                            "is_gold": bool(int(cid) == int(gold_id)),
                            "is_stage1_top1": bool(int(cid) == int(before_top1_id)),
                            "is_stage2_top1": bool(int(cid) == int(after_top1_id)),
                            "rank": {
                                "before_dep": before_ranks.get(int(cid)),
                                "after_dep": after_ranks.get(int(cid)),
                            },
                            "rank_score": {
                                "before_dep": before_rank_scores.get(int(cid)),
                                "after_dep": after_rank_scores.get(int(cid)),
                            },
                            "active_rules": [f"R{rid}" for rid in active_rules],
                            "active_dependencies": [f"D{dep['dep_idx']}" for dep in after["active_dependencies"]],
                            "score": {
                                "before_dep": {
                                    "total_linear": round6(before["total_linear"]),
                                    "intercept": round6(before["intercept"]),
                                    "rule_total": round6(before["rule_total"]),
                                    "rule_components": {"by_rule_type": before["rule_components"]},
                                },
                                "after_dep": {
                                    "total_linear": round6(after["total_linear"]),
                                    "intercept": round6(after["intercept"]),
                                    "rule_total": round6(after["rule_total"]),
                                    "dependency_total": round6(after["dependency_total"]),
                                    "rule_components": {"by_rule_type": after["rule_components"]},
                                    "dependency_components": {
                                        "by_dependency_type": after["dependency_components"],
                                        "by_sign": after["dependency_by_sign"],
                                    },
                                },
                            },
                        }
                    )

                if row["direction"] == "tail":
                    head_id = str(row["query_entity"])
                    tail_id = None
                else:
                    head_id = None
                    tail_id = str(row["fixed_entity"])

                rel_id = str(row["relation_name"])
                example_id = make_case_id(row, written + 1)
                record = {
                    "example_id": example_id,
                    "dataset": dataset,
                    "split": "test",
                    "config_id": best_config,
                    "query": {
                        "direction": row["direction"],
                        "head_id": head_id,
                        "head_label": entity_label(entity_labels, head_id),
                        "relation_id": rel_id,
                        "relation_label": relation_label(relation_labels, rel_id),
                        "tail_id": tail_id,
                        "tail_label": entity_label(entity_labels, tail_id),
                    },
                    "gold": {
                        "entity_id": row["gold_entity"],
                        "entity_label": entity_label(entity_labels, row["gold_entity"]),
                    },
                    "relation_context": {
                        "relation_index": int(row["relation"]),
                        "relation_gloss_zh": row.get("relation_gloss_zh", ""),
                        "mapping_direction": row.get("mapping_direction", ""),
                        "baseline_mrr_relation": round6(row.get("baseline_mrr_relation")),
                        "final_mrr_relation": round6(row.get("final_mrr_relation")),
                    },
                    "type_weights": {
                        "rule": type_weight_dict(final_model, "rule"),
                        "dependency": type_weight_dict(final_model, "dependency"),
                    },
                    "rules": rule_records(mod, all_rule_ids, rule_texts, stage1_model, final_model),
                    "dependencies": [
                        dependency_record(final_model, dep_idx, dep)
                        for dep_idx, dep in sorted(all_deps.items())
                    ],
                    "candidates": candidate_records,
                }
                out_path = OUT_DIR / f"{example_id}.yml"
                out_path.write_text(
                    yaml.safe_dump(compact_lists(record), allow_unicode=True, sort_keys=False, width=240),
                    encoding="utf-8",
                )
                written += 1

    print(f"wrote {written} case files to {OUT_DIR}")


if __name__ == "__main__":
    main()
