import argparse
import glob
import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm


def _init_pool_worker():
    # Keep every process lightweight and avoid oversubscription.
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can be raised when interop threads were already set.
            pass


def _is_small_dataset(dataset_name: str, train_key_count: int, num_relations: int) -> bool:
    name = dataset_name.lower()

    # Explicit name hints first.
    if any(tag in name for tag in ("codex-s", "small", "toy")):
        return True
    if any(tag in name for tag in ("codex-l", "large")):
        return False

    # Fallback heuristic.
    return train_key_count < 300_000 and num_relations <= 100


def _split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def parse_rule_file_stats(rule_file: str):
    num_rules = 0
    max_rule_id = 0
    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            num_rules += 1
            max_rule_id = line_no
    return num_rules, max_rule_id


def save(obj, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def read_index_triples(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                continue
            yield tuple(int(part) for part in parts)


class LocalKvsIndex:
    def __init__(self, mapping):
        self._mapping = mapping

    def keys(self):
        return self._mapping.keys()

    def __contains__(self, key):
        return key in self._mapping

    def __getitem__(self, key):
        return self._mapping[key]

    def __len__(self):
        return len(self._mapping)


class LocalDataset:
    def __init__(self, folder):
        self.folder = folder
        self._indexes = {}
        self._entity_ids = None
        self._relation_ids = None

    def _build_kvs_index(self, split, key_cols, value_col):
        buckets = defaultdict(list)
        for triple in read_index_triples(os.path.join(self.folder, f"{split}.del")):
            key = tuple(triple[col] for col in key_cols)
            buckets[key].append(triple[value_col])
        return LocalKvsIndex(buckets)

    def index(self, name):
        if name not in self._indexes:
            split_name, key, _to, value = name.split("_")
            if (key, value) == ("sp", "o"):
                self._indexes[name] = self._build_kvs_index(split_name, [0, 1], 2)
            elif (key, value) == ("po", "s"):
                self._indexes[name] = self._build_kvs_index(split_name, [1, 2], 0)
            else:
                raise ValueError(f"Unsupported local index: {name}")
        return self._indexes[name]

    def entity_ids(self):
        if self._entity_ids is None:
            self._entity_ids = read_ids(os.path.join(self.folder, "entity_ids.del"))
        return self._entity_ids

    def relation_ids(self):
        if self._relation_ids is None:
            self._relation_ids = read_ids(os.path.join(self.folder, "relation_ids.del"))
        return self._relation_ids

    def num_relations(self):
        return len(self.relation_ids())


def load_applied_rules(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _empty_compact_split():
    return {
        "rules_flat": torch.tensor([], dtype=torch.int32),
        "offsets": torch.tensor([0], dtype=torch.int64),
        "golds": torch.tensor([], dtype=torch.float32).reshape(-1, 1),
        "num_samples": 0,
    }


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


def build_compact_split(sp_to_o, processed_sp, relation, direction="o"):
    rules_flat = []
    offsets = [0]
    golds = []

    for key in sp_to_o.keys():
        if direction == "o":
            e, r = key
        else:
            r, e = key

        if r != relation and relation != -1:
            continue
        if key not in processed_sp:
            continue

        candidates = processed_sp[key]["candidates"]
        rules_per_candidate = processed_sp[key]["rules"]
        for ix, prediction in enumerate(candidates):
            rule_ids = rules_per_candidate[ix]
            if len(rule_ids) == 0:
                continue
            rules_flat.extend(rule_ids)
            offsets.append(len(rules_flat))
            golds.append(int(prediction in sp_to_o[key]))

    rules_flat_t = torch.tensor(rules_flat, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    golds_t = torch.tensor(golds, dtype=torch.float32).reshape(-1, 1)

    return {
        "rules_flat": rules_flat_t,
        "offsets": offsets_t,
        "golds": golds_t,
        "num_samples": int(golds_t.shape[0]),
    }


def concat_compact_splits(split_a, split_b):
    if split_a["num_samples"] == 0:
        return split_b
    if split_b["num_samples"] == 0:
        return split_a

    rules_flat = torch.cat([split_a["rules_flat"], split_b["rules_flat"]], dim=0)
    offsets_b_shifted = split_b["offsets"][1:] + split_a["rules_flat"].shape[0]
    offsets = torch.cat([split_a["offsets"], offsets_b_shifted], dim=0)
    golds = torch.cat([split_a["golds"], split_b["golds"]], dim=0)

    return {
        "rules_flat": rules_flat,
        "offsets": offsets,
        "golds": golds,
        "num_samples": int(golds.shape[0]),
    }


def generate_dataset(relation):
    train_set_o = build_compact_split(train_sp_to_o, processed_sp_train, relation)
    train_set_s = build_compact_split(train_po_to_s, processed_po_train, relation, direction="s")

    train_set = concat_compact_splits(train_set_o, train_set_s)

    data_obj = {
        "format": "compact_varlen_int32_v1",
        "pad_tok": int(PAD_TOK),
        "num_rules": int(LEN_RULES),
        "train": train_set,
    }

    if args["output"] is not None:
        save(data_obj, args["output"], f"dataset_{relation}")

    return relation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates datasets for bce")
    parser.add_argument("-d", "--dataset", help="Name of the dataset", default="codex-m")
    parser.add_argument("--data_root", help="Dataset root folder", default="data")
    parser.add_argument(
        "--applied_rules_dir",
        help="Directory containing relation-wise train applied rules: train_<relation>.json",
        required=True,
    )
    parser.add_argument("--rule_file", help="Path to rules file", default="")
    parser.add_argument("-o", "--output", help="Folder where datasets are written", default=None)
    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="Worker processes for dataset generation.")
    args = vars(parser.parse_args())
    dataset_dir = os.path.join(args["data_root"], args["dataset"])
    if args["output"] is None:
        args["output"] = os.path.join(dataset_dir, "datasets")
    if args["rule_file"] == "":
        args["rule_file"] = os.path.join(dataset_dir, "rules", "rules-1000-5")

    dataset = LocalDataset(dataset_dir)

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    train_sp_to_o = dataset.index("train_sp_to_o")
    train_po_to_s = dataset.index("train_po_to_s")

    entity_ids = read_ids(os.path.join(dataset_dir, "entity_ids.del"))
    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    entity_id_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}
    relation_id_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}

    LEN_RULES, MAX_RULE_ID = parse_rule_file_stats(args["rule_file"])
    PAD_TOK = MAX_RULE_ID + 1

    num_relations = dataset.num_relations()

    head_pattern = os.path.join(args["applied_rules_dir"], "train_*_head.json")
    tail_pattern = os.path.join(args["applied_rules_dir"], "train_*_tail.json")
    if not glob.glob(head_pattern) and not glob.glob(tail_pattern):
        raise FileNotFoundError(
            f"No relation-wise head/tail files found under: {args['applied_rules_dir']}"
        )

    print(f"[create_datasets] relation-wise mode: reading train_<relationId>_head/tail.json")

    for relation in tqdm(range(num_relations), total=num_relations):
        head_path = os.path.join(args["applied_rules_dir"], f"train_{relation}_head.json")
        tail_path = os.path.join(args["applied_rules_dir"], f"train_{relation}_tail.json")

        if not os.path.exists(head_path) and not os.path.exists(tail_path):
            train_set = _empty_compact_split()
        else:
            applied_rules_train = {
                "head": load_applied_rules(head_path) if os.path.exists(head_path) else {},
                "tail": load_applied_rules(tail_path) if os.path.exists(tail_path) else {},
            }
            processed_sp_train, processed_po_train = build_processed_from_applied(
                applied_rules_train,
                entity_id_to_idx,
                relation_id_to_idx,
            )
            train_set_o = build_compact_split(train_sp_to_o, processed_sp_train, relation)
            train_set_s = build_compact_split(train_po_to_s, processed_po_train, relation, direction="s")
            train_set = concat_compact_splits(train_set_o, train_set_s)

        data_obj = {
            "format": "compact_varlen_int32_v1",
            "pad_tok": int(PAD_TOK),
            "num_rules": int(LEN_RULES),
            "train": train_set,
        }
        if args["output"] is not None:
            save(data_obj, args["output"], f"dataset_{relation}")
