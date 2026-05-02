#!/usr/bin/env python
# coding: utf-8
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import copy
import glob
import importlib.util
import json
import math
import os
import pickle
import re
import gc
import shutil
import uuid
import warnings
from datetime import datetime
from functools import partial
from os.path import exists
from pprint import pformat
from time import perf_counter

import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


STEP_TIMINGS = defaultdict(float)
STEP_COUNTS = defaultdict(int)
RELATION_PROCESSED_CACHE = {}
DEPENDENCY_MASK_RULE_WEIGHT_THRESHOLD_RATIO = 0.01
STEP_GPU_REQUIRED = {
    "load_dataloaders": False,
    "epoch_train.iter_create": False,
    "epoch_train.iter_finalize": False,
    "epoch_train.batch_data_wait": False,
    "epoch_train.batch_regularization": True,
    "epoch_train.batch_to_device": False,
    "epoch_train.batch_forward_backward": True,
    "epoch_train": True,
    "epoch_model_to_cpu": False,
    "epoch_eval_head": False,
    "epoch_eval_tail": False,
    "epoch_model_to_device": False,
    "epoch_eval.rank_prepare_tensors": False,
    "epoch_eval.rank_model_infer": True,
    "epoch_eval.rank_rankcalc": True,
    "save_outputs": False,
}


@contextmanager
def step_timer(step_name):
    start = perf_counter()
    try:
        yield
    finally:
        STEP_TIMINGS[step_name] += perf_counter() - start
        STEP_COUNTS[step_name] += 1


def print_step_profile():
    if len(STEP_TIMINGS) == 0:
        return

    total = sum([v for k, v in STEP_TIMINGS.items() if '.' not in k])
    print("\n===== Step Timing Summary =====")
    print("step_name,total_seconds,calls,avg_seconds,gpu_required")

    for step_name, seconds in sorted(STEP_TIMINGS.items(), key=lambda x: x[1], reverse=True):
        calls = STEP_COUNTS[step_name]
        avg = seconds / max(calls, 1)
        gpu_required = STEP_GPU_REQUIRED.get(step_name, "unknown")
        print(f"{step_name},{seconds:.6f},{calls},{avg:.6f},{gpu_required}")

    print(f"TOTAL_PROFILED_SECONDS,{total:.6f}")
    print("===== End Step Timing Summary =====\n")


def save(obj, folder, name=None, override=False):
    if name is None:
        name = uuid.uuid4()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}"
    # if exists(path_to_file):
    #     print(f"Warning name {name} exists in cache, do you want to overwrite y/n?")
    #     confirm = input() if not override else "y"
    #     if confirm != "y":
    #         return None

    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def load(folder, name):
    path_to_file = f"{folder}/{name}"
    if exists(path_to_file):
        return pickle.load(open(f"{folder}/{name}", "rb"))
    else:
        print("No such name in cache")
        return None


def timed_dataloader_batches(dataloader):
    with step_timer("epoch_train.iter_create"):
        data_iter = iter(dataloader)

    try:
        while True:
            with step_timer("epoch_train.batch_data_wait"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
            yield batch
    finally:
        with step_timer("epoch_train.iter_finalize"):
            pass


def train(dataloader, model, loss_fn, optimizer, reg=False, num_unseen=0):
    model.train()
    train_loss = 0
    n_loss = 0
    for i, (rules, y) in enumerate(timed_dataloader_batches(dataloader)):

        # compute regularization
        if reg and num_unseen > 0:
            num_batches = len(dataloader)
            if num_unseen > num_batches:
                num_unseen = num_batches
            if i % int(num_batches / num_unseen) == 0:
                with step_timer("epoch_train.batch_regularization"):
                    rule_confs = torch.nn.functional.sigmoid(model.rules.weight)
                    sudo_false = torch.zeros_like(rule_confs)
                    loss = loss_fn(rule_confs, sudo_false) / dataloader.batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Compute prediction error

        if getattr(dataloader, "on_device", False):
            rules_ = rules
            y_ = y
        else:
            with step_timer("epoch_train.batch_to_device"):
                rules_ = rules.long().to(args.device, non_blocking=True)
                y_ = y.to(args.device, non_blocking=True)

        with step_timer("epoch_train.batch_forward_backward"):
            pred = model(rules_)
            loss = loss_fn(pred.reshape(-1, 1), y_)
            if loss.requires_grad and getattr(args, "dep_l1_lambda", 0.0) > 0 and hasattr(model, "dependency_l1_penalty"):
                loss = loss + float(args.dep_l1_lambda) * model.dependency_l1_penalty()

            train_loss += loss.item()
            n_loss += 1
            # In dependency-only stage, some batches may not activate any trainable dependency pair.
            # In that case the loss has no gradient path to parameters, so we skip the optimizer step.
            if not loss.requires_grad:
                continue
            optimizer.zero_grad()
            loss.backward()
            if (
                hasattr(model, "trainable_dependency_grad_mask")
                and model.trainable_dependency_grad_mask is not None
                and hasattr(model, "dependencies")
                and model.dependencies.weight.grad is not None
            ):
                grad_mask = model.trainable_dependency_grad_mask.reshape(-1, 1).to(model.dependencies.weight.grad.device)
                model.dependencies.weight.grad.mul_(grad_mask)
            optimizer.step()

    return train_loss / n_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (rules, y) in enumerate(dataloader):

            rules = rules.long().to(args.device)
            y = y.to(args.device)

            pred = model(rules).reshape(-1, 1)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            pred_prob = torch.sigmoid(pred)
            correct += ((pred_prob > 0.5) == y.to(args.device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss


def _rank_from_scores_tensor(scores_tensor, golds_t, test_filter_t, fill_value=0.0):
    neg_scores = -1.0 * scores_tensor
    gold_scores = neg_scores[golds_t].clone()

    base_scores = neg_scores.clone()
    base_scores[golds_t] = fill_value
    if test_filter_t is not None:
        base_scores[test_filter_t] = fill_value

    num_golds = int(golds_t.shape[0])
    if num_golds == 0:
        return torch.empty((0,), dtype=torch.float32, device=scores_tensor.device)

    # 对每个 gold 直接做比较计数，避免每个 key 的全排序开销。
    # 这里不做分块：单个 key 下 gold 通常较少，直接一次性计算更简洁。
    pairwise_cmp = base_scores.unsqueeze(0)
    gold_scores_col = gold_scores.unsqueeze(1)
    n_less = (pairwise_cmp < gold_scores_col).sum(dim=1).float()
    n_equal = (pairwise_cmp == gold_scores_col).sum(dim=1).float()

    fill_t = torch.tensor(fill_value, device=scores_tensor.device)
    n_less = n_less - (fill_t < gold_scores).float()
    n_equal = n_equal + 1.0 - (fill_t == gold_scores).float()
    ranks = n_less + (n_equal + 1.0) / 2.0
    return ranks


def rank_batch_group(nnm, batch_items):
    """
    batch_items: list of (golds, candidates, rules, test_filter)
    Returns list of (rank, rank_raw, n)
    """
    model_device = next(nnm.parameters()).device
    if model_device.type == "cpu":
        raise RuntimeError("GPU-only eval is enabled, but model is on CPU")

    fill_value = 0.0
    num_entities = dataset.num_entities()

    outputs = [None] * len(batch_items)

    non_empty_positions = []
    non_empty_rules = []
    non_empty_candidate_lens = []

    for i, (golds_t, candidates_t, rules_t, _test_filter_t) in enumerate(batch_items):
        n = len(golds_t)
        if len(candidates_t) == 0 or len(rules_t) == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=model_device)
            outputs[i] = (empty, empty, n)
            continue

        non_empty_positions.append(i)
        non_empty_rules.append(rules_t)
        non_empty_candidate_lens.append(int(candidates_t.shape[0]))

    if len(non_empty_positions) == 0:
        return outputs

    # Pad rules across keys in this group so we can run one forward pass.
    max_rule_len = max(int(r.shape[1]) for r in non_empty_rules)
    padded_rules = []
    for r in non_empty_rules:
        if int(r.shape[1]) == max_rule_len:
            padded_rules.append(r)
        else:
            pad_cols = max_rule_len - int(r.shape[1])
            pad_block = torch.full((int(r.shape[0]), pad_cols), PAD_TOK, dtype=r.dtype, device=r.device)
            padded_rules.append(torch.cat([r, pad_block], dim=1))
    rules_all = torch.cat(padded_rules, dim=0)

    with step_timer("epoch_eval.rank_model_infer"):
        with torch.no_grad():
            pred_all = nnm(rules_all).detach()
            pred_all = torch.sigmoid(pred_all).detach()
    max_conf_all = RULE_CONF_TABLE[rules_all].max(dim=1, keepdim=True).values
    score_all = (pred_all * max_conf_all).squeeze(dim=1)
    score_raw_all = pred_all.squeeze(dim=1)

    score_chunks = torch.split(score_all, non_empty_candidate_lens, dim=0)
    score_raw_chunks = torch.split(score_raw_all, non_empty_candidate_lens, dim=0)

    with step_timer("epoch_eval.rank_rankcalc"):
        for chunk_ix, pos in enumerate(non_empty_positions):
            golds_t, candidates_t, _rules_t, test_filter_t = batch_items[pos]
            scores = torch.full((num_entities,), fill_value, device=model_device)
            scores_raw = torch.full((num_entities,), fill_value, device=model_device)

            scores[candidates_t] = score_chunks[chunk_ix]
            scores_raw[candidates_t] = score_raw_chunks[chunk_ix]

            rank = _rank_from_scores_tensor(scores, golds_t, test_filter_t, fill_value=fill_value)
            rank_raw = _rank_from_scores_tensor(scores_raw, golds_t, test_filter_t, fill_value=fill_value)
            outputs[pos] = (rank, rank_raw, len(golds_t))

    return outputs


def build_relation_key_index(index_dict, direction="o"):
    relation_to_keys = defaultdict(list)
    if direction == "o":
        for key in index_dict.keys():
            relation_to_keys[key[1]].append(key)
    else:
        for key in index_dict.keys():
            relation_to_keys[key[0]].append(key)
    return relation_to_keys


def get_relation_processed_root():
    explicit_dir = str(getattr(args, "relation_processed_dir", "") or "").strip()
    if explicit_dir != "":
        return explicit_dir
    return os.path.join(args.directory_explanations, "relation")


def _processed_file_name(split_name, direction):
    prefix = "processed_sp" if direction == "o" else "processed_po"
    return f"{prefix}_{split_name}.pkl"


def _relation_processed_file_path(relation, split_name, direction):
    return os.path.join(get_relation_processed_root(), str(int(relation)), _processed_file_name(split_name, direction))


def _load_relation_processed_from_global(relation, split_name, direction):
    direction_name = "o" if direction == "o" else "s"
    global_path = os.path.join(args.directory_explanations, _processed_file_name(split_name, direction))
    if not exists(global_path):
        return {}

    print(
        f"[processed] relation-local file missing; fallback to global subset "
        f"relation={relation} split={split_name} direction={direction_name} path={global_path}"
    )
    processed_global = pickle.load(open(global_path, "rb"))
    relation_subset = {
        key: processed_global[key]
        for key in relation_keys[f"{split_name}_{direction_name}"].get(int(relation), [])
        if key in processed_global
    }
    del processed_global
    gc.collect()
    return relation_subset


def load_relation_processed(relation, split_name, direction):
    cache_key = (int(relation), str(split_name), str(direction))
    if cache_key in RELATION_PROCESSED_CACHE:
        return RELATION_PROCESSED_CACHE[cache_key]

    relation_path = _relation_processed_file_path(relation, split_name, direction)
    if exists(relation_path):
        processed = pickle.load(open(relation_path, "rb"))
    else:
        processed = _load_relation_processed_from_global(relation, split_name, direction)

    RELATION_PROCESSED_CACHE[cache_key] = processed
    return processed


def clear_relation_processed_cache(relation=None):
    if relation is None:
        RELATION_PROCESSED_CACHE.clear()
        return

    relation = int(relation)
    for cache_key in [k for k in RELATION_PROCESSED_CACHE.keys() if int(k[0]) == relation]:
        del RELATION_PROCESSED_CACHE[cache_key]


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


class LocalKvsIndex:
    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, key):
        return self._mapping[key]

    def get(self, key, default_return_value=None):
        return self._mapping.get(key, default_return_value)

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def __len__(self):
        return len(self._mapping)


class LocalDataset:
    def __init__(self, folder):
        self.folder = folder
        self._triples = {}
        self._indexes = {}
        self._entity_ids = None
        self._relation_ids = None

    def _load_triples(self, split):
        path = os.path.join(self.folder, f"{split}.del")
        triples_np = np.loadtxt(path, usecols=(0, 1, 2), dtype=np.int32)
        if triples_np.ndim == 1:
            triples_np = triples_np.reshape(1, 3)
        return torch.from_numpy(triples_np)

    def split(self, split):
        if split not in self._triples:
            self._triples[split] = self._load_triples(split)
        return self._triples[split]

    @staticmethod
    def _build_kvs_index(triples, key_cols, value_col):
        buckets = defaultdict(list)
        triples_np = triples.cpu().numpy()
        for row in triples_np:
            key = tuple(int(row[col]) for col in key_cols)
            buckets[key].append(int(row[value_col]))
        return LocalKvsIndex({
            key: torch.tensor(values, dtype=torch.long)
            for key, values in buckets.items()
        })

    def index(self, name):
        if name not in self._indexes:
            split_name, key, _to, value = name.split("_")
            if (key, value) == ("sp", "o"):
                self._indexes[name] = self._build_kvs_index(self.split(split_name), [0, 1], 2)
            elif (key, value) == ("po", "s"):
                self._indexes[name] = self._build_kvs_index(self.split(split_name), [1, 2], 0)
            elif (key, value) == ("so", "p"):
                self._indexes[name] = self._build_kvs_index(self.split(split_name), [0, 2], 1)
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

    def num_entities(self):
        return len(self.entity_ids())

    def num_relations(self):
        return len(self.relation_ids())

def split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


RULE_TYPE_PAREN_RE = re.compile(r"\([^()]*\)")
RULE_TYPE_WS_RE = re.compile(r"\s+")
_ANALYSIS_RULE_PARSER = None


def load_analysis_rule_parser():
    global _ANALYSIS_RULE_PARSER
    if _ANALYSIS_RULE_PARSER is not None:
        return _ANALYSIS_RULE_PARSER

    module_path = os.path.join(os.path.dirname(__file__), "scripts", "analysis_rule.py")
    spec = importlib.util.spec_from_file_location("analysis_rule_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load RuleParser from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ANALYSIS_RULE_PARSER = module.RuleParser
    return _ANALYSIS_RULE_PARSER


def normalize_rule_type_string(rule_str: str):
    try:
        _head_relation, _body_relations, _variable_count, rule_info = load_analysis_rule_parser().parse_rule(rule_str)
        normalized = rule_info.get("normalized_rule", rule_str)
    except Exception:
        normalized = rule_str

    rule_type = RULE_TYPE_PAREN_RE.sub("", normalized)
    rule_type = rule_type.replace("<=", " <= ")
    rule_type = RULE_TYPE_WS_RE.sub(" ", rule_type).strip()
    return normalized, rule_type


def is_d_rule_struct(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    if len(parts) != 2:
        return False
    body = parts[1]
    return body.count("(A,") + body.count(",A)") == 1


def is_b_rule_struct(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    if len(parts) != 2:
        return False
    head = parts[0].strip()
    return "(X,Y)" in head


def classify_rule_type_r3(rule_str: str) -> str:
    if is_b_rule_struct(rule_str):
        return "B"
    if is_d_rule_struct(rule_str):
        return "Ud"
    return "Uc"


def classify_rule_type_group(rule_type_r3: str, grouping: str):
    grouping = str(grouping).lower()
    if grouping == "none":
        return None
    if rule_type_r3 is None:
        return None
    if grouping == "r2":
        return "B" if rule_type_r3 == "B" else "U"
    if grouping == "r3":
        return str(rule_type_r3)
    raise ValueError(f"Unknown rule grouping: {grouping}")


def classify_dependency_type_group(rule_type_a_r3: str, rule_type_b_r3: str, grouping: str):
    grouping = str(grouping).lower()
    if grouping == "none":
        return None
    if rule_type_a_r3 is None or rule_type_b_r3 is None:
        return None
    if grouping == "d3":
        coarse_a = "B" if rule_type_a_r3 == "B" else "U"
        coarse_b = "B" if rule_type_b_r3 == "B" else "U"
        return tuple(sorted((coarse_a, coarse_b)))
    if grouping == "d6":
        return tuple(sorted((str(rule_type_a_r3), str(rule_type_b_r3))))
    raise ValueError(f"Unknown dependency grouping: {grouping}")


def resolve_type_grouping(type_grouping: str):
    type_grouping = str(type_grouping).lower()
    if type_grouping == "none":
        return {
            "type_grouping": "none",
            "rule_grouping": "none",
            "dependency_grouping": "none",
            "use_global_score_scales": False,
        }
    if type_grouping == "rd":
        return {
            "type_grouping": "rd",
            "rule_grouping": "none",
            "dependency_grouping": "none",
            "use_global_score_scales": True,
        }
    if type_grouping == "r2d3":
        return {
            "type_grouping": "r2d3",
            "rule_grouping": "r2",
            "dependency_grouping": "d3",
            "use_global_score_scales": False,
        }
    if type_grouping == "r3d6":
        return {
            "type_grouping": "r3d6",
            "rule_grouping": "r3",
            "dependency_grouping": "d6",
            "use_global_score_scales": False,
        }
    raise ValueError(f"Unknown type_grouping: {type_grouping}")


def compute_rule_init_values_from_conf(confs: torch.Tensor, sign_constraint: bool, init_mode: str) -> torch.Tensor:
    init_mode = str(init_mode).lower()
    if init_mode == "surprisal":
        confs = confs.clamp(min=0.0, max=1 - 1e-7)
        base = -torch.log(1 - confs)
    elif init_mode == "conf":
        base = confs
    else:
        raise ValueError(f"Unknown rule_init_mode: {init_mode}")

    if sign_constraint:
        return torch.sqrt(torch.clamp(base, min=0.0))
    return base


def extract_head_relation(rule_str: str):
    head = rule_str.split(" <= ", 1)[0].strip()
    if "(" not in head:
        return ""
    return head.split("(", 1)[0].strip()


def parse_rule_file_metadata(rule_file: str, relation_ids):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_map = defaultdict(list)
    rule_conf_by_id = {}
    rule_relation_by_id = {}
    rule_type_r3_by_id = {}
    rule_type_members_r3_by_relation = defaultdict(lambda: defaultdict(list))
    num_rules = 0
    max_rule_id = 0

    print(f"Parsing rule file: {rule_file}")
    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = split_rule_line(line)
            if len(parts) < 4:
                continue

            num_rules += 1
            max_rule_id = line_no

            try:
                num_preds = int(float(parts[0].strip()))
                num_true = int(float(parts[1].strip()))
            except Exception:
                num_preds = 0
                num_true = 0
            conf = (num_true / (num_preds + 5)) if num_preds >= 0 else 0.0
            rule_conf_by_id[line_no] = float(conf)

            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is not None:
                rule_type_r3 = classify_rule_type_r3(parts[3].strip())
                rule_map[rel_id].append(int(line_no))
                rule_relation_by_id[int(line_no)] = int(rel_id)
                rule_type_r3_by_id[int(line_no)] = str(rule_type_r3)
                rule_type_members_r3_by_relation[int(rel_id)][str(rule_type_r3)].append(int(line_no))

    return {
        "rule_map": dict(rule_map),
        "rule_conf_by_id": rule_conf_by_id,
        "rule_relation_by_id": rule_relation_by_id,
        "rule_type_r3_by_id": dict(rule_type_r3_by_id),
        "rule_type_members_r3_by_relation": {
            int(relation): {str(rule_type): list(rule_ids) for rule_type, rule_ids in type_map.items()}
            for relation, type_map in rule_type_members_r3_by_relation.items()
        },
        "num_rules": int(num_rules),
        "max_rule_id": int(max_rule_id),
    }


def parse_filtered_dependency_file(dependency_file: str, rule_relation_by_id, dependency_type: str):
    dependency_by_relation = defaultdict(list)
    if not dependency_file or (not os.path.exists(dependency_file)):
        return {}

    print(f"Parsing filtered dependency file: {dependency_file} ({dependency_type})")
    loaded_dependency_count = 0
    seen_pairs = defaultdict(set)
    with open(dependency_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue

            try:
                id1 = int(parts[0])
                id2 = int(parts[1])
            except Exception:
                continue

            lift = 0.0
            if len(parts) >= 3:
                try:
                    lift = float(parts[2])
                except Exception:
                    lift = 0.0

            rel1 = rule_relation_by_id.get(id1)
            rel2 = rule_relation_by_id.get(id2)
            if rel1 is None or rel2 is None or rel1 != rel2:
                continue

            a, b = (id1, id2) if id1 <= id2 else (id2, id1)
            if (a, b) in seen_pairs[rel1]:
                continue
            seen_pairs[rel1].add((a, b))

            dependency_by_relation[rel1].append((a, b, dependency_type, float(lift)))
            loaded_dependency_count += 1

    print(f"Loaded {loaded_dependency_count} dependencies across {len(dependency_by_relation)} relations")
    return dict(dependency_by_relation)


def summarize_dependency_type_candidates(candidate_map):
    relation_type_counts = {}
    total_pairs = 0
    total_types = 0

    for relation, deps in candidate_map.items():
        if str(getattr(args, "dependency_grouping", "none")).lower() == "none":
            relation_type_counts[int(relation)] = 0
            total_pairs += int(len(deps))
            continue
        dep_type_set = set()
        for a, b in deps:
            type_a_r3 = rule_type_r3_by_id.get(int(a))
            type_b_r3 = rule_type_r3_by_id.get(int(b))
            dep_type = classify_dependency_type_group(type_a_r3, type_b_r3, args.dependency_grouping)
            if dep_type is None:
                continue
            dep_type_set.add(dep_type)
        relation_type_counts[int(relation)] = int(len(dep_type_set))
        total_pairs += int(len(deps))
        total_types += int(len(dep_type_set))

    print(
        f"Dependency-type candidate summary: {total_pairs} candidate pairs across {len(candidate_map)} relations, "
        f"{total_types} relation-local unique dependency types"
    )
    return relation_type_counts


def read_id_names(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                idx, name = parts
                idx = int(idx)
                while len(names) <= idx:
                    names.append("")
                names[idx] = name
            elif parts and parts[0] != "":
                names.append(parts[0])
    return names


ID_NAME_CACHE = {}


def get_entity_relation_names():
    cache_key = (args.data_root, args.dataset)
    if cache_key not in ID_NAME_CACHE:
        root = os.path.join(args.data_root, args.dataset)
        ID_NAME_CACHE[cache_key] = (
            read_id_names(os.path.join(root, "entity_ids.del")),
            read_id_names(os.path.join(root, "relation_ids.del")),
        )
    return ID_NAME_CACHE[cache_key]


def build_rank_rows(relation, direction, keys, data, results, stage):
    entity_names, relation_names = get_entity_relation_names()
    relation_name = relation_names[int(relation)] if int(relation) < len(relation_names) else str(relation)
    experiment_name = str(getattr(args, "export_experiment_name", "") or "").strip()
    if experiment_name == "":
        experiment_name = os.path.basename(os.path.normpath(args.experiment))
    rows = []
    for key, item, result in zip(keys, data, results):
        golds_t = item[0]
        ranks_t, ranks_raw_t, _n = result
        gold_ids = [int(v) for v in golds_t.detach().cpu().tolist()]
        ranks = [float(v) for v in ranks_t.detach().cpu().tolist()]
        ranks_raw = [float(v) for v in ranks_raw_t.detach().cpu().tolist()]
        if direction == "o":
            source_id = int(key[0])
            known_entity_id = source_id
            query = f"{entity_names[source_id]} {relation_name} ?"
            direction_name = "tail"
        else:
            object_id = int(key[1])
            known_entity_id = object_id
            query = f"? {relation_name} {entity_names[object_id]}"
            direction_name = "head"
        if not ranks and gold_ids:
            ranks = [math.inf for _ in gold_ids]
            ranks_raw = [math.inf for _ in gold_ids]
        for gold_id, rank, rank_raw in zip(gold_ids, ranks, ranks_raw):
            rows.append(
                {
                    "dataset": args.dataset,
                    "experiment": experiment_name,
                    "stage": stage,
                    "relation_id": int(relation),
                    "relation": relation_name,
                    "direction": direction_name,
                    "query_key": "|".join(str(int(v)) for v in key),
                    "known_entity_id": known_entity_id,
                    "known_entity": entity_names[known_entity_id],
                    "target_entity_id": gold_id,
                    "target_gt_entity": entity_names[gold_id],
                    "query": query,
                    "rank": rank,
                    "rr": (1.0 / rank) if rank > 0 else 0.0,
                    "rank_raw": rank_raw,
                    "rr_raw": (1.0 / rank_raw) if rank_raw > 0 else 0.0,
                }
            )
    return rows


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False, return_rank_rows=False, stage=""):
    nnm.eval()
    # 优化点：直接使用全局 relation_keys 索引，避免每次 get_ranks 线性扫描所有 keys。
    split_name = "valid" if filter_test else "test"
    direction_name = "o" if direction == "o" else "s"
    keys = relation_keys[f"{split_name}_{direction_name}"].get(relation, [])

    if len(keys) == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=EVAL_DEVICE)
        if return_rank_rows:
            return empty, empty, 0, []
        return empty, empty, 0

    data = []
    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o":
                if key in test_sp_to_o.keys():
                    test_filter = test_sp_to_o[key].long().to(EVAL_DEVICE, non_blocking=True)
            else:
                if key in test_po_to_s.keys():
                    test_filter = test_po_to_s[key].long().to(EVAL_DEVICE, non_blocking=True)

        golds = sp_to_o[key].long().to(EVAL_DEVICE, non_blocking=True)
        candidates = torch.empty((0,), dtype=torch.long, device=EVAL_DEVICE)
        rules = torch.empty((0, 0), dtype=torch.long, device=EVAL_DEVICE)
        if key in processed:
            if "candidates_tensor_gpu" not in processed[key]:
                processed[key]["candidates_tensor_gpu"] = torch.as_tensor(
                    processed[key]["candidates"], dtype=torch.long, device=EVAL_DEVICE
                )
            candidates = processed[key]["candidates_tensor_gpu"]

            # 优化点：对每个 key 的规则列表只做一次 nested->padded 构造并缓存。
            # 否则每次 eval 都会重复执行：
            # [torch.tensor(x) for x in rules] + nested_tensor + to_padded_tensor
            # 这是典型 CPU 热点。缓存后后续 epoch 直接复用张量，显著降低 rank_prepare_tensors 时间。
            if "rules_padded_tensor" not in processed[key]:
                with step_timer("epoch_eval.rank_prepare_tensors"):
                    rule_lists = processed[key]["rules"]
                    if len(rule_lists) > 0:
                        processed[key]["rules_padded_tensor"] = torch.nested.to_padded_tensor(
                            torch.nested.nested_tensor([torch.tensor(x) for x in rule_lists]), padding=PAD_TOK
                        ).long()
                    else:
                        processed[key]["rules_padded_tensor"] = torch.empty((0, 0), dtype=torch.long)
            if "rules_padded_tensor_gpu" not in processed[key]:
                with step_timer("epoch_eval.rank_prepare_tensors"):
                    processed[key]["rules_padded_tensor_gpu"] = processed[key]["rules_padded_tensor"].to(
                        EVAL_DEVICE, non_blocking=True
                    )

            rules = processed[key]["rules_padded_tensor_gpu"]
        data.append((golds, candidates, rules, test_filter))

    results = []
    key_batch_size = max(int(args.eval_key_batch_size), 1)
    for start in range(0, len(data), key_batch_size):
        end = min(start + key_batch_size, len(data))
        group = data[start:end]
        results.extend(rank_batch_group(nnm, group))

    rank, rank_raw, ns = zip(*results)
    ranks = torch.hstack(rank)
    ranks_raw = torch.hstack(rank_raw)
    n = sum(ns)
    if return_rank_rows:
        return ranks, ranks_raw, n, build_rank_rows(relation, direction, keys, data, results, stage)
    return ranks, ranks_raw, n


def build_relation_rule_type_metadata(relation_rule_ids, relation):
    if str(getattr(args, "rule_grouping", "none")).lower() == "none":
        return {
            "keys": [],
            "supports": [],
            "local_ids": [0 for _ in relation_rule_ids] + [0],
            "pad": 0,
        }

    relation_type_members = defaultdict(list)
    local_ids = []
    for rid in relation_rule_ids:
        rule_type_r3 = rule_type_r3_by_id.get(int(rid))
        grouped_type = classify_rule_type_group(rule_type_r3, args.rule_grouping)
        if grouped_type is not None:
            relation_type_members[str(grouped_type)].append(int(rid))
            local_ids.append(str(grouped_type))
        else:
            local_ids.append(None)

    selected_keys = sorted(relation_type_members.keys())
    type_to_local = {key: idx for idx, key in enumerate(selected_keys)}
    pad_tok = len(selected_keys)
    return {
        "keys": selected_keys,
        "supports": [int(len(relation_type_members[key])) for key in selected_keys],
        "local_ids": [int(type_to_local.get(key, pad_tok)) for key in local_ids] + [int(pad_tok)],
        "pad": int(pad_tok),
    }


def build_relation_dependency_type_metadata(relation_rule_ids, global_to_local, source_pairs):
    if str(getattr(args, "dependency_grouping", "none")).lower() == "none":
        return {
            "keys": [],
            "supports": [],
            "pair_a": [],
            "pair_b": [],
            "pair_type": [],
            "pad": 0,
            "source_pair_count": 0,
        }

    type_counts = defaultdict(int)
    local_type_ids = []

    for dep in source_pairs:
        if len(dep) >= 2:
            a, b = dep[0], dep[1]
        else:
            continue
        local_a = int(global_to_local[int(a)].item())
        local_b = int(global_to_local[int(b)].item())
        if local_a == len(relation_rule_ids) or local_b == len(relation_rule_ids):
            continue
        type_a_r3 = rule_type_r3_by_id.get(int(a))
        type_b_r3 = rule_type_r3_by_id.get(int(b))
        dep_type = classify_dependency_type_group(type_a_r3, type_b_r3, args.dependency_grouping)
        if dep_type is None:
            continue
        local_type_ids.append(dep_type)
        type_counts[dep_type] += 1

    selected_type_keys = sorted(type_counts)
    type_to_local = {dep_type: idx for idx, dep_type in enumerate(selected_type_keys)}
    pad_type_tok = len(selected_type_keys)

    dep_type_supports = [int(type_counts[dep_type]) for dep_type in selected_type_keys]
    return {
        "keys": selected_type_keys,
        "supports": dep_type_supports,
        "local_ids": [int(type_to_local.get(dep_type, pad_type_tok)) for dep_type in local_type_ids],
        "pad": pad_type_tok,
        "source_pair_count": int(len(local_type_ids)),
    }


class LinearAggregator(nn.Module):
    SCALE_MIN = -7.0
    SCALE_MAX = 7.0

    def _effective_positive_scale(self, raw_param, target_dtype, target_device):
        raw = torch.clamp(raw_param, min=self.SCALE_MIN, max=self.SCALE_MAX)
        return (raw**2).to(device=target_device, dtype=target_dtype)

    def _get_active_dependency_base_weights(self, active_pair_idx, target_dtype, target_device):
        if active_pair_idx.numel() == 0:
            return torch.empty((0,), dtype=target_dtype, device=target_device)

        dependency_w_active = self.dependencies.weight[active_pair_idx, 0]
        if self.dependency_sign_constraint:
            dependency_w_active = (dependency_w_active**2) * self.dependency_pair_sign[active_pair_idx]
        dependency_mask = getattr(self, "trainable_dependency_grad_mask", None)
        if dependency_mask is not None:
            dependency_mask_active = dependency_mask[active_pair_idx].to(device=target_device, dtype=target_dtype)
            dependency_w_active = dependency_w_active.to(device=target_device, dtype=target_dtype) * dependency_mask_active
        else:
            dependency_w_active = dependency_w_active.to(device=target_device, dtype=target_dtype)
        static_scale = getattr(self, "dependency_pair_static_scale", None)
        if static_scale is not None and active_pair_idx.numel() > 0:
            dependency_w_active = dependency_w_active * static_scale[active_pair_idx].to(
                device=target_device,
                dtype=target_dtype,
            )
        return dependency_w_active

    def dependency_l1_penalty(self):
        if not hasattr(self, "dependencies") or self.num_relation_dependencies <= 0:
            return torch.zeros((), device=self.rules.weight.device, dtype=self.rules.weight.dtype)
        dep_w = self.dependencies.weight[: self.num_relation_dependencies, 0]
        if self.dependency_sign_constraint:
            dep_w = (dep_w**2) * self.dependency_pair_sign.to(device=dep_w.device, dtype=dep_w.dtype)
        static_scale = getattr(self, "dependency_pair_static_scale", None)
        if static_scale is not None:
            dep_w = dep_w * static_scale.to(device=dep_w.device, dtype=dep_w.dtype)
        if self.num_relation_dependency_types > 0:
            active_dep_type_local = self.dependency_local_to_type_local[: self.num_relation_dependencies]
            valid_dep_local = active_dep_type_local != self.pad_dependency_type_tok
            if bool(valid_dep_local.any().item()):
                type_w = torch.ones_like(dep_w)
                type_w[valid_dep_local] = self.dependency_types.weight[
                    active_dep_type_local[valid_dep_local], 0
                ].to(dep_w.dtype)
                dep_w = dep_w * type_w
        if self.use_global_score_scales:
            dep_w = dep_w * self._effective_positive_scale(
                self.dependency_component_scale_raw,
                dep_w.dtype,
                dep_w.device,
            ).reshape(())
        return dep_w.abs().sum()

    def _aggregate_rule_contribution(self, local_rule_values, local_rule_ids):
        if self.num_relation_rule_types <= 0:
            return local_rule_values.sum(dim=1, keepdim=True)

        active_rule_type_local = self.rule_local_to_type_local[local_rule_ids]
        valid_local = active_rule_type_local != self.pad_rule_type_tok
        if not bool(valid_local.any().item()):
            return local_rule_values.sum(dim=1, keepdim=True)

        safe_rule_type_local = active_rule_type_local.masked_fill(~valid_local, 0)
        weighted_rules = local_rule_values * valid_local.to(local_rule_values.dtype)
        bucket_sum = torch.zeros(
            (int(local_rule_values.shape[0]), self.num_relation_rule_types),
            dtype=local_rule_values.dtype,
            device=local_rule_values.device,
        )
        bucket_sum.scatter_add_(1, safe_rule_type_local, weighted_rules)
        rule_type_w = self.rule_types.weight[: self.num_relation_rule_types, 0].to(local_rule_values.dtype)
        return bucket_sum @ rule_type_w.reshape(-1, 1)

    def _aggregate_dependency_contribution(self, pair_active_chunk, dependency_w_active, active_pair_idx_chunk, target_dtype):
        pair_scores = pair_active_chunk.to(target_dtype) * dependency_w_active.reshape(1, -1)
        if self.num_relation_dependency_types <= 0:
            return pair_scores.sum(dim=1, keepdim=True)

        active_dep_type_local = self.dependency_local_to_type_local[active_pair_idx_chunk]
        valid_dep_local = active_dep_type_local != self.pad_dependency_type_tok
        if not bool(valid_dep_local.any().item()):
            return pair_scores.sum(dim=1, keepdim=True)

        safe_dep_type_local = active_dep_type_local.masked_fill(~valid_dep_local, 0)
        weighted_pairs = pair_scores * valid_dep_local.to(pair_scores.dtype).reshape(1, -1)
        bucket_sum = torch.zeros(
            (int(pair_scores.shape[0]), self.num_relation_dependency_types),
            dtype=pair_scores.dtype,
            device=pair_scores.device,
        )
        bucket_sum.scatter_add_(1, safe_dep_type_local.reshape(1, -1).expand(int(pair_scores.shape[0]), -1), weighted_pairs)
        dependency_type_w = self.dependency_types.weight[: self.num_relation_dependency_types, 0].to(pair_scores.dtype)
        return bucket_sum @ dependency_type_w.reshape(-1, 1)

    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = RULE_CONF_TABLE_CPU[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
            rule_init_values = compute_rule_init_values_from_conf(
                confs,
                sign_constraint=self.sign_constraint,
                init_mode=getattr(args, "rule_init_mode", "conf"),
            )
            self.rules.weight[: self.num_relation_rules] = rule_init_values
            if self.num_relation_rule_types > 0:
                self.rule_types.weight[: self.num_relation_rule_types].fill_(1.0)
            if self.num_relation_dependencies > 0:
                if getattr(args, "init_dep_with_lift", False) and hasattr(self, "dependency_init_values"):
                    init_values = self.dependency_init_values[: self.num_relation_dependencies].reshape(-1, 1)
                    if self.dependency_sign_constraint:
                        self.dependencies.weight[: self.num_relation_dependencies] = torch.sqrt(torch.clamp(init_values.abs(), min=0.0))
                    else:
                        self.dependencies.weight[: self.num_relation_dependencies] = init_values
                elif self.dependency_sign_constraint:
                        self.dependencies.weight[: self.num_relation_dependencies].fill_(0.1)
                else:
                    self.dependencies.weight[: self.num_relation_dependencies].zero_()
            if self.num_relation_dependency_types > 0:
                self.dependency_types.weight[: self.num_relation_dependency_types].fill_(1.0)
            if getattr(self, "use_global_score_scales", False):
                self.rule_component_scale_raw.fill_(1.0)
                self.dependency_component_scale_raw.fill_(1.0)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[: self.num_relation_rules].reshape(1, -1))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)

    def __init__(
        self,
        relation,
        sign_constraint=False,
        relation_dependencies=None,
        dependency_type_candidates=None,
        dependency_sign_constraint=False,
    ):
        super().__init__()
        self.sign_constraint = sign_constraint
        self.dependency_sign_constraint = dependency_sign_constraint
        self.dependency_scale_mode = str(getattr(args, "dependency_scale_mode", "none")).lower()
        self.dependency_static_norm = str(getattr(args, "dependency_static_norm", "none")).lower()
        self.use_global_score_scales = bool(getattr(args, "use_global_score_scales", False))

        relation_rule_ids = sorted(rule_map.get(relation, []))
        self.relation_rule_ids = np.array(relation_rule_ids, dtype=np.int64)
        self.num_relation_rules = len(relation_rule_ids)
        self.pad_local_tok = self.num_relation_rules

        if relation_dependencies is None:
            relation_dependencies = []
        relation_dependencies = sorted(relation_dependencies, key=lambda x: (x[0], x[1], x[2]))
        local_pairs = []
        global_pairs_filtered = []
        dependency_signs = []
        dependency_init_values = []

        self.rules = nn.Embedding(self.num_relation_rules + 1, 1, padding_idx=self.pad_local_tok)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        if self.use_global_score_scales:
            self.rule_component_scale_raw = nn.Parameter(torch.ones(1, 1))
            self.dependency_component_scale_raw = nn.Parameter(torch.ones(1, 1))

        global_to_local = torch.full((PAD_TOK + 1,), self.pad_local_tok, dtype=torch.long)
        if self.num_relation_rules > 0:
            global_to_local[torch.tensor(relation_rule_ids, dtype=torch.long)] = torch.arange(self.num_relation_rules, dtype=torch.long)
        self.register_buffer("global_to_local", global_to_local)

        rule_type_meta = build_relation_rule_type_metadata(relation_rule_ids, relation)
        self.num_relation_rule_types = len(rule_type_meta["keys"])
        self.rule_type_keys = list(rule_type_meta["keys"])
        self.rule_type_supports = list(rule_type_meta["supports"])
        self.pad_rule_type_tok = int(rule_type_meta["pad"])
        if self.num_relation_rule_types > 0:
            self.rule_types = nn.Embedding(self.num_relation_rule_types + 1, 1, padding_idx=self.pad_rule_type_tok)
            rule_type_local = torch.tensor(rule_type_meta["local_ids"], dtype=torch.long)
        else:
            rule_type_local = torch.empty((self.num_relation_rules,), dtype=torch.long)
        self.register_buffer("rule_local_to_type_local", rule_type_local)

        for dep in relation_dependencies:
            if len(dep) >= 4:
                a, b, dependency_type, dependency_lift = dep[0], dep[1], dep[2], dep[3]
            else:
                a, b, dependency_type = dep[0], dep[1], dep[2]
                dependency_lift = 0.0
            local_a = int(global_to_local[a].item())
            local_b = int(global_to_local[b].item())
            if local_a == self.pad_local_tok or local_b == self.pad_local_tok:
                continue
            local_pairs.append((local_a, local_b))
            global_pairs_filtered.append((int(a), int(b), str(dependency_type)))
            dependency_signs.append(1.0 if dependency_type == "synergy" else -1.0)
            dependency_init_values.append(float(dependency_lift) * 0.1)

        self.num_relation_dependencies = len(local_pairs)
        self.num_relation_synergy = self.num_relation_dependencies
        self.relation_dependency_pairs_global = global_pairs_filtered
        self.relation_synergy_pairs_global = [(int(a), int(b)) for a, b, _kind in global_pairs_filtered]
        self.pad_dependency_tok = self.num_relation_dependencies
        self.pad_synergy_tok = self.pad_dependency_tok
        if self.num_relation_dependencies > 0:
            self.dependencies = nn.Embedding(self.num_relation_dependencies + 1, 1, padding_idx=self.pad_dependency_tok)
            pair_a = torch.tensor([p[0] for p in local_pairs], dtype=torch.long)
            pair_b = torch.tensor([p[1] for p in local_pairs], dtype=torch.long)
            dependency_sign_t = torch.tensor(dependency_signs, dtype=torch.float32)
            dependency_init_t = torch.tensor(dependency_init_values, dtype=torch.float32)
        else:
            pair_a = torch.empty((0,), dtype=torch.long)
            pair_b = torch.empty((0,), dtype=torch.long)
            dependency_sign_t = torch.empty((0,), dtype=torch.float32)
            dependency_init_t = torch.empty((0,), dtype=torch.float32)
        if self.dependency_static_norm == "none" or len(local_pairs) == 0:
            dependency_static_scale_t = torch.ones((len(local_pairs),), dtype=torch.float32)
        elif self.dependency_static_norm == "per_rule_degree":
            degree = torch.zeros((self.num_relation_rules,), dtype=torch.float32)
            if len(local_pairs) > 0:
                pair_a_for_degree = torch.tensor([p[0] for p in local_pairs], dtype=torch.long)
                pair_b_for_degree = torch.tensor([p[1] for p in local_pairs], dtype=torch.long)
                degree.scatter_add_(0, pair_a_for_degree, torch.ones_like(pair_a_for_degree, dtype=torch.float32))
                degree.scatter_add_(0, pair_b_for_degree, torch.ones_like(pair_b_for_degree, dtype=torch.float32))
                dependency_static_scale_t = 1.0 / torch.sqrt(
                    torch.clamp(degree[pair_a_for_degree], min=1.0)
                    * torch.clamp(degree[pair_b_for_degree], min=1.0)
                )
            else:
                dependency_static_scale_t = torch.empty((0,), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown dependency_static_norm: {self.dependency_static_norm}")
        self.register_buffer("synergy_pair_a_local", pair_a)
        self.register_buffer("synergy_pair_b_local", pair_b)
        self.register_buffer("dependency_pair_sign", dependency_sign_t)
        self.register_buffer("dependency_init_values", dependency_init_t)
        self.register_buffer("dependency_pair_static_scale", dependency_static_scale_t)

        if dependency_type_candidates is None:
            dependency_type_candidates = []
        dependency_type_meta = build_relation_dependency_type_metadata(relation_rule_ids, global_to_local, dependency_type_candidates)
        self.num_relation_dependency_types = len(dependency_type_meta["keys"])
        self.dependency_type_keys = list(dependency_type_meta["keys"])
        self.dependency_type_supports = list(dependency_type_meta["supports"])
        self.num_relation_dependency_type_source_pairs = int(dependency_type_meta["source_pair_count"])
        self.pad_dependency_type_tok = int(dependency_type_meta["pad"])
        if self.num_relation_dependency_types > 0:
            self.dependency_types = nn.Embedding(self.num_relation_dependency_types + 1, 1, padding_idx=self.pad_dependency_type_tok)
            dep_type_local = torch.tensor(dependency_type_meta["local_ids"], dtype=torch.long)
        else:
            dep_type_local = torch.empty((0,), dtype=torch.long)
        self.register_buffer("dependency_local_to_type_local", dep_type_local)
        if self.num_relation_dependency_types > 0 and int(dep_type_local.numel()) != int(self.num_relation_dependencies):
            raise RuntimeError(
                f"dependency type local-id count mismatch for relation={relation}: "
                f"num_relation_dependencies={self.num_relation_dependencies}, "
                f"num_dependency_type_local_ids={int(dep_type_local.numel())}, "
                f"source_pair_count={int(dependency_type_meta['source_pair_count'])}"
            )

        self.init_weights()
        self.trainable_dependency_grad_mask = None

    def forward(self, rules):
        local_rule_ids = self.global_to_local[rules.long()]
        local_rules = local_rule_ids
        mask = local_rules == self.pad_local_tok
        local_rules = self.rules(local_rules)
        local_rules.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            local_rules = local_rules**2
        local_rule_values = local_rules.squeeze(dim=2)

        logits = self._aggregate_rule_contribution(local_rule_values, local_rule_ids)
        if self.use_global_score_scales:
            logits = logits * self._effective_positive_scale(
                self.rule_component_scale_raw,
                logits.dtype,
                logits.device,
            )

        if self.num_relation_dependencies > 0 or self.num_relation_dependency_types > 0:
            batch_size = int(local_rules.shape[0])
            active = ~mask
            active_matrix = torch.zeros(
                (batch_size, self.num_relation_rules), dtype=torch.bool, device=local_rules.device
            )
            row_idx = torch.arange(batch_size, device=local_rules.device).unsqueeze(1).expand_as(local_rules.squeeze(dim=2))
            active_matrix[row_idx[active], local_rule_ids[active]] = True

            pair_chunk = max(int(getattr(args, "dependency_chunk_size", 0)), 1)

            if self.num_relation_dependencies > 0:
                dependency_score = torch.zeros((batch_size, 1), dtype=logits.dtype, device=local_rules.device)
                dependency_active_count = None
                if self.dependency_scale_mode != "none":
                    dependency_active_count = torch.zeros((batch_size, 1), dtype=logits.dtype, device=local_rules.device)
                active_rules_in_batch = active_matrix.any(dim=0)
                active_pair_mask = active_rules_in_batch[self.synergy_pair_a_local] & active_rules_in_batch[
                    self.synergy_pair_b_local
                ]
                active_pair_idx = torch.nonzero(active_pair_mask, as_tuple=False).reshape(-1)

                if active_pair_idx.numel() > 0:
                    pair_a_active = self.synergy_pair_a_local[active_pair_idx]
                    pair_b_active = self.synergy_pair_b_local[active_pair_idx]
                    dependency_w_active = self._get_active_dependency_base_weights(
                        active_pair_idx, local_rule_values.dtype, local_rules.device
                    )

                    active_pair_count = int(active_pair_idx.numel())
                    for start in range(0, active_pair_count, pair_chunk):
                        end = min(start + pair_chunk, active_pair_count)
                        a_local = pair_a_active[start:end]
                        b_local = pair_b_active[start:end]
                        chunk_pair_idx = active_pair_idx[start:end]
                        pair_active_chunk = active_matrix[:, a_local] & active_matrix[:, b_local]
                        if dependency_active_count is not None:
                            dependency_active_count = dependency_active_count + pair_active_chunk.sum(
                                dim=1, keepdim=True
                            ).to(logits.dtype)
                        dependency_score = dependency_score + self._aggregate_dependency_contribution(
                            pair_active_chunk,
                            dependency_w_active[start:end],
                            chunk_pair_idx,
                            local_rule_values.dtype,
                        )
                if dependency_active_count is not None:
                    if self.dependency_scale_mode == "sqrt_active":
                        dependency_den = torch.sqrt(torch.clamp(dependency_active_count, min=1.0))
                    elif self.dependency_scale_mode == "log1p_active":
                        dependency_den = torch.log1p(dependency_active_count)
                        dependency_den = torch.clamp(dependency_den, min=1.0)
                    else:
                        raise ValueError(f"Unknown dependency_scale_mode: {self.dependency_scale_mode}")
                    dependency_score = dependency_score / dependency_den
                if self.use_global_score_scales:
                    dependency_score = dependency_score * self._effective_positive_scale(
                        self.dependency_component_scale_raw,
                        dependency_score.dtype,
                        dependency_score.device,
                    )
                if getattr(args, "dep_score_clip_gamma", 0.0) > 0:
                    clip_limit = float(args.dep_score_clip_gamma) * torch.clamp(logits.detach().abs(), min=1.0e-6)
                    dependency_score = torch.clamp(dependency_score, -clip_limit, clip_limit)
                logits = logits + dependency_score

        logits = logits + self.bias
        return logits


def calc_mrr(tail_mrr, head_mrr, attr="maximums_t"):
    relation = tail_mrr.relation
    if relation != head_mrr.relation:
        raise ValueError("head_mrr and tail_mrr must track the same relation")

    rn = test_torch[test_torch[:, 1] == relation].shape[0]
    if rn == 0:
        return 0.0, 0.0

    tail_rank = getattr(tail_mrr, attr) * rn
    head_rank = getattr(head_mrr, attr) * rn
    tail_rank_raw = getattr(tail_mrr, attr + "_raw") * rn
    head_rank_raw = getattr(head_mrr, attr + "_raw") * rn

    return (head_rank + tail_rank) / (2 * rn), (head_rank_raw + tail_rank_raw) / (2 * rn)


def build_model_for_relation(relation, relation_dependencies=None):
    relation_dependencies = [] if relation_dependencies is None else relation_dependencies
    return LinearAggregator(
        relation=relation,
        sign_constraint=args.sign_constraint,
        relation_dependencies=relation_dependencies,
        dependency_type_candidates=relation_dependencies,
        dependency_sign_constraint=args.sign_constraint_dependency,
    )


def build_rule_only_model_for_relation(relation):
    return build_model_for_relation(relation, relation_dependencies=None)


class MRR:
    def __init__(self, relation, direction="o", model_builder=None):
        self.relation = relation
        self.direction = direction
        self.model_builder = model_builder if model_builder is not None else build_rule_only_model_for_relation

        self.best_hps = None

        # Use -1 so the first eval checkpoint is always accepted,
        # even when metric values can be exactly 0.
        self.maximums_v = -1.0
        self.maximums_v_raw = -1.0
        self.maximums_v_1 = -1.0
        self.maximums_v_1_raw = -1.0
        self.maximums_v_10 = -1.0
        self.maximums_v_10_raw = -1.0

        self.maximums_t = 0.0
        self.maximums_t_raw = 0.0
        self.maximums_t_1 = 0.0
        self.maximums_t_1_raw = 0.0
        self.maximums_t_10 = 0.0
        self.maximums_t_10_raw = 0.0

        self.valid_sp_to_o = valid_sp_to_o if direction == "o" else valid_po_to_s
        self.valid_processed = load_relation_processed(relation, "valid", direction)
        self.test_sp_to_o = test_sp_to_o if direction == "o" else test_po_to_s
        self.test_processed = load_relation_processed(relation, "test", direction)
        self.nnm = None

    def calc_metrics_(self, ranks, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        mrr = ((1 / ranks).sum() / n).item()
        h1 = ((ranks == 1.0).sum() / n).item()
        h10 = ((ranks <= 10.0).sum() / n).item()
        return mrr, h1, h10

    def calc_metrics(self, nnm, sp_to_o, processed, direction, filter_test=False, return_rank_rows=False, stage=""):
        relation = self.relation
        result = get_ranks(
            nnm,
            sp_to_o,
            processed,
            relation,
            direction,
            filter_test,
            return_rank_rows=return_rank_rows,
            stage=stage,
        )
        if return_rank_rows:
            ranks, ranks_raw, n, rank_rows = result
        else:
            ranks, ranks_raw, n = result
            rank_rows = None
        mrr, h1, h10 = self.calc_metrics_(ranks, n)
        mrr_raw, h1_raw, h10_raw = self.calc_metrics_(ranks_raw, n)
        if return_rank_rows:
            return (mrr, h1, h10, mrr_raw, h1_raw, h10_raw, rank_rows)
        return (mrr, h1, h10, mrr_raw, h1_raw, h10_raw)

    @staticmethod
    def _clone_state_dict(nnm):
        if isinstance(nnm, dict):
            return {k: v.detach().cpu().clone() for k, v in nnm.items()}
        return {k: v.detach().cpu().clone() for k, v in nnm.state_dict().items()}

    def update_from_metrics(self, metrics, nnm, hps):
        v_mrr, v_h1, v_h10, v_mrr_raw, v_h1_raw, v_h10_raw = metrics
        if v_mrr <= self.maximums_v:
            return
        self.maximums_v = float(v_mrr)
        self.maximums_v_1 = float(v_h1)
        self.maximums_v_10 = float(v_h10)
        self.maximums_v_raw = float(v_mrr_raw)
        self.maximums_v_1_raw = float(v_h1_raw)
        self.maximums_v_10_raw = float(v_h10_raw)
        self.nnm = self._clone_state_dict(nnm)
        self.best_hps = hps

    def _evaluate_saved_state_on_test(self, state_dict):
        if state_dict is None:
            return None
        model = self.model_builder(self.relation).to(args.device)
        model.load_state_dict(state_dict, strict=True)
        return self.calc_metrics(model, self.test_sp_to_o, self.test_processed, direction=self.direction)

    def finalize_test(self):
        test_metrics = self._evaluate_saved_state_on_test(self.nnm)
        if test_metrics is None:
            return
        self.maximums_t = float(test_metrics[0])
        self.maximums_t_1 = float(test_metrics[1])
        self.maximums_t_10 = float(test_metrics[2])
        self.maximums_t_raw = float(test_metrics[3])
        self.maximums_t_1_raw = float(test_metrics[4])
        self.maximums_t_10_raw = float(test_metrics[5])

def compact_mrr_for_save(mrr_obj):
    mrr_light = copy.copy(mrr_obj)

    # Drop large references to dataset/processed structures
    mrr_light.valid_sp_to_o = None
    mrr_light.valid_processed = None
    mrr_light.test_sp_to_o = None
    mrr_light.test_processed = None
    mrr_light.model_builder = None

    # Keep only model parameters instead of full model objects
    if mrr_light.nnm is not None:
        if isinstance(mrr_light.nnm, dict):
            mrr_light.nnm = {k: v.detach().cpu().clone() for k, v in mrr_light.nnm.items()}
        else:
            mrr_light.nnm = {k: v.detach().cpu() for k, v in mrr_light.nnm.state_dict().items()}

    return mrr_light


class FastTensorBatchLoader:
    def __init__(self, rules, ys, batch_size, shuffle=False, device=None, preload_to_device=False):
        self.rules = rules.contiguous()
        self.ys = ys.contiguous()
        self.cpu_rules = self.rules
        self.cpu_ys = self.ys
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.size = int(ys.shape[0])
        self.on_device = False

        if preload_to_device and device is not None:
            # 一次性把该 relation 的训练数据搬到设备，避免每个 batch 反复 host->device 拷贝。
            self.rules = self.rules.long().to(device, non_blocking=True)
            self.ys = self.ys.to(device, non_blocking=True)
            self.on_device = True

    def __len__(self):
        if self.size == 0:
            return 0
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.size == 0:
            return
        if self.shuffle:
            perm = torch.randperm(self.size)
            for i in range(0, self.size, self.batch_size):
                idx = perm[i: i + self.batch_size]
                yield self.rules[idx], self.ys[idx]
        else:
            for i in range(0, self.size, self.batch_size):
                yield self.rules[i: i + self.batch_size], self.ys[i: i + self.batch_size]


def materialize_compact_split_to_padded(split_dict):
    offsets = split_dict["offsets"].long()
    rules_flat = split_dict["rules_flat"].int()
    ys = split_dict["golds"].float()

    num_samples = int(ys.shape[0])
    if num_samples == 0:
        return torch.empty((0, 0), dtype=torch.int32), ys

    lengths = offsets[1:] - offsets[:-1]
    max_len = int(lengths.max().item())
    padded = torch.full((num_samples, max_len), PAD_TOK, dtype=torch.int32)

    for i in range(num_samples):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        n = end - start
        if n > 0:
            padded[i, :n] = rules_flat[start:end]

    return padded, ys


def get_relation_rule_count(relation):
    return int(len(rule_map.get(int(relation), [])))


def is_large_relation(relation):
    return get_relation_rule_count(relation) > 1_000_000


def get_effective_batch_size_for_relation(relation):
    base_batch_size = int(args.batch_size)
    relation_rule_count = get_relation_rule_count(relation)
    if relation_rule_count > 2_000_000:
    #     return min(base_batch_size, 1024)
    # if relation_rule_count > 1_000_000:
        return min(base_batch_size, 2048)
    return base_batch_size


def load_dataloaders(dataset_directory, relation):
    with step_timer("load_dataloaders"):
        data_obj = load(dataset_directory, f"dataset_{relation}.p")

        if not (isinstance(data_obj, dict) and data_obj.get("format") == "compact_varlen_int32_v1"):
            raise ValueError(
                "dataset format is not compact_varlen_int32_v1. "
                "Please regenerate dataset_*.p with updated create_datasets.py"
            )

        train_split = data_obj["train"]
        rules_padded, ys = materialize_compact_split_to_padded(train_split)
        effective_batch_size = get_effective_batch_size_for_relation(relation)
        relation_rule_count = get_relation_rule_count(relation)
        if effective_batch_size != int(args.batch_size):
            print(
                f"[batch_size] relation={relation} num_relation_rules={relation_rule_count} "
                f"base_batch_size={int(args.batch_size)} effective_batch_size={effective_batch_size}"
            )

        train_loader = FastTensorBatchLoader(
            rules_padded,
            ys,
            batch_size=effective_batch_size,
            shuffle=args.shuffle_train,
            device=args.device,
            preload_to_device=False,
        )

        if len(train_loader) == 0:
            return None, train_split
        return train_loader, train_split


def get_effective_rule_weights(model):
    with torch.no_grad():
        raw = model.rules.weight[: model.num_relation_rules, 0].detach().cpu()
        if getattr(model, "sign_constraint", False):
            return raw**2
        return raw


def build_dependency_blocks(model, train_split, dependency_chunk_size):
    if getattr(model, "num_relation_dependencies", 0) <= 0:
        return []

    relation_rule_ids = [int(rid) for rid in model.relation_rule_ids.tolist()]
    effective_rule_weights = get_effective_rule_weights(model)
    rule_rank = {
        int(rid): idx
        for idx, rid in enumerate(
            sorted(
                relation_rule_ids,
                key=lambda rid: (
                    -float(effective_rule_weights[int(model.global_to_local[int(rid)].item())].item()),
                    int(rid),
                ),
            )
        )
    }

    owner_to_pair_indices = defaultdict(list)
    for pair_idx, (a, b, _kind) in enumerate(model.relation_dependency_pairs_global):
        a = int(a)
        b = int(b)
        rank_a = rule_rank.get(a, math.inf)
        rank_b = rule_rank.get(b, math.inf)
        owner = a if (rank_a, a) <= (rank_b, b) else b
        owner_to_pair_indices[int(owner)].append(int(pair_idx))

    ordered_rules = sorted(owner_to_pair_indices.keys(), key=lambda rid: (rule_rank[rid], rid))
    blocks = []
    tail_rule_ids = []
    tail_pair_indices = []
    limit = max(int(dependency_chunk_size), 1)

    for rid in ordered_rules:
        pair_indices = owner_to_pair_indices[rid]
        if len(pair_indices) >= limit:
            if tail_pair_indices:
                blocks.append({"rule_ids": tail_rule_ids, "pair_indices": tail_pair_indices})
                tail_rule_ids = []
                tail_pair_indices = []
            blocks.append({"rule_ids": [int(rid)], "pair_indices": list(pair_indices)})
            continue

        if tail_pair_indices and (len(tail_pair_indices) + len(pair_indices) >= limit):
            blocks.append({"rule_ids": tail_rule_ids, "pair_indices": tail_pair_indices})
            tail_rule_ids = []
            tail_pair_indices = []

        tail_rule_ids.append(int(rid))
        tail_pair_indices.extend(int(idx) for idx in pair_indices)

    if tail_pair_indices:
        blocks.append({"rule_ids": tail_rule_ids, "pair_indices": tail_pair_indices})

    return blocks


def build_block_dataloader(base_dataloader, owner_rule_ids):
    if base_dataloader is None or len(owner_rule_ids) == 0:
        return None

    cpu_rules = base_dataloader.cpu_rules
    cpu_ys = base_dataloader.cpu_ys
    owner_rule_ids_t = torch.tensor(sorted(set(int(rid) for rid in owner_rule_ids)), dtype=cpu_rules.dtype)
    sample_mask = torch.isin(cpu_rules, owner_rule_ids_t).any(dim=1)
    if not bool(sample_mask.any().item()):
        return None

    return FastTensorBatchLoader(
        cpu_rules[sample_mask].contiguous(),
        cpu_ys[sample_mask].contiguous(),
        batch_size=base_dataloader.batch_size,
        shuffle=base_dataloader.shuffle,
        device=args.device,
        preload_to_device=False,
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", action="store", help="Name of dataset (libkge)", default="codex-m")
    parser.add_argument("--data_root", action="store", help="Dataset root directory", default="data")
    parser.add_argument("-dev", "--device", action="store", help="Device cpu/cuda", default="cuda")
    parser.add_argument("--max_worker_dataloader", action="store", help="Number of processes for dataloader", default=len(os.sched_getaffinity(0)) - 1, type=int)
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffles the examples before creating batches")
    parser.add_argument("--batch_size", action="store", help="Size of batch", default=4096, type=int)
    parser.add_argument("--lr", action="store", default="0.01,0.005,0.001", help="Learning rate or comma-separated phase learning rates, e.g. 0.01,0.005,0.001")
    parser.add_argument("--max_epoch", action="store", default=60, help="Epochs to run for each learning rate", type=int)
    parser.add_argument("--evaluate_every", action="store", default="4,2,1", help="Evaluation interval or comma-separated phase intervals, e.g. 4,2,1. Use 0 for no eval in a phase.")
    parser.add_argument("--early_stopping", action="store", default=3, type=int, help="Stop if valid metric does not improve for X consecutive evaluations. -1 disables.")
    parser.add_argument("--pos", action="store", default="auto_sqrt", help="Scaling of the loss for positive examples. Use 'auto_sqrt' (default) for sqrt(neg/pos), 'auto_ratio' for neg/pos, or provide a positive number.",)
    parser.add_argument(
        "--rule_init_mode",
        action="store",
        default="conf",
        choices=["conf", "surprisal"],
        help="Initialization for LinearAggregator rule weights: confidence or surprisal transformed from confidence.",
    )
    parser.add_argument("--no_sign_constraint", dest="sign_constraint", action="store_false", help="Disable sign constraint for rule weights.")
    parser.add_argument("--sign_constraint_dependency", dest="sign_constraint_dependency", action="store_true", help="Enable sign constraint for dependency weights.")
    parser.add_argument("--no_sign_constraint_dependency", dest="sign_constraint_dependency", action="store_false", help="Disable sign constraint for dependency weights.")
    parser.add_argument("--init_dep_with_lift", action="store_true", default=False, help="Initialize dependency weights with 0.1 * lift from filtered dependency files.")
    parser.add_argument("--train_rule_in_dependency_stage", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dependency_scale_mode",
        action="store",
        default="none",
        choices=["none", "sqrt_active", "log1p_active"],
        help="Normalize dependency score by the number of active dependencies per query.",
    )
    parser.add_argument(
        "--dependency_static_norm",
        action="store",
        default="none",
        choices=["none", "per_rule_degree"],
        help="Apply a static normalization to each dependency pair before scoring.",
    )
    parser.add_argument(
        "--dep_l1_lambda",
        action="store",
        default=0.0,
        type=float,
        help="L1 regularization strength for effective dependency weights.",
    )
    parser.add_argument(
        "--dep_score_clip_gamma",
        action="store",
        default=0.0,
        type=float,
        help="Clamp dependency score to +/- gamma * abs(rule score) before adding it to the logits. Disabled at 0.",
    )
    parser.add_argument(
        "--dependency_topk_per_rule",
        action="store",
        default=0,
        type=int,
        help="Keep only the top-k incident dependency pairs per rule before model construction. Disabled at 0.",
    )
    parser.add_argument(
        "--dependency_topk_per_kind",
        action="store",
        default=0,
        type=int,
        help="Keep the top-k incident dependency pairs per rule and dependency kind before model construction. Disabled at 0.",
    )
    parser.add_argument(
        "--dependency_topk_score",
        action="store",
        default="abs_lift",
        choices=["abs_lift"],
        help="Static score used by dependency_topk_per_rule.",
    )
    parser.add_argument(
        "--dependency_mask_low_rule_weight",
        action="store_true",
        default=False,
        help="In stage2, mask dependency pairs whose endpoint rules have low stage1 rule weights.",
    )
    parser.set_defaults(sign_constraint=True, sign_constraint_dependency=False)
    parser.add_argument("--relation", action="store", help="Relation to train on", default=0, type=int)
    parser.add_argument("--multiprocess", action="store", help="Number of processes for all-relation run. 0/1 means single-process.", default=0, type=int)
    parser.add_argument(
        "--resume_relation_sweep",
        action="store_true",
        default=False,
        help="Resume an all-relation run by skipping relations that already have valid metric-<rel>.json files in EXPERIMENT_DIR, then re-write metrics-final.json.",
    )
    parser.add_argument(
        "--stage1_only",
        action="store_true",
        default=False,
        help="Run only the rule-only stage. Useful for exporting exact stage-1 per-query ranks.",
    )
    parser.add_argument(
        "--export_per_query_rr_dir",
        default="",
        help="When set, write exact official per-query rank/RR CSV files under this directory.",
    )
    parser.add_argument(
        "--export_experiment_name",
        default="",
        help="Experiment label to store in exported per-query RR rows. Defaults to EXPERIMENT_DIR basename.",
    )
    parser.add_argument("--eval_key_batch_size", action="store", default=64, type=int, help="How many eval keys to group into one model inference call.")
    parser.add_argument("--dependency_chunk_size", action="store", default=4096, type=int, help="Target dependency count for merged stage-2 blocks; also used as forward chunk size for dependency pairs.")
    parser.add_argument("--synergy_pair_chunk_size", dest="dependency_chunk_size", action="store", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rule_file", action="store", default="", help="Path to rules file. Default: <data_root>/<dataset>/rules/rules-1000-5")
    parser.add_argument(
        "--relation_processed_dir",
        action="store",
        default="",
        help="Directory containing relation-local processed_sp_*.pkl and processed_po_*.pkl. Default: <data_root>/<dataset>/application/relation",
    )
    parser.add_argument("--synergy", action="store_true", default=False, help="Load dependencies from synergy_filtered.txt.")
    parser.add_argument("--redundancy", action="store_true", default=False, help="Load dependencies from redundancy_filtered.txt.")
    parser.add_argument("--synergy_file", default="", help="Override synergy dependency file path.")
    parser.add_argument("--redundancy_file", default="", help="Override redundancy dependency file path.")
    parser.add_argument(
        "--type_grouping",
        action="store",
        default="none",
        choices=["none", "rd", "r2d3", "r3d6"],
        help="Aggregation type grouping: none (direct sum), rd (global rule/dep ratios), r2d3, or r3d6.",
    )
    return parser


def parse_csv_schedule(raw_value, cast_fn, name):
    parts = [p.strip() for p in str(raw_value).split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        values = [cast_fn(p) for p in parts]
    except Exception as e:
        raise ValueError(f"Invalid {name}: {raw_value}") from e
    return values


def resolve_pos_weight(pos_arg, train_split, relation):
    pos_raw = str(pos_arg).strip()
    pos_mode = pos_raw.lower()
    ys = train_split["golds"].float().reshape(-1)
    num_samples = int(ys.shape[0])
    num_positive = float(ys.sum().item())
    num_negative = float(num_samples) - num_positive

    if pos_mode in {"auto", "auto_sqrt", "auto_ratio"}:
        if num_positive <= 0 or num_negative <= 0:
            pos_weight = 1.0
            pos_source = f"{pos_mode}_fallback"
        else:
            ratio = num_negative / num_positive
            if pos_mode == "auto_ratio":
                pos_weight = ratio
                pos_source = "auto_ratio"
            else:
                pos_weight = math.sqrt(ratio)
                pos_source = "auto_sqrt"
    else:
        try:
            pos_weight = float(pos_raw)
        except ValueError as e:
            raise ValueError(f"Invalid --pos value: {pos_arg}") from e
        if pos_weight <= 0:
            raise ValueError(f"--pos must be > 0, got {pos_arg}")
        pos_source = "manual"

    print(
        f"[pos_weight] relation={relation} source={pos_source} "
        f"positive={int(num_positive)} negative={int(num_negative)} "
        f"pos_weight={pos_weight:.6g}"
    )
    return pos_weight, pos_source, int(num_positive), int(num_negative)


def build_dependency_stage_training_plan(stage1_lr_values, stage1_eval_every_values, stage1_max_epoch):
    # Stage 2 starts from the stage-1 combined-best checkpoint, so we bias it toward
    # conservative dependency finetuning: use only the smallest LR, but keep the
    # same epoch budget as stage 1 and rely on softer stopping/optimizer settings
    # for stability instead of simply stretching the schedule.
    min_lr = min(float(v) for v in stage1_lr_values)
    stage2_lr_values = [float(min_lr)]
    stage2_eval_every_values = [1]
    stage2_max_epoch = int(stage1_max_epoch)
    return stage2_lr_values, stage2_eval_every_values, stage2_max_epoch


def build_dependency_stage_early_stop_plan(base_patience):
    base_patience = int(base_patience)
    if base_patience <= 0:
        return base_patience, 0

    stage2_patience = max(base_patience * 3, 9)
    stage2_min_epochs_before_stop = max(base_patience * 2, 6)
    return stage2_patience, stage2_min_epochs_before_stop


def build_phase_lengths(max_epoch, num_phases):
    max_epoch = int(max_epoch)
    num_phases = int(num_phases)
    if max_epoch <= 0:
        return []
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if num_phases > max_epoch:
        raise ValueError(f"num_phases ({num_phases}) cannot exceed max_epoch ({max_epoch})")

    if num_phases == 3:
        return [int(max_epoch * 0.4), int(max_epoch * 0.4), max_epoch - int(max_epoch * 0.8)]

    base = max_epoch // num_phases
    rem = max_epoch % num_phases
    return [base + (1 if i < rem else 0) for i in range(num_phases)]


def phase_value_for_epoch(epoch_idx, phase_lengths, phase_values):
    cursor = 0
    for i, phase_len in enumerate(phase_lengths):
        next_cursor = cursor + phase_len
        if epoch_idx < next_cursor:
            local_epoch = epoch_idx - cursor + 1
            return i, local_epoch, phase_values[i]
        cursor = next_cursor
    i = len(phase_values) - 1
    return i, phase_lengths[-1], phase_values[i]


def BCELossR(weights=[1, 1], reduction="mean", apply_sigmoid=False):
    def loss(input, target):
        if apply_sigmoid:
            input = torch.sigmoid(input)
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        if reduction == "libkge":
            bce = (
                bce[target.bool()].sum() / target.bool().sum() + bce[~target.bool()].sum() / (~target.bool()).sum()
            ) / 2.0
        elif reduction == "sum":
            bce = torch.sum(bce)
        elif reduction == "mean":
            bce = torch.mean(bce)
        return bce

    return loss


def copy_rule_state_from_model(src_model, dst_model):
    with torch.no_grad():
        if hasattr(src_model, "rules") and hasattr(dst_model, "rules"):
            n = min(int(src_model.num_relation_rules), int(dst_model.num_relation_rules))
            if n > 0:
                dst_model.rules.weight[:n].copy_(src_model.rules.weight[:n])
        if hasattr(src_model, "rule_types") and hasattr(dst_model, "rule_types"):
            n = min(int(src_model.num_relation_rule_types), int(dst_model.num_relation_rule_types))
            if n > 0:
                dst_model.rule_types.weight[:n].copy_(src_model.rule_types.weight[:n])
        if hasattr(src_model, "bias") and hasattr(dst_model, "bias"):
            dst_model.bias.copy_(src_model.bias)


def copy_rule_state_from_state_dict(src_state_dict, dst_model):
    if src_state_dict is None:
        return

    with torch.no_grad():
        if hasattr(dst_model, "rules") and "rules.weight" in src_state_dict:
            n = min(int(dst_model.num_relation_rules), int(src_state_dict["rules.weight"].shape[0] - 1))
            if n > 0:
                dst_model.rules.weight[:n].copy_(src_state_dict["rules.weight"][:n])
        if hasattr(dst_model, "rule_types") and "rule_types.weight" in src_state_dict:
            n = min(int(dst_model.num_relation_rule_types), int(src_state_dict["rule_types.weight"].shape[0] - 1))
            if n > 0:
                dst_model.rule_types.weight[:n].copy_(src_state_dict["rule_types.weight"][:n])
        if hasattr(dst_model, "bias") and "bias" in src_state_dict:
            dst_model.bias.copy_(src_state_dict["bias"])


def get_effective_rule_weights_from_state_dict(relation, state_dict):
    relation_rule_ids = sorted(rule_map.get(relation, []))
    num_relation_rules = len(relation_rule_ids)
    if num_relation_rules == 0 or state_dict is None or "rules.weight" not in state_dict:
        return torch.empty((0,), dtype=torch.float32)

    raw = state_dict["rules.weight"][:num_relation_rules, 0].detach().cpu().float()
    if args.sign_constraint:
        return raw**2
    return raw


def filter_relation_dependencies_by_rule_strength(relation, relation_dependencies, stage1_state_dict):
    original_dependencies = list(relation_dependencies or [])
    if not getattr(args, "dependency_mask_low_rule_weight", False):
        return original_dependencies, {
            "enabled": False,
            "threshold_ratio": None,
            "threshold_value": None,
            "before": int(len(original_dependencies)),
            "after": int(len(original_dependencies)),
            "removed": 0,
        }

    ratio = float(DEPENDENCY_MASK_RULE_WEIGHT_THRESHOLD_RATIO)
    if ratio < 0:
        raise ValueError(f"dependency_mask_rule_weight_threshold_ratio must be >= 0, got {ratio}")

    effective_rule_weights = get_effective_rule_weights_from_state_dict(relation, stage1_state_dict)
    relation_rule_ids = sorted(rule_map.get(relation, []))
    if effective_rule_weights.numel() == 0 or len(original_dependencies) == 0 or len(relation_rule_ids) == 0:
        return original_dependencies, {
            "enabled": True,
            "threshold_ratio": float(ratio),
            "threshold_value": 0.0,
            "before": int(len(original_dependencies)),
            "after": int(len(original_dependencies)),
            "removed": 0,
        }

    max_weight = float(effective_rule_weights.max().item()) if effective_rule_weights.numel() > 0 else 0.0
    threshold_value = float(max_weight * ratio)
    local_weight_by_rule_id = {
        int(rule_id): float(effective_rule_weights[idx].item())
        for idx, rule_id in enumerate(relation_rule_ids)
    }

    filtered_dependencies = []
    for dep in original_dependencies:
        a, b = int(dep[0]), int(dep[1])
        weight_a = local_weight_by_rule_id.get(a, 0.0)
        weight_b = local_weight_by_rule_id.get(b, 0.0)
        if weight_a < threshold_value or weight_b < threshold_value:
            continue
        filtered_dependencies.append(dep)

    return filtered_dependencies, {
        "enabled": True,
        "threshold_ratio": float(ratio),
        "threshold_value": float(threshold_value),
        "before": int(len(original_dependencies)),
        "after": int(len(filtered_dependencies)),
        "removed": int(len(original_dependencies) - len(filtered_dependencies)),
    }


def filter_relation_dependencies_topk_per_rule(relation_dependencies, topk, score_mode="abs_lift", per_kind_topk=0):
    original_dependencies = list(relation_dependencies or [])
    topk = int(topk)
    per_kind_topk = int(per_kind_topk)
    if topk <= 0 and per_kind_topk <= 0:
        return original_dependencies, {
            "enabled": False,
            "topk": None,
            "per_kind_topk": None,
            "score_mode": str(score_mode),
            "before": int(len(original_dependencies)),
            "after": int(len(original_dependencies)),
            "removed": 0,
        }
    if str(score_mode) != "abs_lift":
        raise ValueError(f"Unknown dependency_topk_score: {score_mode}")
    if len(original_dependencies) == 0:
        return original_dependencies, {
            "enabled": True,
            "topk": None if topk <= 0 else int(topk),
            "per_kind_topk": None if per_kind_topk <= 0 else int(per_kind_topk),
            "score_mode": str(score_mode),
            "before": 0,
            "after": 0,
            "removed": 0,
        }

    incident = defaultdict(list)
    for idx, dep in enumerate(original_dependencies):
        if len(dep) < 2:
            continue
        a, b = int(dep[0]), int(dep[1])
        kind = str(dep[2]) if len(dep) >= 3 else "unknown"
        lift = float(dep[3]) if len(dep) >= 4 else 0.0
        score = abs(lift)
        item = (-score, idx)
        if per_kind_topk > 0:
            incident[(a, kind)].append(item)
            incident[(b, kind)].append(item)
        else:
            incident[a].append(item)
            incident[b].append(item)

    kept = set()
    for items in incident.values():
        limit = per_kind_topk if per_kind_topk > 0 else topk
        for _neg_score, idx in sorted(items)[:limit]:
            kept.add(int(idx))

    filtered_dependencies = [dep for idx, dep in enumerate(original_dependencies) if idx in kept]
    return filtered_dependencies, {
        "enabled": True,
        "topk": None if topk <= 0 else int(topk),
        "per_kind_topk": None if per_kind_topk <= 0 else int(per_kind_topk),
        "score_mode": str(score_mode),
        "before": int(len(original_dependencies)),
        "after": int(len(filtered_dependencies)),
        "removed": int(len(original_dependencies) - len(filtered_dependencies)),
    }


def freeze_rule_parameters_for_synergy_stage(model):
    if hasattr(model, "rules"):
        model.rules.weight.requires_grad_(False)
    if hasattr(model, "rule_types"):
        model.rule_types.weight.requires_grad_(False)
    if hasattr(model, "bias"):
        model.bias.requires_grad_(False)


def build_rule_type_weight_rows(model, initial_rule_type_weights):
    if (
        model is None
        or not hasattr(model, "rule_types")
        or getattr(model, "num_relation_rule_types", 0) <= 0
    ):
        return []

    with torch.no_grad():
        trained_rule_type_weights = (
            model.rule_types.weight[: model.num_relation_rule_types, 0]
            .detach()
            .cpu()
            .numpy()
        )

    return [
        (
            str(rule_type),
            int(support),
            round(float(o), 7),
            round(float(t), 7),
        )
        for rule_type, support, o, t in zip(
            getattr(model, "rule_type_keys", []),
            getattr(model, "rule_type_supports", []),
            initial_rule_type_weights.tolist(),
            trained_rule_type_weights.tolist(),
        )
    ]


def build_dependency_weight_rows(model, dependency_pairs, initial_dependency_weights):
    if (
        model is None
        or len(dependency_pairs) == 0
        or not hasattr(model, "dependencies")
        or getattr(model, "num_relation_dependencies", 0) <= 0
    ):
        return []

    with torch.no_grad():
        trained_dependency_weights = (
            model.dependencies.weight[: model.num_relation_dependencies, 0]
            .detach()
            .cpu()
            .numpy()
        )

    return [
        (
            int(a),
            int(b),
            str(kind),
            round(float(o), 7),
            round(float(t), 7),
            round(float(((o**2) * (1.0 if kind == "synergy" else -1.0)) if args.sign_constraint_dependency else o), 7),
            round(float(((t**2) * (1.0 if kind == "synergy" else -1.0)) if args.sign_constraint_dependency else t), 7),
        )
        for (a, b, kind), o, t in zip(dependency_pairs, initial_dependency_weights.tolist(), trained_dependency_weights.tolist())
    ]


def build_dependency_type_weight_rows(model, initial_dependency_type_weights):
    if (
        model is None
        or not hasattr(model, "dependency_types")
        or getattr(model, "num_relation_dependency_types", 0) <= 0
    ):
        return []

    with torch.no_grad():
        trained_dependency_type_weights = (
            model.dependency_types.weight[: model.num_relation_dependency_types, 0]
            .detach()
            .cpu()
            .numpy()
        )

    return [
        (
            json.dumps(list(dep_type), ensure_ascii=False),
            int(support),
            round(float(o), 7),
            round(float(t), 7),
        )
        for dep_type, support, o, t in zip(
            getattr(model, "dependency_type_keys", []),
            getattr(model, "dependency_type_supports", []),
            initial_dependency_type_weights.tolist(),
            trained_dependency_type_weights.tolist(),
        )
    ]


def build_global_scale_metrics(model, initial_rule_scale, initial_dependency_scale):
    if model is None or not getattr(model, "use_global_score_scales", False):
        return {
            "enabled": False,
            "rule_original": None,
            "rule_trained": None,
            "dependency_original": None,
            "dependency_trained": None,
        }

    with torch.no_grad():
        rule_scale_trained = float((model.rule_component_scale_raw.detach().reshape(-1)[0].cpu().item()) ** 2)
        dependency_scale_trained = float((model.dependency_component_scale_raw.detach().reshape(-1)[0].cpu().item()) ** 2)

    return {
        "enabled": True,
        "rule_original": None if initial_rule_scale is None else float(initial_rule_scale),
        "rule_trained": float(rule_scale_trained),
        "dependency_original": None if initial_dependency_scale is None else float(initial_dependency_scale),
        "dependency_trained": float(dependency_scale_trained),
    }


def build_optimizer_for_model(model, lr, stage_name="rule"):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for optimizer")

    if stage_name == "dependency" and getattr(args, "train_rule_in_dependency_stage", False):
        inherited_params = []
        dependency_params = []
        assigned_param_ids = set()

        def append_param(target_list, param):
            if param is not None and param.requires_grad and id(param) not in assigned_param_ids:
                target_list.append(param)
                assigned_param_ids.add(id(param))

        append_param(inherited_params, getattr(getattr(model, "rules", None), "weight", None))
        append_param(inherited_params, getattr(getattr(model, "rule_types", None), "weight", None))
        append_param(inherited_params, getattr(model, "bias", None))

        append_param(dependency_params, getattr(getattr(model, "dependencies", None), "weight", None))
        append_param(dependency_params, getattr(getattr(model, "dependency_types", None), "weight", None))

        for param in trainable_params:
            if id(param) not in assigned_param_ids:
                inherited_params.append(param)
                assigned_param_ids.add(id(param))

        if dependency_params:
            inherited_lr = max(float(lr) * 0.1, 1e-5)
            param_groups = []
            if inherited_params:
                param_groups.append({"params": inherited_params, "lr": float(inherited_lr)})
            param_groups.append({"params": dependency_params, "lr": float(lr)})
            return torch.optim.Adam(param_groups)

    return torch.optim.Adam(trainable_params, lr=lr)


def build_test_metrics_from_raw(init_head, init_tail):
    return {
        "mrr": float((init_head[0] + init_tail[0]) / 2.0),
        "h1": float((init_head[1] + init_tail[1]) / 2.0),
        "h10": float((init_head[2] + init_tail[2]) / 2.0),
        "mrr_raw": float((init_head[3] + init_tail[3]) / 2.0),
        "h1_raw": float((init_head[4] + init_tail[4]) / 2.0),
        "h10_raw": float((init_head[5] + init_tail[5]) / 2.0),
    }


def evaluate_model_on_test(relation, model, direction_builders):
    head_metrics = MRR(relation=relation, direction="s", model_builder=direction_builders)
    tail_metrics = MRR(relation=relation, direction="o", model_builder=direction_builders)
    head_raw = head_metrics.calc_metrics(model, head_metrics.test_sp_to_o, head_metrics.test_processed, direction="s")
    tail_raw = tail_metrics.calc_metrics(model, tail_metrics.test_sp_to_o, tail_metrics.test_processed, direction="o")
    return build_test_metrics_from_raw(head_raw, tail_raw)


PER_QUERY_RR_FIELDS = [
    "dataset",
    "experiment",
    "stage",
    "relation_id",
    "relation",
    "direction",
    "query_key",
    "known_entity_id",
    "known_entity",
    "target_entity_id",
    "target_gt_entity",
    "query",
    "rank",
    "rr",
    "rank_raw",
    "rr_raw",
]


def per_query_export_dir():
    raw = str(getattr(args, "export_per_query_rr_dir", "") or "").strip()
    if raw == "":
        return None
    return raw


def write_per_query_rr_rows(relation, stage, rows):
    out_root = per_query_export_dir()
    if out_root is None:
        return None
    experiment_name = str(getattr(args, "export_experiment_name", "") or "").strip()
    if experiment_name == "":
        experiment_name = os.path.basename(os.path.normpath(args.experiment))
    out_dir = os.path.join(out_root, args.dataset, experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relation-{int(relation)}-{stage}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_QUERY_RR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[per-query-rr] wrote {len(rows)} rows to {out_path}", flush=True)
    return out_path


def export_model_per_query_rr(relation, model, model_builder, stage):
    if per_query_export_dir() is None:
        return None
    head_metrics = MRR(relation=relation, direction="s", model_builder=model_builder)
    tail_metrics = MRR(relation=relation, direction="o", model_builder=model_builder)
    head_result = head_metrics.calc_metrics(
        model,
        head_metrics.test_sp_to_o,
        head_metrics.test_processed,
        direction="s",
        return_rank_rows=True,
        stage=stage,
    )
    tail_result = tail_metrics.calc_metrics(
        model,
        tail_metrics.test_sp_to_o,
        tail_metrics.test_processed,
        direction="o",
        return_rank_rows=True,
        stage=stage,
    )
    rows = list(head_result[6]) + list(tail_result[6])
    return write_per_query_rr_rows(relation, stage, rows)


def evaluate_current_stage_result(relation, model, model_builder, evaluate_every=1):
    head_mrr = MRR(relation=relation, direction="s", model_builder=model_builder)
    tail_mrr = MRR(relation=relation, direction="o", model_builder=model_builder)
    model = model.to(args.device)

    eval_start = perf_counter()
    head_valid = head_mrr.calc_metrics(
        model, head_mrr.valid_sp_to_o, head_mrr.valid_processed, direction=head_mrr.direction, filter_test=True
    )
    tail_valid = tail_mrr.calc_metrics(
        model, tail_mrr.valid_sp_to_o, tail_mrr.valid_processed, direction=tail_mrr.direction, filter_test=True
    )
    with step_timer("epoch_eval_head"):
        head_test = head_mrr.calc_metrics(model, head_mrr.test_sp_to_o, head_mrr.test_processed, direction=head_mrr.direction)
    with step_timer("epoch_eval_tail"):
        tail_test = tail_mrr.calc_metrics(model, tail_mrr.test_sp_to_o, tail_mrr.test_processed, direction=tail_mrr.direction)
    eval_seconds = perf_counter() - eval_start

    head_mrr.nnm = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tail_mrr.nnm = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_mrr.maximums_v, head_mrr.maximums_v_raw = float(head_valid[0]), float(head_valid[3])
    head_mrr.maximums_v_1, head_mrr.maximums_v_10 = float(head_valid[1]), float(head_valid[2])
    head_mrr.maximums_v_1_raw, head_mrr.maximums_v_10_raw = float(head_valid[4]), float(head_valid[5])
    tail_mrr.maximums_v, tail_mrr.maximums_v_raw = float(tail_valid[0]), float(tail_valid[3])
    tail_mrr.maximums_v_1, tail_mrr.maximums_v_10 = float(tail_valid[1]), float(tail_valid[2])
    tail_mrr.maximums_v_1_raw, tail_mrr.maximums_v_10_raw = float(tail_valid[4]), float(tail_valid[5])
    head_mrr.maximums_t, head_mrr.maximums_t_1, head_mrr.maximums_t_10 = (
        float(head_test[0]),
        float(head_test[1]),
        float(head_test[2]),
    )
    head_mrr.maximums_t_raw, head_mrr.maximums_t_1_raw, head_mrr.maximums_t_10_raw = (
        float(head_test[3]),
        float(head_test[4]),
        float(head_test[5]),
    )
    tail_mrr.maximums_t, tail_mrr.maximums_t_1, tail_mrr.maximums_t_10 = (
        float(tail_test[0]),
        float(tail_test[1]),
        float(tail_test[2]),
    )
    tail_mrr.maximums_t_raw, tail_mrr.maximums_t_1_raw, tail_mrr.maximums_t_10_raw = (
        float(tail_test[3]),
        float(tail_test[4]),
        float(tail_test[5]),
    )

    return {
        "model": model,
        "optimizer": None,
        "tail_mrr": tail_mrr,
        "head_mrr": head_mrr,
        "valid": build_test_metrics_from_raw(head_valid, tail_valid),
        "test_initial": build_test_metrics_from_raw(head_test, tail_test),
        "test": build_test_metrics_from_raw(head_test, tail_test),
        "epochs_trained": 0,
        "train_seconds": 0.0,
        "eval_seconds": float(eval_seconds),
        "evaluate_every": int(evaluate_every),
        "best_valid_epoch": 0,
        "best_valid_combined": float((head_valid[0] + tail_valid[0]) / 2.0),
        "best_valid_combined_raw": float((head_valid[3] + tail_valid[3]) / 2.0),
    }


def build_model_from_state_dict(relation, model_builder, state_dict):
    model = model_builder(relation).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    return model


def run_training_stage(
    relation,
    model,
    model_builder,
    dataloader,
    loss_fn,
    pos,
    lr_values,
    eval_every_values,
    max_epoch,
    stage_name,
    checkpoint_selection="directional",
    early_stopping_patience=None,
    min_epochs_before_stop=0,
):
    tail_mrr = MRR(relation=relation, direction="o", model_builder=model_builder)
    head_mrr = MRR(relation=relation, direction="s", model_builder=model_builder)

    model = model.to(args.device)
    if model.rules.weight.device.type == "cpu":
        raise RuntimeError("GPU-only eval requires CUDA device; please set --device cuda")

    optimizer = build_optimizer_for_model(model, lr_values[0], stage_name=stage_name)

    init_tail = tail_mrr.calc_metrics(model, tail_mrr.test_sp_to_o, tail_mrr.test_processed, direction="o")
    init_head = head_mrr.calc_metrics(model, head_mrr.test_sp_to_o, head_mrr.test_processed, direction="s")
    test_initial = build_test_metrics_from_raw(init_head, init_tail)

    lr_phase_lengths = build_phase_lengths(max_epoch, len(lr_values))
    eval_phase_lengths = build_phase_lengths(max_epoch, len(eval_every_values))

    if early_stopping_patience is None:
        early_stopping_patience = int(args.early_stopping)
    else:
        early_stopping_patience = int(early_stopping_patience)
    min_epochs_before_stop = int(min_epochs_before_stop)
    best_valid_combined = -1.0
    best_valid_combined_raw = -1.0
    no_improve_eval_rounds = 0
    epochs_trained = 0
    train_seconds = 0.0
    eval_seconds = 0.0
    final_loss = None
    evaluate_every = eval_every_values[0]
    best_state = None
    best_state_raw = None
    best_valid_epoch = None
    best_valid_epoch_raw = None

    # Evaluate the untrained starting point as a valid checkpoint candidate.
    has_non_finite = False
    for _name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            has_non_finite = True
            break
    if has_non_finite:
        print(f"[WARN] relation={relation} stage={stage_name}: non-finite params at init valid eval, skip init checkpoint")
    else:
        eval_start = perf_counter()
        with step_timer("epoch0_eval_head"):
            init_head_valid = head_mrr.calc_metrics(
                model, head_mrr.valid_sp_to_o, head_mrr.valid_processed, direction=head_mrr.direction, filter_test=True
            )
        with step_timer("epoch0_eval_tail"):
            init_tail_valid = tail_mrr.calc_metrics(
                model, tail_mrr.valid_sp_to_o, tail_mrr.valid_processed, direction=tail_mrr.direction, filter_test=True
            )
        eval_seconds += perf_counter() - eval_start

        best_valid_combined = (init_head_valid[0] + init_tail_valid[0]) / 2.0
        best_valid_combined_raw = (init_head_valid[3] + init_tail_valid[3]) / 2.0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_state_raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_valid_epoch = 0
        best_valid_epoch_raw = 0
        head_mrr.update_from_metrics(init_head_valid, model, (pos, float(lr_values[0]), -1))
        tail_mrr.update_from_metrics(init_tail_valid, model, (pos, float(lr_values[0]), -1))

    pbar = tqdm(range(max_epoch), desc=f"r{relation}-{stage_name}", leave=False)
    for t in pbar:
        epochs_trained = t + 1
        _lr_phase_idx, _lr_local_epoch, current_lr = phase_value_for_epoch(t, lr_phase_lengths, lr_values)
        _eval_phase_idx, eval_local_epoch, current_eval_every = phase_value_for_epoch(
            t, eval_phase_lengths, eval_every_values
        )
        evaluate_every = current_eval_every

        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)

        train_start = perf_counter()
        with step_timer("epoch_train"):
            final_loss = train(dataloader, model, loss_fn, optimizer, False, 0)
        train_seconds += perf_counter() - train_start

        do_eval_in_phase = (current_eval_every > 0) and ((eval_local_epoch % int(current_eval_every)) == 0)
        do_eval = do_eval_in_phase or (t == max_epoch - 1)

        if do_eval:
            has_non_finite = False
            for _name, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    has_non_finite = True
                    break

            eval_start = perf_counter()
            if has_non_finite:
                print(f"[WARN] relation={relation} stage={stage_name}: non-finite params on valid eval, skip update")
            else:
                with step_timer("epoch_eval_head"):
                    current_head_valid = head_mrr.calc_metrics(
                        model, head_mrr.valid_sp_to_o, head_mrr.valid_processed, direction=head_mrr.direction, filter_test=True
                    )
                with step_timer("epoch_eval_tail"):
                    current_tail_valid = tail_mrr.calc_metrics(
                        model, tail_mrr.valid_sp_to_o, tail_mrr.valid_processed, direction=tail_mrr.direction, filter_test=True
                    )
            eval_seconds += perf_counter() - eval_start

            if not has_non_finite:
                head_mrr.update_from_metrics(current_head_valid, model, (pos, float(current_lr), t))
                tail_mrr.update_from_metrics(current_tail_valid, model, (pos, float(current_lr), t))
                valid_combined = (current_head_valid[0] + current_tail_valid[0]) / 2.0
                valid_combined_raw = (current_head_valid[3] + current_tail_valid[3]) / 2.0
                if valid_combined > best_valid_combined:
                    best_valid_combined = valid_combined
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_valid_epoch = int(t + 1)
                    no_improve_eval_rounds = 0
                else:
                    no_improve_eval_rounds += 1
                if valid_combined_raw > best_valid_combined_raw:
                    best_valid_combined_raw = valid_combined_raw
                    best_state_raw = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_valid_epoch_raw = int(t + 1)

            if (
                (not has_non_finite)
                and early_stopping_patience > 0
                and epochs_trained >= min_epochs_before_stop
                and no_improve_eval_rounds >= early_stopping_patience
            ):
                pbar.set_postfix(
                    loss=f"{final_loss:.5f}",
                    max_mrr=f"{valid_combined:.5f}",
                    lr=f"{current_lr:.6g}",
                )
                break

        max_mrr = max(best_valid_combined, 0.0)
        pbar.set_postfix(loss=f"{final_loss:.5f}", max_mrr=f"{max_mrr:.5f}")

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint selected for relation {relation} at stage {stage_name}")
    if best_state_raw is None:
        best_state_raw = {k: v.detach().cpu().clone() for k, v in best_state.items()}
        best_valid_epoch_raw = best_valid_epoch

    with step_timer("epoch_eval_head"):
        head_mrr.finalize_test()
    with step_timer("epoch_eval_tail"):
        tail_mrr.finalize_test()

    final_test = {
        "mrr": float(calc_mrr(tail_mrr, head_mrr)[0]),
        "h1": float(calc_mrr(tail_mrr, head_mrr, "maximums_t_1")[0]),
        "h10": float(calc_mrr(tail_mrr, head_mrr, "maximums_t_10")[0]),
        "mrr_raw": float(calc_mrr(tail_mrr, head_mrr)[1]),
        "h1_raw": float(calc_mrr(tail_mrr, head_mrr, "maximums_t_1")[1]),
        "h10_raw": float(calc_mrr(tail_mrr, head_mrr, "maximums_t_10")[1]),
    }

    selected_state_dict = {k: v.detach().cpu().clone() for k, v in best_state.items()}
    selected_model = build_model_from_state_dict(relation, model_builder, selected_state_dict)
    selected_stage_result = evaluate_current_stage_result(
        relation,
        selected_model,
        model_builder,
        evaluate_every=evaluate_every,
    )
    selected_test = selected_stage_result["test"]

    selected_state_dict_raw = {k: v.detach().cpu().clone() for k, v in best_state_raw.items()}
    if selected_state_dict_raw.keys() == selected_state_dict.keys() and all(
        torch.equal(selected_state_dict_raw[k], selected_state_dict[k]) for k in selected_state_dict.keys()
    ):
        selected_stage_result_raw = selected_stage_result
    else:
        selected_model_raw = build_model_from_state_dict(relation, model_builder, selected_state_dict_raw)
        selected_stage_result_raw = evaluate_current_stage_result(
            relation,
            selected_model_raw,
            model_builder,
            evaluate_every=evaluate_every,
        )
    selected_test_raw = selected_stage_result_raw["test"]

    if checkpoint_selection == "combined":
        result_model = selected_stage_result["model"]
        result_head_mrr = selected_stage_result["head_mrr"]
        result_tail_mrr = selected_stage_result["tail_mrr"]
        result_test = selected_test
    elif checkpoint_selection == "directional":
        result_model = selected_stage_result["model"]
        result_head_mrr = head_mrr
        result_tail_mrr = tail_mrr
        result_test = final_test
    else:
        raise ValueError(f"Unknown checkpoint_selection: {checkpoint_selection}")

    return {
        "model": result_model,
        "optimizer": optimizer,
        "tail_mrr": result_tail_mrr,
        "head_mrr": result_head_mrr,
        "selected_valid": selected_stage_result.get("valid"),
        "selected_valid_raw": selected_stage_result_raw.get("valid"),
        "test_initial": test_initial,
        "test": result_test,
        "selected_state_dict": selected_state_dict,
        "selected_test": selected_test,
        "selected_state_dict_raw": selected_state_dict_raw,
        "selected_test_raw": selected_test_raw,
        "epochs_trained": int(epochs_trained),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "evaluate_every": int(evaluate_every),
        "best_valid_epoch": None if best_valid_epoch is None else int(best_valid_epoch),
        "best_valid_epoch_raw": None if best_valid_epoch_raw is None else int(best_valid_epoch_raw),
        "best_valid_combined": float(best_valid_combined),
        "best_valid_combined_raw": float(best_valid_combined_raw),
    }


def run_dependency_stage(
    relation,
    model,
    model_builder,
    dataloader,
    loss_fn,
    pos,
    lr_values,
    eval_every_values,
    max_epoch,
    early_stopping_patience,
    min_epochs_before_stop,
):
    if dataloader is None or len(dataloader) == 0:
        raise ValueError(f"Dependency stage received empty dataloader for relation {relation}")

    if hasattr(model, "trainable_dependency_grad_mask"):
        model.trainable_dependency_grad_mask = None

    return run_training_stage(
        relation=relation,
        model=model,
        model_builder=model_builder,
        dataloader=dataloader,
        loss_fn=loss_fn,
        pos=pos,
        lr_values=lr_values,
        eval_every_values=eval_every_values,
        max_epoch=max_epoch,
        stage_name="dependency",
        checkpoint_selection="directional",
        early_stopping_patience=early_stopping_patience,
        min_epochs_before_stop=min_epochs_before_stop,
    )


def aggregate_single(relation):
    def build_best_valid_metrics(stage_result, valid_key="selected_valid", combined_key="best_valid_combined", epoch_key="best_valid_epoch"):
        if stage_result is None:
            return None
        selected_valid = stage_result.get(valid_key)
        if selected_valid is not None:
            return {
                "mrr": float(selected_valid["mrr"]),
                "h1": float(selected_valid["h1"]),
                "h10": float(selected_valid["h10"]),
                "mrr_raw": float(selected_valid["mrr_raw"]),
                "h1_raw": float(selected_valid["h1_raw"]),
                "h10_raw": float(selected_valid["h10_raw"]),
                "combined": float(stage_result.get(combined_key, selected_valid["mrr"])),
                "combined_raw": float(stage_result.get("best_valid_combined_raw", selected_valid["mrr_raw"])),
                "epoch": None if stage_result.get(epoch_key) is None else int(stage_result[epoch_key]),
            }

        stage_head_mrr = stage_result["head_mrr"]
        stage_tail_mrr = stage_result["tail_mrr"]
        best_valid_mrr, best_valid_mrr_raw = calc_mrr(stage_tail_mrr, stage_head_mrr, "maximums_v")
        best_valid_h1, best_valid_h1_raw = calc_mrr(stage_tail_mrr, stage_head_mrr, "maximums_v_1")
        best_valid_h10, best_valid_h10_raw = calc_mrr(stage_tail_mrr, stage_head_mrr, "maximums_v_10")
        return {
            "mrr": float(best_valid_mrr),
            "h1": float(best_valid_h1),
            "h10": float(best_valid_h10),
            "mrr_raw": float(best_valid_mrr_raw),
            "h1_raw": float(best_valid_h1_raw),
            "h10_raw": float(best_valid_h10_raw),
            "head_mrr": float(stage_head_mrr.maximums_v),
            "tail_mrr": float(stage_tail_mrr.maximums_v),
            "head_mrr_raw": float(stage_head_mrr.maximums_v_raw),
            "tail_mrr_raw": float(stage_tail_mrr.maximums_v_raw),
            "combined": float(stage_result.get(combined_key, stage_result["best_valid_combined_raw"])),
            "combined_raw": float(stage_result["best_valid_combined_raw"]),
            "epoch": None if stage_result.get(epoch_key) is None else int(stage_result[epoch_key]),
        }

    relation_start_time = perf_counter()
    load_start_time = perf_counter()
    dataloader, train_split = load_dataloaders(args.directory_preprocessed_datasets, relation)
    load_seconds = perf_counter() - load_start_time

    pos, pos_source, num_train_positive_samples, num_train_negative_samples = resolve_pos_weight(args.pos, train_split, relation)
    lr_values = parse_csv_schedule(args.lr, float, "lr")
    if any(v <= 0 for v in lr_values):
        raise ValueError(f"All lr values must be > 0, got {lr_values}")
    max_epoch = args.max_epoch
    relation_rule_ids = sorted(rule_map.get(relation, []))
    train_dataloader = dataloader
    if train_dataloader is None:
        raise ValueError(f"No training data for relation {relation}")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())

    eval_every_values = parse_csv_schedule(args.evaluate_every, int, "evaluate_every")
    if any(v < 0 for v in eval_every_values):
        raise ValueError(f"All evaluate_every values must be >= 0, got {eval_every_values}")
    dependency_lr_values, dependency_eval_every_values, dependency_max_epoch = build_dependency_stage_training_plan(
        lr_values, eval_every_values, max_epoch
    )
    early_stopping_patience = int(args.early_stopping)
    dependency_early_stopping_patience, dependency_min_epochs_before_stop = build_dependency_stage_early_stop_plan(
        early_stopping_patience
    )

    rule_model_builder = build_rule_only_model_for_relation
    rule_model = rule_model_builder(relation)
    with torch.no_grad():
        initial_rule_weights = rule_model.rules.weight[: rule_model.num_relation_rules, 0].detach().cpu().numpy().copy()
        initial_rule_type_weights = (
            rule_model.rule_types.weight[: rule_model.num_relation_rule_types, 0].detach().cpu().numpy().copy()
            if hasattr(rule_model, "rule_types") and rule_model.num_relation_rule_types > 0
            else np.array([], dtype=np.float32)
        )
        initial_rule_global_scale = (
            float((rule_model.rule_component_scale_raw.detach().reshape(-1)[0].cpu().item()) ** 2)
            if getattr(rule_model, "use_global_score_scales", False)
            else None
        )
        initial_dependency_global_scale = (
            float((rule_model.dependency_component_scale_raw.detach().reshape(-1)[0].cpu().item()) ** 2)
            if getattr(rule_model, "use_global_score_scales", False)
            else None
        )
        initial_bias_value = float(rule_model.bias.detach().reshape(-1)[0].cpu().item()) if hasattr(rule_model, "bias") else None
    test_stage_1 = evaluate_model_on_test(relation, rule_model.to(args.device), rule_model_builder)

    stage1_result = run_training_stage(
        relation=relation,
        model=rule_model,
        model_builder=rule_model_builder,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        pos=pos,
        lr_values=lr_values,
        eval_every_values=eval_every_values,
        max_epoch=max_epoch,
        stage_name="rule",
        checkpoint_selection="combined",
    )
    best_valid_stage1 = build_best_valid_metrics(stage1_result)
    best_valid_stage1_raw = build_best_valid_metrics(
        stage1_result,
        valid_key="selected_valid_raw",
        combined_key="best_valid_combined_raw",
        epoch_key="best_valid_epoch_raw",
    )
    export_model_per_query_rr(relation, stage1_result["model"], rule_model_builder, "stage1")

    final_result = stage1_result
    dependency_stage_result = None
    dependency_stage_accepted = None
    selection_reason = "rule stage only"
    selection_metric_name = "best_valid_mrr"
    stage1_metrics = {
        "pos_weight": float(pos),
        "pos_weight_source": pos_source,
        "epochs_trained": int(stage1_result["epochs_trained"]),
        "evaluate_every": int(stage1_result["evaluate_every"]),
        "checkpoint_selection": "filtered_best_valid",
        "best_valid_epoch": int(stage1_result["best_valid_epoch"]),
        "best_valid_epoch_raw": int(stage1_result["best_valid_epoch_raw"]),
        "best_valid_combined": float(stage1_result["best_valid_combined"]),
        "best_valid_combined_raw": float(stage1_result["best_valid_combined_raw"]),
    }
    best_valid_stage2 = None
    stage2_metrics = None
    initial_dependency_weights = np.array([], dtype=np.float32)
    initial_dependency_type_weights = np.array([], dtype=np.float32)
    dependency_pairs = []
    dependency_mask_info = {
        "enabled": False,
        "threshold_ratio": None,
        "threshold_value": None,
        "before": int(len(dependency_map.get(relation, []))),
        "after": int(len(dependency_map.get(relation, []))),
        "removed": 0,
    }
    dependency_topk_info = {
        "enabled": False,
        "topk": None,
        "per_kind_topk": None,
        "score_mode": str(getattr(args, "dependency_topk_score", "abs_lift")),
        "before": int(len(dependency_map.get(relation, []))),
        "after": int(len(dependency_map.get(relation, []))),
        "removed": 0,
    }
    test_stage_2 = stage1_result.get("selected_test", stage1_result["test"])
    test_stage_3 = None

    if (args.synergy or args.redundancy) and not getattr(args, "stage1_only", False):
        selected_stage1_state_dict = stage1_result.get("selected_state_dict")
        relation_dependencies_for_stage2, dependency_mask_info = filter_relation_dependencies_by_rule_strength(
            relation,
            dependency_map.get(relation, []),
            selected_stage1_state_dict,
        )
        relation_dependencies_for_stage2, dependency_topk_info = filter_relation_dependencies_topk_per_rule(
            relation_dependencies_for_stage2,
            getattr(args, "dependency_topk_per_rule", 0),
            getattr(args, "dependency_topk_score", "abs_lift"),
            getattr(args, "dependency_topk_per_kind", 0),
        )
        dependency_model_builder = partial(build_model_for_relation, relation_dependencies=relation_dependencies_for_stage2)
        dependency_model = dependency_model_builder(relation)
        if selected_stage1_state_dict is not None:
            copy_rule_state_from_state_dict(selected_stage1_state_dict, dependency_model)
        else:
            copy_rule_state_from_model(stage1_result["model"], dependency_model)
        if not getattr(args, "train_rule_in_dependency_stage", False):
            freeze_rule_parameters_for_synergy_stage(dependency_model)
        dependency_model = dependency_model.to(args.device)

        dependency_pairs = list(getattr(dependency_model, "relation_dependency_pairs_global", []))
        has_dependency_features = (
            getattr(dependency_model, "num_relation_dependencies", 0) > 0
            or getattr(dependency_model, "num_relation_dependency_types", 0) > 0
        )
        if has_dependency_features:
            with torch.no_grad():
                if getattr(dependency_model, "num_relation_dependencies", 0) > 0:
                    initial_dependency_weights = (
                        dependency_model.dependencies.weight[: dependency_model.num_relation_dependencies, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                if getattr(dependency_model, "num_relation_dependency_types", 0) > 0:
                    initial_dependency_type_weights = (
                        dependency_model.dependency_types.weight[: dependency_model.num_relation_dependency_types, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
            test_stage_3 = evaluate_model_on_test(relation, dependency_model, dependency_model_builder)

            dependency_stage_result = run_dependency_stage(
                relation=relation,
                model=dependency_model,
                model_builder=dependency_model_builder,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                pos=pos,
                lr_values=dependency_lr_values,
                eval_every_values=dependency_eval_every_values,
                max_epoch=dependency_max_epoch,
                early_stopping_patience=dependency_early_stopping_patience,
                min_epochs_before_stop=dependency_min_epochs_before_stop,
            )
            best_valid_stage2 = build_best_valid_metrics(dependency_stage_result)
            stage2_metrics = {
                "pos_weight": float(pos),
                "pos_weight_source": pos_source,
                "epochs_trained": int(dependency_stage_result["epochs_trained"]),
                "evaluate_every": int(dependency_stage_result["evaluate_every"]),
                "max_epoch": int(dependency_max_epoch),
                "lr_schedule": [float(v) for v in dependency_lr_values],
                "checkpoint_selection": "head_tail_best_valid",
                "early_stopping_patience": int(dependency_early_stopping_patience),
                "min_epochs_before_stop": int(dependency_min_epochs_before_stop),
                "best_valid_epoch": int(dependency_stage_result["best_valid_epoch"]),
                "best_valid_combined_raw": float(dependency_stage_result["best_valid_combined_raw"]),
            }

            rule_best_valid_mrr = float(best_valid_stage1["mrr"])
            dependency_best_valid_mrr = float(best_valid_stage2["mrr"])
            dependency_stage_accepted = dependency_best_valid_mrr > rule_best_valid_mrr
            if dependency_stage_accepted:
                final_result = dependency_stage_result
                selection_reason = "accepted dependency stage because its best valid mrr exceeded the rule-only stage"
            else:
                final_result = stage1_result
                selection_reason = "rejected dependency stage because its best valid mrr did not exceed the rule-only stage"
        else:
            final_result = stage1_result
            dependency_stage_accepted = False
            selection_reason = "rejected dependency stage because no relation-local dependency features remained after filtering"

    nnm = final_result["model"]
    head_mrr = final_result["head_mrr"]
    tail_mrr = final_result["tail_mrr"]
    evaluate_every = final_result["evaluate_every"]
    epochs_trained = final_result["epochs_trained"]
    train_seconds = stage1_result["train_seconds"] + (
        0.0 if dependency_stage_result is None else dependency_stage_result["train_seconds"]
    )
    eval_seconds = stage1_result["eval_seconds"] + (
        0.0 if dependency_stage_result is None else dependency_stage_result["eval_seconds"]
    )
    test_stage_4 = final_result["test"] if dependency_stage_result is None else dependency_stage_result["test"]
    final_test_metrics = (
        dependency_stage_result["test"]
        if (dependency_stage_result is not None and final_result is dependency_stage_result)
        else stage1_result.get("selected_test", stage1_result["test"])
    )
    if per_query_export_dir() is not None and not getattr(args, "stage1_only", False):
        if dependency_stage_result is not None:
            export_model_per_query_rr(relation, dependency_stage_result["model"], dependency_model_builder, "stage2")
        else:
            export_model_per_query_rr(relation, stage1_result["model"], rule_model_builder, "stage2")

    learned_weights = []
    if len(relation_rule_ids) > 0:
        with torch.no_grad():
            trained_rule_weights = nnm.rules.weight[: nnm.num_relation_rules, 0].detach().cpu().numpy()
            learned_weights = list(
                zip(
                    relation_rule_ids,
                    [round(float(v), 7) for v in initial_rule_weights.tolist()],
                    [round(float(v), 7) for v in trained_rule_weights.tolist()],
                )
            )

    learned_rule_type_weights = build_rule_type_weight_rows(nnm, initial_rule_type_weights)

    learned_dependency_weights_trial = []
    dependency_weights_trial_model = dependency_stage_result["model"] if dependency_stage_result is not None else None
    if dependency_weights_trial_model is not None:
        learned_dependency_weights_trial = build_dependency_weight_rows(
            dependency_weights_trial_model,
            dependency_pairs,
            initial_dependency_weights,
        )

    learned_dependency_type_weights_trial = []
    if dependency_weights_trial_model is not None:
        learned_dependency_type_weights_trial = build_dependency_type_weight_rows(
            dependency_weights_trial_model,
            initial_dependency_type_weights,
        )

    learned_dependency_weights_final = []
    if dependency_stage_result is not None and final_result is dependency_stage_result:
        learned_dependency_weights_final = build_dependency_weight_rows(
            final_result["model"],
            dependency_pairs,
            initial_dependency_weights,
        )

    learned_dependency_type_weights_final = []
    if dependency_stage_result is not None and final_result is dependency_stage_result:
        learned_dependency_type_weights_final = build_dependency_type_weight_rows(
            final_result["model"],
            initial_dependency_type_weights,
        )

    with torch.no_grad():
        trained_bias_value = float(nnm.bias.detach().reshape(-1)[0].cpu().item()) if hasattr(nnm, "bias") else None

    num_test_samples = int(test_torch[test_torch[:, 1] == relation].shape[0])
    num_relation_rules = int(len(relation_rule_ids))
    num_relation_dependencies = int(len(dependency_pairs))
    num_relation_rule_types = int(getattr(nnm, "num_relation_rule_types", 0))
    num_relation_dependency_types = int(getattr(nnm, "num_relation_dependency_types", 0))
    num_relation_dependency_type_source_pairs = int(getattr(nnm, "num_relation_dependency_type_source_pairs", 0))

    relation_total_seconds = perf_counter() - relation_start_time
    other_seconds = relation_total_seconds - load_seconds - train_seconds - eval_seconds
    if other_seconds < 0:
        other_seconds = 0.0

    metrics = {
        "relation": int(relation),
        "num_test_samples": num_test_samples,
        "num_relation_rules": num_relation_rules,
        "num_relation_rule_types": num_relation_rule_types,
        "num_relation_synergy": num_relation_dependencies,
        "num_relation_dependencies": num_relation_dependencies,
        "num_relation_dependencies_before_rule_mask": int(dependency_mask_info["before"]),
        "num_relation_dependency_types": num_relation_dependency_types,
        "num_relation_dependency_type_source_pairs": num_relation_dependency_type_source_pairs,
        "num_relation_features": int(
            num_relation_rules + num_relation_rule_types + num_relation_dependencies + num_relation_dependency_types
        ),
        "train": {
            "pos_weight": float(pos),
            "pos_weight_source": pos_source,
            "num_positive_samples": int(num_train_positive_samples),
            "num_negative_samples": int(num_train_negative_samples),
            "max_epoch": int(max_epoch),
            "epochs_trained": int(epochs_trained),
            "evaluate_every": int(evaluate_every),
            "early_stopping_patience_eval_rounds": int(early_stopping_patience),
            "stage1_rule_only": stage1_metrics,
            "stage2_dependency_only": stage2_metrics,
        },
        "time_seconds": {
            "total": float(relation_total_seconds),
            "load_dataloaders": float(load_seconds),
            "train": float(train_seconds),
            "eval": float(eval_seconds),
            "other": float(other_seconds),
        },
        "best_valid_stage1": best_valid_stage1,
        "best_valid_stage1_raw": best_valid_stage1_raw,
        "best_valid_stage2": best_valid_stage2,
        "model_selection": {
            "selected_stage": (
                "dependency" if (dependency_stage_result is not None and final_result is dependency_stage_result) else "rule_only"
            ),
            "selection_metric": selection_metric_name,
            "dependency_stage_attempted": bool(dependency_stage_result is not None),
            "dependency_stage_accepted": (
                None if dependency_stage_result is None else bool(dependency_stage_accepted)
            ),
            "rule_best_valid_mrr": float(best_valid_stage1["mrr"]),
            "rule_best_valid_mrr_raw_selected": float(best_valid_stage1_raw["mrr"]),
            "dependency_best_valid_mrr": (
                None if best_valid_stage2 is None else float(best_valid_stage2["mrr"])
            ),
            "rule_best_valid_combined": float(stage1_result["best_valid_combined"]),
            "rule_best_valid_combined_raw": float(stage1_result["best_valid_combined_raw"]),
            "dependency_best_valid_combined_raw": (
                None if dependency_stage_result is None else float(dependency_stage_result["best_valid_combined_raw"])
            ),
            "reason": selection_reason,
        },
        "params": {
            "bias": {
                "original": initial_bias_value,
                "trained": trained_bias_value,
            },
            "global_scales": build_global_scale_metrics(
                nnm,
                initial_rule_global_scale,
                initial_dependency_global_scale,
            ),
            "dependency_rule_mask": dependency_mask_info,
            "dependency_topk_per_rule": dependency_topk_info,
            "rule_type_weights": learned_rule_type_weights,
            "dependency_type_weights_trial": learned_dependency_type_weights_trial,
            "dependency_type_weights_final": learned_dependency_type_weights_final,
        },
        "test_before_stage1": test_stage_1,
        "test_after_stage1": test_stage_2,
        "test_after_stage1_raw_selected": stage1_result.get("selected_test_raw"),
        "test_before_stage2": test_stage_3,
        "test_after_stage2": None if dependency_stage_result is None else test_stage_4,
        "test": final_test_metrics,
    }

    with step_timer("save_outputs"):
        save((compact_mrr_for_save(head_mrr), compact_mrr_for_save(tail_mrr)), args.experiment, f"mrr-{relation}.pkl")
        with open(f"{args.experiment}/metric-{relation}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        with open(f"{args.experiment}/weight-{relation}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ruleID", "original", "trained"])
            writer.writerows(learned_weights)

        if len(learned_dependency_weights_trial) > 0:
            with open(f"{args.experiment}/dependency-trial-{relation}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rule1ID",
                        "rule2ID",
                        "type",
                        "raw_original",
                        "raw_trained",
                        "effective_original",
                        "effective_trained",
                    ]
                )
                writer.writerows(learned_dependency_weights_trial)

        if len(learned_dependency_weights_final) > 0:
            with open(f"{args.experiment}/dependency-final-{relation}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "rule1ID",
                        "rule2ID",
                        "type",
                        "raw_original",
                        "raw_trained",
                        "effective_original",
                        "effective_trained",
                    ]
                )
                writer.writerows(learned_dependency_weights_final)

    # 显式释放 relation 级别对象，尽量降低长跑时显存峰值。
    del train_dataloader, dataloader, train_split
    del loss_fn, nnm
    del head_mrr, tail_mrr
    has_stage1_result = "stage1_result" in locals()
    has_final_result = "final_result" in locals()
    final_result_differs = has_stage1_result and has_final_result and (final_result is not stage1_result)

    if has_stage1_result:
        del stage1_result
    if final_result_differs:
        del final_result
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return metrics


def _get_all_relations():
    return list(range(dataset.num_relations()))


def _aggregate_single_and_cleanup(relation):
    try:
        return aggregate_single(relation)
    finally:
        clear_relation_processed_cache(relation)
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _get_relation_test_counts():
    relation_ids = test_torch[:, 1].long().cpu()
    counts = torch.bincount(relation_ids, minlength=dataset.num_relations())
    return {int(i): int(c) for i, c in enumerate(counts.tolist())}


def _discover_completed_relations():
    completed = set()
    metric_files = sorted(glob.glob(os.path.join(args.experiment, "metric-*.json")))
    for path in metric_files:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            relation = int(payload["relation"])
        except Exception:
            # Corrupt / partial metric files should be recomputed.
            continue
        completed.add(relation)
    return completed


def _relations_remaining_for_sweep(relations):
    if not getattr(args, "resume_relation_sweep", False):
        return list(relations), set()
    completed_relations = _discover_completed_relations()
    remaining_relations = [relation for relation in relations if relation not in completed_relations]
    skipped = len(relations) - len(remaining_relations)
    print(
        f"Resume mode enabled: completed_relations={len(completed_relations)}, "
        f"remaining_relations={len(remaining_relations)}, skipped={skipped}"
    )
    return remaining_relations, completed_relations


def _merge_metric_files(metric_files, relation_test_counts):
    rows_by_stage = {
        "test_before_stage1": [],
        "test_after_stage1": [],
        "test_after_stage1_raw_selected": [],
        "test_before_stage2": [],
        "test_after_stage2": [],
    }

    def append_stage_row(stage_key, metrics_obj, relation, count):
        if metrics_obj is None:
            return
        rows_by_stage[stage_key].append(
            {
                "relation": relation,
                "count": count,
                "mrr": float(metrics_obj["mrr"]),
                "h1": float(metrics_obj["h1"]),
                "h10": float(metrics_obj["h10"]),
                "mrr_raw": float(metrics_obj["mrr_raw"]),
                "h1_raw": float(metrics_obj["h1_raw"]),
                "h10_raw": float(metrics_obj["h10_raw"]),
            }
        )

    for path in metric_files:
        with open(path, "r") as f:
            m = json.load(f)
        relation = int(m["relation"])
        count = relation_test_counts.get(relation, 0)
        test_before_stage1 = m.get("test_before_stage1", m.get("test_stage_1_rule_init", m.get("test_initial")))
        test_after_stage1 = m.get("test_after_stage1", m.get("test_stage_2_rule_trained"))
        test_after_stage1_raw_selected = m.get("test_after_stage1_raw_selected")
        # Keep summary denominators comparable across stages by carrying forward the
        # stage-1 result for relations that never entered stage 2.
        test_before_stage2 = m.get("test_before_stage2", m.get("test_stage_3_synergy_init")) or test_after_stage1
        test_after_stage2 = m.get("test_after_stage2", m.get("test_stage_4_synergy_trained")) or test_before_stage2
        append_stage_row("test_before_stage1", test_before_stage1, relation, count)
        append_stage_row("test_after_stage1", test_after_stage1, relation, count)
        append_stage_row("test_after_stage1_raw_selected", test_after_stage1_raw_selected, relation, count)
        append_stage_row("test_before_stage2", test_before_stage2, relation, count)
        append_stage_row("test_after_stage2", test_after_stage2, relation, count)

    if not rows_by_stage["test_after_stage2"] and not rows_by_stage["test_after_stage1"]:
        return {
            "num_relations": 0,
            "macro": {},
            "weighted_by_test_triples": {},
        }

    keys = ["mrr", "h1", "h10", "mrr_raw", "h1_raw", "h10_raw"]
    weighted_by_stage = {}
    total_weight = 0
    num_relations = 0
    for stage_key, stage_rows in rows_by_stage.items():
        weighted_rows = [r for r in stage_rows if r["count"] > 0]
        stage_total_weight = sum(r["count"] for r in weighted_rows)
        if stage_total_weight > 0:
            weighted_by_stage[stage_key] = {
                k: float(sum(r[k] * r["count"] for r in weighted_rows) / stage_total_weight) for k in keys
            }
        else:
            weighted_by_stage[stage_key] = None
        if stage_key in ["test_after_stage1", "test_after_stage2"] and stage_total_weight > total_weight:
            total_weight = stage_total_weight
        num_relations = max(num_relations, len(stage_rows))

    return {
        "num_relations": int(num_relations),
        "test_before_stage1": weighted_by_stage["test_before_stage1"],
        "test_after_stage1": weighted_by_stage["test_after_stage1"],
        "test_after_stage1_raw_selected": weighted_by_stage["test_after_stage1_raw_selected"],
        "test_before_stage2": weighted_by_stage["test_before_stage2"],
        "test_after_stage2": weighted_by_stage["test_after_stage2"],
        "test": weighted_by_stage["test_after_stage2"] or weighted_by_stage["test_after_stage1"],
        "total_test_triples_used_for_weight": int(total_weight),
    }


def _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds):
    metric_files = sorted(glob.glob(os.path.join(args.experiment, "metric-*.json")))
    merged = _merge_metric_files(metric_files, relation_test_counts)

    time_keys = ["total", "load_dataloaders", "train", "eval", "other"]
    summed_time_seconds = {k: 0.0 for k in time_keys}
    for path in metric_files:
        with open(path, "r") as f:
            m = json.load(f)
        metric_time = m.get("time_seconds", {})
        for k in time_keys:
            summed_time_seconds[k] += float(metric_time.get(k, 0.0))

    summed_time_seconds["sweep"] = float(sweep_seconds)

    final_result = {
        "experiment": args.experiment,
        "model": MODEL_NAME,
        "dataset": args.dataset,
        "failed_relations": failed_relations,
        "summary": merged,
        "time_seconds": summed_time_seconds,
    }

    out_path = os.path.join(args.experiment, "metrics-final.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"Finished relation sweep. failed={len(failed_relations)}")
    print(f"Final summary saved to {out_path}")
    return final_result


def aggregate_all_relations_sequential():
    sweep_start_time = perf_counter()
    relations = _get_all_relations()
    relations, completed_relations = _relations_remaining_for_sweep(relations)
    relation_test_counts = _get_relation_test_counts()

    print(
        f"Start relation sweep (sequential), remaining relations: {len(relations)}, "
        f"completed relations reused: {len(completed_relations)}"
    )

    failed_relations = {}

    for relation in relations:
        try:
            _aggregate_single_and_cleanup(relation)
        except Exception as e:
            failed_relations[int(relation)] = str(e)

    sweep_seconds = perf_counter() - sweep_start_time
    return _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds)


def _run_one_relation(relation):
    # Pool workers are daemonic; they cannot spawn children.
    # So DataLoader must run in-process in each worker.
    args.max_worker_dataloader = 0
    _aggregate_single_and_cleanup(relation)
    return int(relation)


def aggregate_multiple():
    sweep_start_time = perf_counter()
    # 在多进程 worker 内强制 DataLoader 单进程加载（num_workers=0）
    args.max_worker_dataloader = 0

    relations = _get_all_relations()
    relations, completed_relations = _relations_remaining_for_sweep(relations)
    deferred_large_relations = [relation for relation in relations if is_large_relation(relation)]
    pooled_relations = [relation for relation in relations if not is_large_relation(relation)]
    relation_test_counts = _get_relation_test_counts()
    num_processes = min(max(int(args.multiprocess), 2), len(pooled_relations)) if pooled_relations else 0

    print(
        f"Start relation sweep, remaining relations: {len(relations)}, reused completed relations: {len(completed_relations)}, "
        f"pooled relations: {len(pooled_relations)}, deferred large relations: {len(deferred_large_relations)}, "
        f"processes: {num_processes}"
    )
    if deferred_large_relations:
        print(
            "Deferred large relations (sequential after pooled run): "
            + ", ".join(
                f"{relation}[rules={get_relation_rule_count(relation)}]"
                for relation in deferred_large_relations
            )
        )

    failed_relations = {}

    if pooled_relations:
        with mp.get_context("spawn").Pool(processes=num_processes) as pool:
            results = [pool.apply_async(_run_one_relation, (relation,)) for relation in pooled_relations]
            for relation, result in zip(pooled_relations, results):
                try:
                    result.get()
                except Exception as e:
                    failed_relations[int(relation)] = str(e)

    for relation in deferred_large_relations:
        try:
            _aggregate_single_and_cleanup(relation)
        except Exception as e:
            failed_relations[int(relation)] = str(e)

    sweep_seconds = perf_counter() - sweep_start_time
    return _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds)

MODEL_NAME = "LinearAggregator"

args = get_parser().parse_args()
type_grouping_config = resolve_type_grouping(getattr(args, "type_grouping", "none"))
args.type_grouping = type_grouping_config["type_grouping"]
args.rule_grouping = type_grouping_config["rule_grouping"]
args.dependency_grouping = type_grouping_config["dependency_grouping"]
args.use_global_score_scales = bool(type_grouping_config["use_global_score_scales"])
args.dependency_mask_rule_weight_threshold_ratio = float(DEPENDENCY_MASK_RULE_WEIGHT_THRESHOLD_RATIO)
EVAL_DEVICE = torch.device(args.device)
dataset_dir = os.path.join(args.data_root, args.dataset)
args.directory_explanations = f"./{dataset_dir}/application/"
args.directory_preprocessed_datasets = f"./{dataset_dir}/datasets/"
if "EXPERIMENT_DIR" not in os.environ:
    sign_bit = 1 if args.sign_constraint else 0
    sign_dependency_bit = 1 if args.sign_constraint_dependency else 0
    dependency_bit = 1 if (args.synergy or args.redundancy) else 0
    exp_name = (
        f"exp{args.relation}_{MODEL_NAME}_{sign_bit}_{sign_dependency_bit}_{dependency_bit}_{args.pos}_"
        f"{int(args.synergy)}_{int(args.redundancy)}_{int(args.init_dep_with_lift)}_"
        f"tg_{args.type_grouping}_"
        f"ri_{args.rule_init_mode}_ds_{args.dependency_scale_mode}_"
        f"dm_{int(args.dependency_mask_low_rule_weight)}_{args.dependency_mask_rule_weight_threshold_ratio}_"
        f"dn_{args.dependency_static_norm}_dl1_{args.dep_l1_lambda}_"
        f"dc_{args.dep_score_clip_gamma}_dtk_{args.dependency_topk_per_rule}_dtpk_{args.dependency_topk_per_kind}"
    )
    os.environ["EXPERIMENT_DIR"] = f"./{dataset_dir}/aggregation/{exp_name}"
args.experiment = os.environ["EXPERIMENT_DIR"]

# Set up experiment folder
if not os.path.exists(args.experiment):
    os.makedirs(args.experiment)
# Copy stuff for reproducibility
shutil.copy(__file__, args.experiment)
with open(f"{args.experiment}/config.json", "w") as f:
    json.dump(vars(args), f, indent=4)

dataset = LocalDataset(dataset_dir)

test_sp_to_o = dataset.index("test_sp_to_o")
test_po_to_s = dataset.index("test_po_to_s")
test_torch = dataset.split("test")

valid_sp_to_o = dataset.index("valid_sp_to_o")
valid_po_to_s = dataset.index("valid_po_to_s")

print("Loading processed explanations...")
print(f"Using relation-local processed explanations root: {get_relation_processed_root()}")

rule_file = args.rule_file if args.rule_file else f"./{dataset_dir}/rules/rules-1000-5"
dependency_dir = os.path.dirname(rule_file)
synergy_filtered_file = args.synergy_file if getattr(args, "synergy_file", "") else os.path.join(dependency_dir, "synergy_filtered.txt")
redundancy_filtered_file = args.redundancy_file if getattr(args, "redundancy_file", "") else os.path.join(dependency_dir, "redundancy_filtered.txt")
relation_ids = read_ids(f"./{dataset_dir}/relation_ids.del")
rule_meta = parse_rule_file_metadata(rule_file, relation_ids)
rule_type_r3_by_id = rule_meta["rule_type_r3_by_id"]
rule_type_members_r3_by_relation = rule_meta["rule_type_members_r3_by_relation"]
dependency_map = defaultdict(list)
if args.synergy:
    for relation, deps in parse_filtered_dependency_file(
        synergy_filtered_file, rule_meta["rule_relation_by_id"], "synergy"
    ).items():
        dependency_map[int(relation)].extend(deps)
if args.redundancy:
    for relation, deps in parse_filtered_dependency_file(
        redundancy_filtered_file, rule_meta["rule_relation_by_id"], "redundancy"
    ).items():
        dependency_map[int(relation)].extend(deps)
dependency_map = dict(dependency_map)
dependency_type_candidate_map = {
    int(relation): sorted({(int(a), int(b)) for a, b, *_rest in deps})
    for relation, deps in dependency_map.items()
}
dependency_type_count_by_relation = summarize_dependency_type_candidates(dependency_type_candidate_map)

LEN_RULES = rule_meta["num_rules"]
MAX_RULE_ID = rule_meta["max_rule_id"]
PAD_TOK = MAX_RULE_ID + 1
rule_map = rule_meta["rule_map"]

# 优化点：预构建规则置信度查表，替代 eval 阶段的 np.vectorize(get_conf) 重复计算。
# 约定最后一个位置 PAD_TOK 的置信度为 0。
rule_conf_values = [0.0] * (PAD_TOK + 1)
for rule_id, conf in rule_meta["rule_conf_by_id"].items():
    rid = int(rule_id)
    if rid < 0 or rid >= PAD_TOK:
        continue
    rule_conf_values[rid] = float(conf)

# PAD_TOK 的置信度保留为 0.0
RULE_CONF_TABLE_CPU = torch.tensor(rule_conf_values, dtype=torch.float32)
if EVAL_DEVICE.type == "cpu":
    raise RuntimeError("This version uses GPU-only eval. Please run with --device cuda")
RULE_CONF_TABLE = RULE_CONF_TABLE_CPU.to(EVAL_DEVICE)

# 优化点：预构建 relation -> keys 索引，避免每次 get_ranks 线性扫描所有 keys。
print("Building relation key indices...")
relation_keys = {
    "valid_o": build_relation_key_index(valid_sp_to_o, direction="o"),
    "valid_s": build_relation_key_index(valid_po_to_s, direction="s"),
    "test_o": build_relation_key_index(test_sp_to_o, direction="o"),
    "test_s": build_relation_key_index(test_po_to_s, direction="s"),
}

if __name__ == "__main__":
    if args.dataset == "hetionet":
        args.multiprocess = 1
    if args.relation == -1:
        if args.multiprocess > 1:
            result = aggregate_multiple()
        else:
            result = aggregate_all_relations_sequential()
        print(json.dumps(result["summary"], indent=2))
    else:
        metrics = aggregate_single(args.relation)
        print(pformat(metrics))
    print_step_profile()
