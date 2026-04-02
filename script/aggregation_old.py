#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import ctypes
import json
import logging
import math
import os
import pickle
import re
import shutil
import uuid
import warnings
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from os.path import exists
from pprint import pformat

import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def save(obj, folder, name=None, override=False):
    if name is None:
        name = uuid.uuid4()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    if exists(path_to_file):
        print(f"Warning name {name} exists in cache, do you want to overwrite y/n?")
        confirm = input() if not override else "y"
        if confirm != "y":
            return None

    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def load(folder, name):
    path_to_file = f"{folder}/{name}.p"
    if exists(path_to_file):
        return pickle.load(open(path_to_file, "rb"))
    print("No such name in cache")
    return None


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
        self._triples = {}
        self._indexes = {}
        self._entity_ids = None
        self._relation_ids = None

    def _load_triples(self, split):
        triples = list(read_index_triples(os.path.join(self.folder, f"{split}.del")))
        if len(triples) == 0:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(triples, dtype=torch.long)

    def split(self, split):
        if split not in self._triples:
            self._triples[split] = self._load_triples(split)
        return self._triples[split]

    def _build_kvs_index(self, split, key_cols, value_col):
        buckets = defaultdict(list)
        for triple in self.split(split).tolist():
            key = tuple(int(triple[col]) for col in key_cols)
            buckets[key].append(int(triple[value_col]))
        return LocalKvsIndex({
            key: torch.tensor(values, dtype=torch.long)
            for key, values in buckets.items()
        })

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

    def num_entities(self):
        return len(self.entity_ids())

    def num_relations(self):
        return len(self.relation_ids())


def split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def extract_head_relation(rule_str: str):
    head = rule_str.split(" <= ", 1)[0].strip()
    if "(" not in head:
        return ""
    return head.split("(", 1)[0].strip()


def parse_rule_file_metadata(rule_file: str, relation_ids):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_map = defaultdict(list)
    rule_conf_by_id = {}
    num_rules = 0
    max_rule_id = 0

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
            rule_conf_by_id[line_no] = float(num_true / (num_preds + 5)) if num_preds >= 0 else 0.0

            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is not None:
                rule_map[int(rel_id)].append(int(line_no))

    return {
        "rule_map": dict(rule_map),
        "rule_conf_by_id": rule_conf_by_id,
        "num_rules": int(num_rules),
        "max_rule_id": int(max_rule_id),
    }


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
        if end > start:
            padded[i, : end - start] = rules_flat[start:end]

    return padded, ys


def train(dataloader, model, loss_fn, optimizer, relation, reg=False, num_unseen=0):
    model.train()
    train_loss = 0
    n_loss = 0
    for i, (rules, y) in enumerate(dataloader):
        if reg and num_unseen > 0:
            num_batches = len(dataloader)
            if num_unseen > num_batches:
                num_unseen = num_batches
            every = max(int(num_batches / max(num_unseen, 1)), 1)
            if i % every == 0:
                rule_confs = torch.nn.functional.sigmoid(model.rules.weight)
                sudo_false = torch.zeros_like(rule_confs)
                reg_loss = loss_fn(rule_confs, sudo_false) / dataloader.batch_size
                optimizer.zero_grad()
                reg_loss.backward()
                optimizer.step()

        rules = rules.long().to(args.device)
        y = y.to(args.device)
        pred = model(rules, relation)
        loss = loss_fn(pred.reshape(-1, 1), y)

        train_loss += loss.item()
        n_loss += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if n_loss == 0:
        return 0.0
    return train_loss / n_loss


def test(dataloader, model, loss_fn, relation):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for rules, y in dataloader:
            rules = rules.long().to(args.device)
            y = y.to(args.device)

            pred = model(rules, relation).reshape(-1, 1)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()

    test_loss /= max(num_batches, 1)
    correct /= max(size, 1)
    logging.info(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss


def rank_batch(nnm, golds, candidates, rules, test_filter, relation):
    batch_rank, batch_rank_raw = [], []
    if len(candidates) > 0 and len(rules) > 0:
        fill_value = 0.0
        scores = torch.full((dataset.num_entities(),), fill_value)
        scores_raw = torch.full((dataset.num_entities(),), fill_value)

        rules_ = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor([torch.tensor(x) for x in rules]),
            padding=PAD_TOK,
        ).long()
        if rules_.numel() == 0:
            return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)

        with torch.no_grad():
            pred = nnm(rules_, relation).detach()
            if args.model != "NoisyOrAggregator":
                pred = torch.sigmoid(pred).detach()

        max_conf = RULE_CONF_TABLE_CPU[rules_.cpu()].max(dim=1, keepdim=True).values.float()
        scores[candidates] = (pred.cpu() * max_conf).squeeze(dim=1)
        scores_raw[candidates] = pred.cpu().squeeze(dim=1)

        def get_rank(local_scores):
            local_batch_rank = []
            local_scores = -1 * local_scores
            gold_scores = local_scores[golds].clone()
            local_scores[golds] = fill_value
            if test_filter is not None:
                local_scores[test_filter] = fill_value
            for ix, gold in enumerate(golds):
                gold = gold.item()
                local_scores[gold] = gold_scores[ix]
                ranking = scipy.stats.rankdata(local_scores.detach().numpy())
                local_batch_rank.append(ranking[gold])
                local_scores[gold] = fill_value
            return local_batch_rank

        batch_rank = get_rank(scores)
        batch_rank_raw = get_rank(scores_raw)

    return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False):
    nnm.eval()
    if direction == "o":
        keys = [key for key in sp_to_o.keys() if key[1] == relation]
    else:
        keys = [key for key in sp_to_o.keys() if key[0] == relation]

    if len(keys) == 0:
        return torch.tensor([]), torch.tensor([]), 0

    batch_rank = []
    batch_rank_raw = []
    total = 0
    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o" and key in test_sp_to_o:
                test_filter = test_sp_to_o[key].long()
            elif direction == "s" and key in test_po_to_s:
                test_filter = test_po_to_s[key].long()

        golds = sp_to_o[key].long()
        candidates = []
        rules = []
        if key in processed:
            candidates = processed[key]["candidates"]
            rules = processed[key]["rules"]
        rank, rank_raw, n = rank_batch(nnm, golds, candidates, rules, test_filter, relation)
        batch_rank.append(rank)
        batch_rank_raw.append(rank_raw)
        total += n

    if len(batch_rank) == 0:
        return torch.tensor([]), torch.tensor([]), 0
    return torch.hstack(batch_rank), torch.hstack(batch_rank_raw), total


class LinearAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            for r in rule_map:
                rules = torch.tensor(rule_map[r], dtype=torch.long)
                confs = RULE_CONF_TABLE_CPU[rules].reshape(-1, 1)
                self.rules.weight[rules] = confs.float()
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[rules].reshape(1, -1))
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias[r] = self.bias[r].uniform_(-bound, bound)

    def __init__(self, sign_constraint=False):
        super().__init__()
        self.rules = nn.Embedding(NUM_RULE_TOKENS, 1, padding_idx=PAD_TOK)
        self.bias = nn.Parameter(torch.zeros(dataset.num_relations(), 1))
        self.init_weights()
        self.sign_constraint = sign_constraint

    def forward(self, rules, relation):
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            rules = rules**2
        return rules.sum(dim=1) + self.bias[int(relation)]


class NoisyOrAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            for r in rule_map:
                rules = torch.tensor(rule_map[r], dtype=torch.long)
                confs = RULE_CONF_TABLE_CPU[rules].reshape(-1, 1).float().clamp(min=1e-6, max=1 - 1e-6)
                self.rules.weight[rules] = torch.log(confs / (1 - confs)).float()

    def __init__(self):
        super().__init__()
        self.rules = nn.Embedding(NUM_RULE_TOKENS, 1, padding_idx=PAD_TOK)
        self.init_weights()

    def forward(self, rules, relation):
        del relation
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), float("-inf"))
        no = 1 - (1 - torch.sigmoid(rules)).prod(dim=1)
        return no.clamp(min=0.0001, max=0.99999)


def calc_mrr(tail_mrr, head_mrr, relations, attr="maximums_t"):
    head_rank = 0.0
    tail_rank = 0.0
    head_rank_raw = 0.0
    tail_rank_raw = 0.0
    n = 0
    for relation in relations:
        rn = int((test_torch[:, 1] == int(relation)).sum().item())
        tail_rank += getattr(tail_mrr, attr).get(relation, 0.0) * rn
        head_rank += getattr(head_mrr, attr).get(relation, 0.0) * rn
        tail_rank_raw += getattr(tail_mrr, attr + "_raw").get(relation, 0.0) * rn
        head_rank_raw += getattr(head_mrr, attr + "_raw").get(relation, 0.0) * rn
        n += rn
    if n == 0:
        return 0.0, 0.0
    return (head_rank + tail_rank) / (2 * n), (head_rank_raw + tail_rank_raw) / (2 * n)


class MRR:
    def __init__(self, direction="o"):
        self.direction = direction

        self.best_hps = {}
        self.best_hps_raw = {}

        self.maximums_v = defaultdict(float)
        self.maximums_v_raw = defaultdict(float)

        self.maximums_t = defaultdict(float)
        self.maximums_t_raw = defaultdict(float)
        self.maximums_t_1 = defaultdict(float)
        self.maximums_t_1_raw = defaultdict(float)
        self.maximums_t_10 = defaultdict(float)
        self.maximums_t_10_raw = defaultdict(float)

        self.valid_sp_to_o = valid_sp_to_o if direction == "o" else valid_po_to_s
        self.valid_processed = processed_sp_valid if direction == "o" else processed_po_valid
        self.test_sp_to_o = test_sp_to_o if direction == "o" else test_po_to_s
        self.test_processed = processed_sp_test if direction == "o" else processed_po_test
        self.nnm = {}
        self.nnm_raw = {}

    def calc_metrics_(self, ranks, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        mrr = ((1 / ranks).sum() / n).item()
        h1 = ((ranks == 1.0).sum() / n).item()
        h10 = ((ranks <= 10.0).sum() / n).item()
        return mrr, h1, h10

    def calc_metrics(self, nnm, sp_to_o, processed, relation, direction, filter_test=False):
        ranks, ranks_raw, n = get_ranks(nnm, sp_to_o, processed, relation, direction, filter_test)
        mrr, h1, h10 = self.calc_metrics_(ranks, n)
        mrr_raw, h1_raw, h10_raw = self.calc_metrics_(ranks_raw, n)
        return mrr, h1, h10, mrr_raw, h1_raw, h10_raw

    def update(self, nnm, relation, hps):
        v_mrr, v_h1, v_h10, v_mrr_raw, v_h1_raw, v_h10_raw = self.calc_metrics(
            nnm,
            self.valid_sp_to_o,
            self.valid_processed,
            relation,
            direction=self.direction,
            filter_test=True,
        )
        if (relation not in self.best_hps) or (v_mrr > self.maximums_v[relation]):
            self.maximums_v[relation] = v_mrr
            self.nnm[relation] = copy.deepcopy(nnm)
            self.best_hps[relation] = hps

        if (relation not in self.best_hps_raw) or (v_mrr_raw > self.maximums_v_raw[relation]):
            self.maximums_v_raw[relation] = v_mrr_raw
            self.nnm_raw[relation] = copy.deepcopy(nnm)
            self.best_hps_raw[relation] = hps

        return {
            "valid_mrr": float(v_mrr),
            "valid_h1": float(v_h1),
            "valid_h10": float(v_h10),
            "valid_mrr_raw": float(v_mrr_raw),
            "valid_h1_raw": float(v_h1_raw),
            "valid_h10_raw": float(v_h10_raw),
            "best_valid_mrr": float(self.maximums_v[relation]),
            "best_valid_mrr_raw": float(self.maximums_v_raw[relation]),
        }

    def finalize_test(self, relations):
        for relation in relations:
            if relation in self.nnm:
                t_mrr, t_h1, t_h10, _t_mrr_raw_unused, _t_h1_raw_unused, _t_h10_raw_unused = self.calc_metrics(
                    self.nnm[relation],
                    self.test_sp_to_o,
                    self.test_processed,
                    relation,
                    direction=self.direction,
                )
                self.maximums_t[relation] = t_mrr
                self.maximums_t_1[relation] = t_h1
                self.maximums_t_10[relation] = t_h10

            if relation in self.nnm_raw:
                _t_mrr_unused, _t_h1_unused, _t_h10_unused, t_mrr_raw, t_h1_raw, t_h10_raw = self.calc_metrics(
                    self.nnm_raw[relation],
                    self.test_sp_to_o,
                    self.test_processed,
                    relation,
                    direction=self.direction,
                )
                self.maximums_t_raw[relation] = t_mrr_raw
                self.maximums_t_1_raw[relation] = t_h1_raw
                self.maximums_t_10_raw[relation] = t_h10_raw


class SharedDataset(Dataset):
    def get_empty_shared_array(self, shape, type_):
        shared_array_base = torch.multiprocessing.Array(type_, torch.tensor(shape).prod().item())
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        return torch.from_numpy(shared_array)

    def __init__(self, xs, ys):
        self.shared_x = self.get_empty_shared_array(xs.shape, ctypes.c_int)
        self.shared_x[:] = xs
        self.shared_y = self.get_empty_shared_array(ys.shape, ctypes.c_float)
        self.shared_y[:] = ys
        self.len = xs.shape[0]

    def __getitem__(self, index):
        return self.shared_x[index], self.shared_y[index]

    def __len__(self):
        return self.len


def load_dataloader(dataset_directory, relation):
    data_obj = load(dataset_directory, f"dataset_{relation}")
    if data_obj is None:
        return None
    if not (isinstance(data_obj, dict) and data_obj.get("format") == "compact_varlen_int32_v1"):
        raise ValueError(
            f"dataset_{relation}.p is not compact_varlen_int32_v1. "
            "Please regenerate dataset_*.p with the current create_datasets.py"
        )

    train_split = data_obj["train"]
    xs, ys = materialize_compact_split_to_padded(train_split)
    if xs.shape[0] == 0:
        return None

    train_set = SharedDataset(xs, ys)
    return DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=args.shuffle_train,
        num_workers=max(int(args.max_worker_dataloader), 0),
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        help="Path to config file; ORDER: default->command line->config file",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        help="Name of experiment",
        default=None,
    )
    parser.add_argument("-d", "--dataset", action="store", help="Dataset name", default="codex-m")
    parser.add_argument("--data_root", action="store", help="Dataset root directory", default="data")
    parser.add_argument("-dev", "--device", action="store", help="Device cpu/cuda", default="cuda")
    parser.add_argument(
        "--max_worker_dataloader",
        action="store",
        help="Number of processes for dataloader",
        default=max(len(os.sched_getaffinity(0)) - 1, 0),
        type=int,
    )
    parser.add_argument(
        "--directory_explanations",
        action="store",
        help="Folder containing processed_sp_*.pkl and processed_po_*.pkl",
        default=None,
    )
    parser.add_argument(
        "--directory_preprocessed_datasets",
        action="store",
        help="Directory containing compact train datasets from create_datasets.py",
        default=None,
    )
    parser.add_argument(
        "--rule_file",
        action="store",
        help="Path to rule.txt; default: <data_root>/<dataset>/rules/rule.txt",
        default="",
    )
    parser.add_argument(
        "--model",
        action="store",
        help="Aggregator to use; one of ['LinearAggregator', 'NoisyOrAggregator']",
        default="LinearAggregator",
    )
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffle training examples")
    parser.add_argument("--batch_size", action="store", help="Batch size", default=4096, type=int)
    parser.add_argument(
        "--lr_hpo",
        action="store",
        nargs="+",
        type=float,
        default=[0.001, 0.01],
        help="Learning rates of the adam optimizer",
    )
    parser.add_argument(
        "--max_epoch_hpo",
        action="store",
        nargs="+",
        type=int,
        default=[20, 10],
        help="Epochs to run for each learning rate; max_epoch[i] is trained using lr[i]",
    )
    parser.add_argument(
        "--pos_hpo",
        action="store",
        nargs="+",
        type=float,
        default=[5, 15, 30, 100, 400],
        help="Scaling of the loss for positive examples",
    )
    parser.add_argument(
        "--sign_constraint",
        action="store_true",
        help="Constrains the rule weights to be >=0. Only implemented for LinearAggregator.",
    )
    parser.add_argument("--noisy_or_reg", action="store_true", help="Sudo negative examples for noisy-or learning.")
    parser.add_argument(
        "--num_unseen",
        action="store",
        nargs="+",
        type=int,
        default=[0],
        help="Num sudo negative-example regularization passes for noisy-or learning.",
    )
    parser.add_argument(
        "--relation",
        action="store",
        type=int,
        default=-1,
        help="Relation id to train on; -1 means train/evaluate all relations.",
    )
    return parser


def BCELossR(weights=(1, 1), reduction="mean", apply_sigmoid=False):
    def loss(input, target):
        if apply_sigmoid:
            input = torch.sigmoid(input)
            input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        if reduction == "libkge":
            pos_mask = target.bool()
            neg_mask = ~pos_mask
            pos_term = bce[pos_mask].sum() / max(int(pos_mask.sum().item()), 1)
            neg_term = bce[neg_mask].sum() / max(int(neg_mask.sum().item()), 1)
            bce = (pos_term + neg_term) / 2.0
        elif reduction == "sum":
            bce = torch.sum(bce)
        elif reduction == "mean":
            bce = torch.mean(bce)
        return bce

    return loss


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        args_dict = vars(args)
        assert set(config.keys()).issubset(args_dict.keys()), "There are keys in your config file not recognized"
        args = Namespace(**{**args_dict, **config})

    dataset_dir = os.path.join(args.data_root, args.dataset)
    if args.experiment is None:
        args.experiment = os.path.join(
            dataset_dir,
            "aggregation",
            f"exp_old_r{args.relation}_{args.model}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        )
    if args.directory_explanations is None:
        args.directory_explanations = os.path.join(dataset_dir, "application") + "/"
    if args.directory_preprocessed_datasets is None:
        args.directory_preprocessed_datasets = os.path.join(dataset_dir, "datasets") + "/"
    if args.rule_file == "":
        preferred_rule_file = os.path.join(dataset_dir, "rules", "rule.txt")
        fallback_rule_file = os.path.join(dataset_dir, "rules", "rules-1000-5")
        args.rule_file = preferred_rule_file if os.path.exists(preferred_rule_file) else fallback_rule_file

    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)
    shutil.copy(__file__, args.experiment)
    with open(f"{args.experiment}/config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        filename=f"{args.experiment}/{os.path.basename(args.experiment)}.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(ch)
    logging.info(f"Starting experiment {args.experiment}")
    logging.info(pformat(vars(args)))

    dataset = LocalDataset(dataset_dir)
    if args.relation == -1:
        selected_relations = list(range(dataset.num_relations()))
    else:
        if args.relation < 0 or args.relation >= dataset.num_relations():
            raise ValueError(f"--relation out of range: {args.relation}, valid range is [0, {dataset.num_relations() - 1}]")
        selected_relations = [int(args.relation)]
    logging.info(f"Selected relations: {selected_relations[:10]}{'...' if len(selected_relations) > 10 else ''}")

    test_sp_to_o = dataset.index("test_sp_to_o")
    test_po_to_s = dataset.index("test_po_to_s")
    test_torch = dataset.split("test")

    valid_sp_to_o = dataset.index("valid_sp_to_o")
    valid_po_to_s = dataset.index("valid_po_to_s")

    rule_meta = parse_rule_file_metadata(args.rule_file, dataset.relation_ids())
    rule_map = rule_meta["rule_map"]
    MAX_RULE_ID = rule_meta["max_rule_id"]
    PAD_TOK = MAX_RULE_ID + 1
    NUM_RULE_TOKENS = PAD_TOK + 1

    RULE_CONF_TABLE_CPU = torch.zeros((NUM_RULE_TOKENS,), dtype=torch.float32)
    for rule_id, conf in rule_meta["rule_conf_by_id"].items():
        RULE_CONF_TABLE_CPU[int(rule_id)] = float(conf)

    dataloaders = {}
    for relation in selected_relations:
        dataloaders[relation] = load_dataloader(args.directory_preprocessed_datasets, relation)

    processed_sp_test = pickle.load(open(args.directory_explanations + "processed_sp_test.pkl", "rb"))
    processed_po_test = pickle.load(open(args.directory_explanations + "processed_po_test.pkl", "rb"))
    processed_sp_valid = pickle.load(open(args.directory_explanations + "processed_sp_valid.pkl", "rb"))
    processed_po_valid = pickle.load(open(args.directory_explanations + "processed_po_valid.pkl", "rb"))

    if len(args.lr_hpo) != len(args.max_epoch_hpo):
        raise ValueError("--lr_hpo and --max_epoch_hpo must have the same length")

    for pos in args.pos_hpo:
        for unseen in args.num_unseen:
            for lr, max_epoch in zip(args.lr_hpo, args.max_epoch_hpo):
                tail_mrr = MRR(direction="o")
                head_mrr = MRR(direction="s")
                logging.info(f"Pos weight: {pos}, Lr: {lr}, Max epoch: {max_epoch}, Unseen: {unseen}")

                if args.model == "LinearAggregator":
                    nnm = LinearAggregator(sign_constraint=args.sign_constraint)
                    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos), device=args.device))
                elif args.model == "NoisyOrAggregator":
                    nnm = NoisyOrAggregator()
                    loss_fn = BCELossR([1, float(pos)])
                else:
                    raise ValueError(f"Unknown --model: {args.model}")

                nnm = nnm.to(args.device)
                logging.info(nnm)
                optimizer = torch.optim.Adam(nnm.parameters(), lr=float(lr))

                for t in range(int(max_epoch)):
                    for relation in tqdm(selected_relations, desc=f"epoch {t + 1}/{max_epoch}", leave=False):
                        train_dataloader = dataloaders.get(relation)
                        if train_dataloader is None:
                            continue

                        loss = train(
                            train_dataloader,
                            nnm,
                            loss_fn,
                            optimizer,
                            relation,
                            args.noisy_or_reg,
                            int(unseen),
                        )
                        nnm.cpu()
                        head_valid_metrics = head_mrr.update(nnm, relation, (float(pos), float(lr), int(t)))
                        tail_valid_metrics = tail_mrr.update(nnm, relation, (float(pos), float(lr), int(t)))
                        nnm.to(args.device)
                        max_tail_valid_mrr = tail_valid_metrics["best_valid_mrr_raw"]
                        max_head_valid_mrr = head_valid_metrics["best_valid_mrr_raw"]
                        logging.info(
                            f"{relation} tail loss: {loss:>7f} "
                            f"best_valid_tail_raw={max_tail_valid_mrr} "
                            f"best_valid_head_raw={max_head_valid_mrr} "
                            f"[{t:>5d}/{int(max_epoch):>5d}]"
                        )

                head_mrr.finalize_test(selected_relations)
                tail_mrr.finalize_test(selected_relations)
                logging.info(calc_mrr(tail_mrr, head_mrr, selected_relations))
                logging.info(calc_mrr(tail_mrr, head_mrr, selected_relations, "maximums_t_1"))
                logging.info(calc_mrr(tail_mrr, head_mrr, selected_relations, "maximums_t_10"))
                save(head_mrr, args.experiment, f"head_mrr_r{args.relation}_{pos}_{lr}", override=True)
                save(tail_mrr, args.experiment, f"tail_mrr_r{args.relation}_{pos}_{lr}", override=True)

    logging.info("Done")
