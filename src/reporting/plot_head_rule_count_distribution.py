#!/usr/bin/env python3
import os
import math
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/sy/RuleDep"
DATASETS = [
    "FB15k-237",
    "KG20C",
    "WN18RR",
    "YAGO3-10",
    "codex-l",
    "codex-m",
    "hetionet",
]
BIN_WIDTH = 5
OUT_DIR = os.path.join(ROOT, "reports", "0415", "head_rule_distribution")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_head(rule_line: str):
    if "<=" not in rule_line:
        return None
    return rule_line.split("<=", 1)[0].strip()


def head_kind(head: str):
    normalized = head.replace(" ", "")
    if "X" in normalized and "Y" in normalized:
        return "binary"
    return "unary"


def load_head_rule_counts(rule_file: str):
    head_counter = Counter()
    with open(rule_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            head = extract_head(parts[3])
            if head is None:
                continue
            head_counter[head] += 1
    counts_by_kind = {"unary": [], "binary": []}
    for head, count in head_counter.items():
        counts_by_kind[head_kind(head)].append(count)
    return counts_by_kind


def make_log2_bins(values):
    if not values:
        return np.array([], dtype=int), []
    vmax = max(values)
    max_exp = int(math.floor(math.log2(vmax))) if vmax > 0 else 0
    edges = [1]
    for exp in range(1, max_exp + 2):
        edges.append(2 ** exp)
    if edges[-1] <= vmax:
        edges.append(2 ** (max_exp + 2))

    hist = []
    labels = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if left == 1:
            mask = [v == 1 for v in values]
            label = "1"
        else:
            mask = [left <= v < right for v in values]
            label = f"{left}-{right - 1}"
        hist.append(int(sum(mask)))
        labels.append(label)

    return np.array(hist, dtype=int), labels


def plot_group(ax, dataset: str, group_name: str, values):
    if not values:
        ax.set_title(f"{group_name}: no heads")
        ax.axis("off")
        return 0.0

    hist, labels = make_log2_bins(values)
    x = np.arange(len(labels))
    ax.bar(x, hist, color="#4C72B0" if group_name == "unary" else "#DD8452", edgecolor="black", linewidth=0.5)
    ax.set_title(f"{group_name} heads (mean={np.mean(values):.2f}, n={len(values)})")
    ax.set_xlabel("Rule count per head (log2 bins)")
    ax.set_ylabel("Head atom count")
    tick_step = max(1, len(labels) // 14)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return float(np.mean(values))


def plot_one_dataset(dataset: str):
    rule_file = os.path.join(ROOT, "data", dataset, "rules", "rule.txt")
    if not os.path.exists(rule_file):
        print(f"[WARN] missing rule file: {rule_file}")
        return

    counts_by_kind = load_head_rule_counts(rule_file)
    all_counts = counts_by_kind["unary"] + counts_by_kind["binary"]
    if not all_counts:
        print(f"[WARN] no valid rules parsed: {dataset}")
        return

    fig_w = 14
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.5), constrained_layout=True)
    unary_mean = plot_group(axes[0], dataset, "unary", counts_by_kind["unary"])
    binary_mean = plot_group(axes[1], dataset, "binary", counts_by_kind["binary"])
    fig.suptitle(f"{dataset}: Rule Count per Head Atom by Head Type", y=1.03, fontsize=14)

    out_png = os.path.join(OUT_DIR, f"{dataset}_head_rule_count_distribution_split.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(OUT_DIR, f"{dataset}_head_rule_count_distribution_summary.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "head_type", "num_heads", "mean_rules_per_head"])
        writer.writerow([dataset, "unary", len(counts_by_kind["unary"]), f"{unary_mean:.6f}"])
        writer.writerow([dataset, "binary", len(counts_by_kind["binary"]), f"{binary_mean:.6f}"])

    print(f"[OK] {dataset}: unary_mean={unary_mean:.4f}, binary_mean={binary_mean:.4f}, plot={out_png}")


def main():
    for ds in DATASETS:
        plot_one_dataset(ds)


if __name__ == "__main__":
    main()
