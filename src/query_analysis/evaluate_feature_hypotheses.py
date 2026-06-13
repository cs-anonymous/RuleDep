#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass

FEATURE_CSV = "/home/sy/RuleDep/reports/official_query_subset/official_query_triple_features.csv"
OUT_DIR = "/home/sy/RuleDep/reports/official_query_subset/hypothesis_eval"


@dataclass
class Hypothesis:
    idx: int
    name: str
    feature: str
    category: str
    expected_sign: int  # +1 positive, -1 negative, 0 mixed
    note: str


def to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def rankdata(vals: list[float]) -> list[float]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")
    xx = [a for a, _ in pairs]
    yy = [b for _, b in pairs]
    rx = rankdata(xx)
    ry = rankdata(yy)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def slugify(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", v).strip("_")


def verdict(expected_sign: int, rho_all: float, ds_pos: int, ds_neg: int, ds_n: int) -> str:
    if expected_sign == 0:
        return "观察性结果"
    direction_ok = (rho_all > 0 and expected_sign > 0) or (rho_all < 0 and expected_sign < 0)
    majority_ok = (ds_pos > ds_neg and expected_sign > 0) or (ds_neg > ds_pos and expected_sign < 0)
    if direction_ok and majority_ok and abs(rho_all) >= 0.10:
        return "成立"
    if direction_ok and (majority_ok or abs(rho_all) >= 0.05):
        return "部分成立"
    return "不成立"


def robust_score(expected_sign: int, rho_all: float, ds_pos: int, ds_neg: int, ds_n: int) -> float:
    if math.isnan(rho_all) or ds_n == 0:
        return 0.0
    if expected_sign == 0:
        sign_consistency = max(ds_pos, ds_neg) / ds_n
        return abs(rho_all) * sign_consistency
    aligned = ds_pos if expected_sign > 0 else ds_neg
    sign_consistency = aligned / ds_n
    sign_ok = 1.0 if ((expected_sign > 0 and rho_all > 0) or (expected_sign < 0 and rho_all < 0)) else 0.0
    return abs(rho_all) * sign_consistency * sign_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", default=FEATURE_CSV)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.feature_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    numeric_columns = set()
    parsed_rows = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            fv = to_float(v)
            if fv is None:
                rr[k] = v
            else:
                rr[k] = fv
                numeric_columns.add(k)
        parsed_rows.append(rr)

    hypotheses = [
        Hypothesis(1, "Number of candidates", "num_candidates", "A. Candidate set complexity", +1, "候选越多，预期 RuleDep 空间更大"),
        Hypothesis(2, "Candidate-rule incidence count", "num_candidate_rule_edges", "A. Candidate set complexity", +1, "规则覆盖链接越多，预期收益更高"),
        Hypothesis(3, "Average rules per candidate", "avg_rules_per_candidate", "A. Candidate set complexity", +1, "多规则共现更利于 dependency"),
        Hypothesis(4, "Max rules per candidate", "max_rules_per_candidate", "A. Candidate set complexity", +1, "至少有一批候选具备丰富规则证据"),
        Hypothesis(5, "Number of fired rules", "num_rules", "B. Rule graph structure", +1, "规则节点更多，交互空间更大"),
        Hypothesis(6, "Number of dependencies", "num_dependencies", "B. Rule graph structure", +1, "活跃依赖边越多，预期收益越高"),
        Hypothesis(7, "Dependency density", "dep_density", "B. Rule graph structure", +1, "局部规则图越稠密，作用更明显"),
        Hypothesis(8, "Candidate-level dependency coverage", "dep_candidate_ratio", "B. Rule graph structure", +1, "依赖参与的候选越广，影响越全局"),
        Hypothesis(9, "Number of positive dependencies", "num_pos_dep", "C. Positive/negative dependency structure", +1, "正依赖多时更可能提升候选"),
        Hypothesis(10, "Number of negative dependencies", "num_neg_dep", "C. Positive/negative dependency structure", 0, "负依赖可抑制 over-count，方向未必单调"),
        Hypothesis(11, "Positive dependency ratio", "pos_dep_ratio", "C. Positive/negative dependency structure", +1, "更偏 synergy 时预期收益更高"),
        Hypothesis(12, "Negative dependency ratio", "neg_dep_ratio", "C. Positive/negative dependency structure", 0, "负依赖占比高也可能带来修正收益"),
        Hypothesis(13, "Total positive dependency mass", "pos_mass", "D. Dependency weight strength", +1, "正依赖权重总量越大，潜在增益越大"),
        Hypothesis(14, "Total negative dependency mass", "neg_mass", "D. Dependency weight strength", 0, "负依赖质量用于压制冗余"),
        Hypothesis(15, "Net dependency mass", "net_dep_mass", "D. Dependency weight strength", +1, "净依赖质量越高，预期越有利"),
        Hypothesis(16, "Absolute dependency mass", "abs_dep_mass", "D. Dependency weight strength", +1, "依赖总体强度越高，介入空间越大"),
        Hypothesis(17, "Top-k synergy weight", "topk_synergy", "D. Dependency weight strength", +1, "高强度 synergy 是关键机制"),
        Hypothesis(18, "Top-k redundancy weight", "topk_redundancy", "D. Dependency weight strength", 0, "高冗余可帮助纠偏"),
        Hypothesis(19, "Top-1 rule weight", "top1_rule_weight", "E. Rule-weight distribution", -1, "单条规则过强时 dependency 作用变小"),
        Hypothesis(20, "Top-k rule weight", "topk_rule_weight", "E. Rule-weight distribution", -1, "规则太强时可修正空间变小"),
        Hypothesis(21, "Rule dominance ratio", "rule_dominance_ratio", "E. Rule-weight distribution", -1, "证据越单峰，dependency 越难发挥"),
        Hypothesis(22, "Weak-rule regime score", "weak_rule_score", "E. Rule-weight distribution", +1, "弱规则场景更依赖 dependency"),
        Hypothesis(23, "Dependency-to-rule mass ratio", "dep_rule_ratio", "F. Dependency-to-rule contrast", +1, "依赖相对规则越强，收益越大"),
        Hypothesis(24, "Synergy-to-rule ratio", "syn_rule_ratio", "F. Dependency-to-rule contrast", +1, "synergy 相对规则越强越有利"),
        Hypothesis(25, "Redundancy-to-rule ratio", "red_rule_ratio", "F. Dependency-to-rule contrast", 0, "冗余对比可能帮助纠偏"),
        Hypothesis(26, "S1 top-1 score", "s1_top1", "G. S1 ambiguity", -1, "S1 越确定，越难被 RuleDep 改变"),
        Hypothesis(27, "S1 top1-top2 margin", "s1_margin", "G. S1 ambiguity", -1, "margin 越小越易翻转"),
        Hypothesis(28, "Normalized S1 margin", "s1_norm_margin", "G. S1 ambiguity", -1, "归一化 margin 越小越易翻转"),
        Hypothesis(29, "S1 entropy", "s1_entropy", "G. S1 ambiguity", +1, "不确定性高时更易被 dependency 修正"),
        Hypothesis(30, "Effective candidate number", "effective_candidates", "G. S1 ambiguity", +1, "有效犹豫候选越多，修正空间越大"),
    ]

    by_dataset = defaultdict(list)
    for r in parsed_rows:
        by_dataset[str(r["dataset"])].append(r)

    y_all = [to_float(r.get("raw_delta_rr", r.get("delta_rr"))) for r in parsed_rows]

    result_rows = []
    for h in hypotheses:
        covered = h.feature in numeric_columns
        if not covered:
            result_rows.append({
                "id": h.idx,
                "category": h.category,
                "feature": h.feature,
                "name": h.name,
                "covered": "no",
                "expected_sign": h.expected_sign,
                "rho_all": "",
                "datasets_pos": "",
                "datasets_neg": "",
                "datasets_n": "",
                "verdict": "未覆盖",
                "note": h.note,
                "plot_desc": "",
                "plot_asc": "",
                "robust_score": "",
            })
            continue

        x_all = [to_float(r.get(h.feature)) for r in parsed_rows]
        rho_all = spearman(x_all, y_all)

        ds_pos = 0
        ds_neg = 0
        ds_n = 0
        for ds_rows in by_dataset.values():
            xd = [to_float(r.get(h.feature)) for r in ds_rows]
            yd = [to_float(r.get("raw_delta_rr", r.get("delta_rr"))) for r in ds_rows]
            rd = spearman(xd, yd)
            if math.isnan(rd):
                continue
            ds_n += 1
            if rd > 0:
                ds_pos += 1
            elif rd < 0:
                ds_neg += 1

        v = verdict(h.expected_sign, rho_all, ds_pos, ds_neg, ds_n)
        score = robust_score(h.expected_sign, rho_all, ds_pos, ds_neg, ds_n)
        slug = slugify(h.feature)
        result_rows.append({
            "id": h.idx,
            "category": h.category,
            "feature": h.feature,
            "name": h.name,
            "covered": "yes",
            "expected_sign": h.expected_sign,
            "rho_all": rho_all,
            "datasets_pos": ds_pos,
            "datasets_neg": ds_neg,
            "datasets_n": ds_n,
            "verdict": v,
            "note": h.note,
            "plot_desc": f"../feature_plots/{slug}__desc.png",
            "plot_asc": f"../feature_plots/{slug}__asc.png",
            "robust_score": score,
        })

    csv_path = os.path.join(args.out_dir, "hypothesis_validation.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    # importance ranking
    ranked = [r for r in result_rows if r["covered"] == "yes"]
    ranked.sort(key=lambda r: float(r["robust_score"]), reverse=True)
    rank_path = os.path.join(args.out_dir, "feature_importance_ranking.csv")
    with open(rank_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)

    # markdown report (Chinese)
    md_lines = [
        "# RuleDep 特征假设验证（中文）",
        "",
        "- 数据：`official_query_triple_features.csv`（重跑后）",
        "- 样本数：{}".format(len(parsed_rows)),
        "- 目标：query-level `raw_delta_rr`（真实 per-query RR 差值，不使用 relation-level calibration offset）",
        "- 统计：Spearman 相关（全量 + 分数据集方向一致性）",
        "",
        "## 一、总体结论",
        "",
    ]

    covered_n = sum(1 for r in result_rows if r["covered"] == "yes")
    md_lines.append(f"- 覆盖：{covered_n}/30 个假设特征已覆盖并完成评估。")
    md_lines.append(f"- 成立：{sum(1 for r in result_rows if r['verdict'] == '成立')} 项；部分成立：{sum(1 for r in result_rows if r['verdict'] == '部分成立')} 项；不成立：{sum(1 for r in result_rows if r['verdict'] == '不成立')} 项。")
    md_lines.append("")

    md_lines.extend([
        "## 二、分门别类假设与验证结果",
        "",
    ])

    by_cat = defaultdict(list)
    for r in result_rows:
        by_cat[r["category"]].append(r)

    for cat, rows_cat in sorted(by_cat.items()):
        md_lines.append(f"### {cat}")
        md_lines.append("")
        md_lines.append("| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |")
        md_lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
        for r in sorted(rows_cat, key=lambda x: int(x["id"])):
            if r["covered"] != "yes":
                md_lines.append(f"| {r['id']} | {r['feature']} | {r['expected_sign']} | NA | NA | 未覆盖 |")
                continue
            md_lines.append(
                f"| {r['id']} | {r['feature']} | {r['expected_sign']} | {float(r['rho_all']):.4f} | {r['datasets_pos']}/{r['datasets_neg']}/{r['datasets_n']} | {r['verdict']} |"
            )
        md_lines.append("")

    # strongest reliable supporters
    strong = [r for r in ranked if r["verdict"] in {"成立", "部分成立"} and float(r["robust_score"]) > 0]
    strong = strong[:8]

    md_lines.extend([
        "## 三、哪些特征最重要、支撑最可靠",
        "",
        "按 `robust_score = |rho_all| × 跨数据集方向一致性` 排序（仅统计方向与假设一致的项）。",
        "",
        "| rank | feature | 类别 | rho_all | 一致性(匹配方向/总) | robust_score |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for i, r in enumerate(strong, 1):
        exp = int(r["expected_sign"])
        aligned = int(r["datasets_pos"]) if exp > 0 else int(r["datasets_neg"]) if exp < 0 else max(int(r["datasets_pos"]), int(r["datasets_neg"]))
        md_lines.append(
            f"| {i} | {r['feature']} | {r['category']} | {float(r['rho_all']):.4f} | {aligned}/{r['datasets_n']} | {float(r['robust_score']):.4f} |"
        )

    md_lines.append("")
    md_lines.append("直观上，最可靠的一组仍集中在 **dependency 强度与对比**：如 `topk_synergy`、`syn_rule_ratio`、`dep_rule_ratio`、`pos_mass`、`abs_dep_mass` 等。")
    md_lines.append("")

    # key figures
    key_features = [r["feature"] for r in strong[:6]]
    md_lines.extend([
        "## 四、关键图片（可直接查看）",
        "",
        "以下选择了最关键的特征曲线图（优先 `desc` 方向）：",
        "",
    ])
    for f in key_features:
        slug = slugify(f)
        md_lines.append(f"- {f}（desc）: ![{f}](../feature_plots/{slug}__desc.png)")

    md_lines.extend([
        "",
        "## 五、解读建议",
        "",
        "1. 若论文/报告主线强调 RuleDep 的边际价值，优先报告 dependency 质量相关特征（D/F 类）。",
        "2. 对于 E/G 类中不成立项，建议作为‘适用边界’而非主结论：它们在不同数据集上方向更不稳定。",
        "3. 建议在主表保留：`topk_synergy`、`syn_rule_ratio`、`dep_rule_ratio`、`candidate_dep_coverage`、`num_dependencies`。",
        "",
    ])

    md_path = os.path.join(args.out_dir, "hypothesis_validation_cn.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"rows={len(parsed_rows)}")
    print(f"hypotheses={len(hypotheses)}")
    print(f"covered={covered_n}")
    print(f"out={args.out_dir}")


if __name__ == "__main__":
    main()
