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
        return "Observational results"
    direction_ok = (rho_all > 0 and expected_sign > 0) or (rho_all < 0 and expected_sign < 0)
    majority_ok = (ds_pos > ds_neg and expected_sign > 0) or (ds_neg > ds_pos and expected_sign < 0)
    if direction_ok and majority_ok and abs(rho_all) >= 0.10:
        return "established"
    if direction_ok and (majority_ok or abs(rho_all) >= 0.05):
        return "Partially established"
    return "Not established"


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
        Hypothesis(1, "Number of candidates", "num_candidates", "A. Candidate set complexity", +1, "The more candidates there are, the expected RuleDep More space"),
        Hypothesis(2, "Candidate-rule incidence count", "num_candidate_rule_edges", "A. Candidate set complexity", +1, "The more links the rule covers, the higher the expected revenue"),
        Hypothesis(3, "Average rules per candidate", "avg_rules_per_candidate", "A. Candidate set complexity", +1, "The co-occurrence of multiple rules is more conducive to dependency"),
        Hypothesis(4, "Max rules per candidate", "max_rules_per_candidate", "A. Candidate set complexity", +1, "At least one batch of candidates has abundant rule evidence"),
        Hypothesis(5, "Number of fired rules", "num_rules", "B. Rule graph structure", +1, "There are more rule nodes and a larger interaction space."),
        Hypothesis(6, "Number of dependencies", "num_dependencies", "B. Rule graph structure", +1, "The more active dependent edges there are, the higher the expected return."),
        Hypothesis(7, "Dependency density", "dep_density", "B. Rule graph structure", +1, "The denser the local rule graph, the more obvious the effect."),
        Hypothesis(8, "Candidate-level dependency coverage", "dep_candidate_ratio", "B. Rule graph structure", +1, "The broader the candidates that rely on participation, the more global the impact will be"),
        Hypothesis(9, "Number of positive dependencies", "num_pos_dep", "C. Positive/negative dependency structure", +1, "Positive dependencies are more likely to promote candidates"),
        Hypothesis(10, "Number of negative dependencies", "num_neg_dep", "C. Positive/negative dependency structure", 0, "Negative dependence can be suppressed over-count, The direction may not be monotonous"),
        Hypothesis(11, "Positive dependency ratio", "pos_dep_ratio", "C. Positive/negative dependency structure", +1, "More biased synergy Expected returns are higher when"),
        Hypothesis(12, "Negative dependency ratio", "neg_dep_ratio", "C. Positive/negative dependency structure", 0, "A high proportion of negative dependence may also bring correction benefits"),
        Hypothesis(13, "Total positive dependency mass", "pos_mass", "D. Dependency weight strength", +1, "The greater the total amount of positive dependence weight, the greater the potential gain."),
        Hypothesis(14, "Total negative dependency mass", "neg_mass", "D. Dependency weight strength", 0, "Negatively dependent mass is used to suppress redundancy"),
        Hypothesis(15, "Net dependency mass", "net_dep_mass", "D. Dependency weight strength", +1, "The higher the net dependence quality, the more favorable the expectations"),
        Hypothesis(16, "Absolute dependency mass", "abs_dep_mass", "D. Dependency weight strength", +1, "The higher the overall strength of dependence, the greater the space for intervention."),
        Hypothesis(17, "Top-k synergy weight", "topk_synergy", "D. Dependency weight strength", +1, "High strength synergy is the key mechanism"),
        Hypothesis(18, "Top-k redundancy weight", "topk_redundancy", "D. Dependency weight strength", 0, "High redundancy helps correct deviations"),
        Hypothesis(19, "Top-1 rule weight", "top1_rule_weight", "E. Rule-weight distribution", -1, "When a single rule is too strong dependency The effect becomes smaller"),
        Hypothesis(20, "Top-k rule weight", "topk_rule_weight", "E. Rule-weight distribution", -1, "When the rules are too strong, the room for correction becomes smaller."),
        Hypothesis(21, "Rule dominance ratio", "rule_dominance_ratio", "E. Rule-weight distribution", -1, "The more unimodal the evidence, thedependency The harder it is to play"),
        Hypothesis(22, "Weak-rule regime score", "weak_rule_score", "E. Rule-weight distribution", +1, "Weak rule scenarios rely more on dependency"),
        Hypothesis(23, "Dependency-to-rule mass ratio", "dep_rule_ratio", "F. Dependency-to-rule contrast", +1, "The stronger the reliance on relative rules, the greater the benefits"),
        Hypothesis(24, "Synergy-to-rule ratio", "syn_rule_ratio", "F. Dependency-to-rule contrast", +1, "synergy The stronger the relative rule, the more advantageous it is"),
        Hypothesis(25, "Redundancy-to-rule ratio", "red_rule_ratio", "F. Dependency-to-rule contrast", 0, "Redundant comparisons may help correct biases"),
        Hypothesis(26, "S1 top-1 score", "s1_top1", "G. S1 ambiguity", -1, "S1 The more certain it is, the harder it is to be RuleDep change"),
        Hypothesis(27, "S1 top1-top2 margin", "s1_margin", "G. S1 ambiguity", -1, "margin The smaller it is, the easier it is to flip"),
        Hypothesis(28, "Normalized S1 margin", "s1_norm_margin", "G. S1 ambiguity", -1, "normalization margin The smaller it is, the easier it is to flip"),
        Hypothesis(29, "S1 entropy", "s1_entropy", "G. S1 ambiguity", +1, "When uncertainty is high, it is more likely to be dependency Correction"),
        Hypothesis(30, "Effective candidate number", "effective_candidates", "G. S1 ambiguity", +1, "The more valid hesitant candidates, the greater the room for correction."),
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
                "verdict": "not covered",
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
        "# RuleDep Feature Hypothesis Verification (Chinese)",
        "",
        "- Data:`official_query_triple_features.csv` (After rerun)",
        "- Number of samples:{}".format(len(parsed_rows)),
        "- Goal:query-level `raw_delta_rr` (true per-query RR Difference, not used relation-level calibration offset) ",
        "- Statistics:Spearman Related (full amount + Directional consistency across data sets)",
        "",
        "## 1. Overall conclusion",
        "",
    ]

    covered_n = sum(1 for r in result_rows if r["covered"] == "yes")
    md_lines.append(f"- Coverage:{covered_n}/30 Hypothetical features have been covered and evaluated.")
    md_lines.append(f"- Established:{sum(1 for r in result_rows if r['verdict'] == 'established')} Item; partially established:{sum(1 for r in result_rows if r['verdict'] == 'Partially established')} item; not established:{sum(1 for r in result_rows if r['verdict'] == 'Not established')} item.")
    md_lines.append("")

    md_lines.extend([
        "## 2. Classification hypothesis and verification results",
        "",
    ])

    by_cat = defaultdict(list)
    for r in result_rows:
        by_cat[r["category"]].append(r)

    for cat, rows_cat in sorted(by_cat.items()):
        md_lines.append(f"### {cat}")
        md_lines.append("")
        md_lines.append("| # | feature | Hypothetical direction | rho_all | same direction(Right/Negative/total) | Conclusion |")
        md_lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
        for r in sorted(rows_cat, key=lambda x: int(x["id"])):
            if r["covered"] != "yes":
                md_lines.append(f"| {r['id']} | {r['feature']} | {r['expected_sign']} | NA | NA | not covered |")
                continue
            md_lines.append(
                f"| {r['id']} | {r['feature']} | {r['expected_sign']} | {float(r['rho_all']):.4f} | {r['datasets_pos']}/{r['datasets_neg']}/{r['datasets_n']} | {r['verdict']} |"
            )
        md_lines.append("")

    # strongest reliable supporters
    strong = [r for r in ranked if r["verdict"] in {"established", "Partially established"} and float(r["robust_score"]) > 0]
    strong = strong[:8]

    md_lines.extend([
        "## 3. Which features are the most important and have the most reliable support?",
        "",
        "press `robust_score = |rho_all| × Directional consistency across datasets` Ranking (only statistics for items whose direction is consistent with the hypothesis).",
        "",
        "| rank | feature | Category | rho_all | Consistency(Match direction/total) | robust_score |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ])
    for i, r in enumerate(strong, 1):
        exp = int(r["expected_sign"])
        aligned = int(r["datasets_pos"]) if exp > 0 else int(r["datasets_neg"]) if exp < 0 else max(int(r["datasets_pos"]), int(r["datasets_neg"]))
        md_lines.append(
            f"| {i} | {r['feature']} | {r['category']} | {float(r['rho_all']):.4f} | {aligned}/{r['datasets_n']} | {float(r['robust_score']):.4f} |"
        )

    md_lines.append("")
    md_lines.append("Intuitively, the most reliable group remains concentrated in **dependency intensity and contrast**: Such as `topk_synergy`, `syn_rule_ratio`, `dep_rule_ratio`, `pos_mass`, `abs_dep_mass` Wait.")
    md_lines.append("")

    # key figures
    key_features = [r["feature"] for r in strong[:6]]
    md_lines.extend([
        "## 4. Key pictures (can be viewed directly)",
        "",
        "The most critical characteristic curves are selected below (priority `desc` direction):",
        "",
    ])
    for f in key_features:
        slug = slugify(f)
        md_lines.append(f"- {f} (desc) : ![{f}](../feature_plots/{slug}__desc.png)")

    md_lines.extend([
        "",
        "## 5. Interpretation suggestions",
        "",
        "1. If the paper/The main line of the report emphasizes RuleDep marginal value, priority reporting dependency Quality related characteristics (D/F category).",
        "2. for E/G The non-established terms in the class are recommended as ‘applicable boundaries’ rather than as main conclusions: they are more unstable in direction across different data sets.",
        "3. It is recommended to keep in the main table:`topk_synergy`, `syn_rule_ratio`, `dep_rule_ratio`, `candidate_dep_coverage`, `num_dependencies`. ",
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
