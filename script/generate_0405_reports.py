from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/home/sy/RuleDep")
REPORT = ROOT / "reports" / "0405"
DATASETS = ["FB15k-237", "KG20C", "WN18RR", "YAGO3-10", "codex-l", "codex-m", "hetionet"]


def parse_rule_line(line: str):
    parts = line.split("\t", 3)
    if len(parts) < 4:
        return None
    try:
        body_size = int(parts[0])
        support = int(parts[1])
        score = float(parts[2])
    except ValueError:
        return None
    return {"bodySize": body_size, "support": support, "score": score, "rule": parts[3]}


def is_d_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    if len(parts) != 2:
        return False
    body = parts[1]
    return body.count("(A,") + body.count(",A)") == 1


def is_z_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    if len(parts) != 2:
        return False
    return parts[1].strip() == ""


def is_b_rule(rule_str: str) -> bool:
    parts = rule_str.split("<=", 1)
    if len(parts) != 2:
        return False
    return "(X,Y)" in parts[0].strip()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def count_dict_file(path: Path) -> int:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.name.endswith(".txt") and rows and rows[0].strip().isdigit() and "\t" not in rows[0] and " " not in rows[0]:
        return int(rows[0].strip())
    return len(rows)


def count_entities_relations(dataset: str) -> tuple[int, int]:
    base = ROOT / "data" / dataset
    ent = reln = 0
    for path in [base / "entity_ids.del", base / "entities.dict", base / "entity2id.txt"]:
        if path.exists():
            ent = count_dict_file(path)
            break
    for path in [base / "relation_ids.del", base / "relations.dict", base / "relation2id.txt"]:
        if path.exists():
            reln = count_dict_file(path)
            break
    return ent, reln


def copy_relation_analysis_csvs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in [
        "relation_dependency_analysis.csv",
        "relation_gain_group_summary.csv",
        "relation_gain_dataset_summary.csv",
        "relation_gain_stage1_bucket_summary.csv",
        "relation_gain_dep_density_bucket_summary.csv",
        "relation_type_weight_summary.csv",
    ]:
        src = ROOT / "reports" / name
        dst = REPORT / name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def generate_relation_gain_tables() -> None:
    comparison = pd.read_csv(REPORT / "structural_filtered_comparison.csv")
    rel = pd.read_csv(REPORT / "relation_dependency_analysis.csv")
    best_map = dict(zip(comparison["dataset"], comparison["best_by_mrr"]))
    rel = rel[rel.apply(lambda row: best_map.get(row["dataset"]) == row["aggregation"], axis=1)].copy()
    rel["relation"] = rel["relation"].astype(int)

    out = rel[
        [
            "dataset",
            "aggregation",
            "relation",
            "relation_name",
            "test_triple_count",
            "stage1_mrr",
            "final_test_mrr",
            "abs_gain",
            "rel_gain_pct",
            "selected_stage",
        ]
    ].copy()
    out.rename(
        columns={
            "final_test_mrr": "test_mrr",
            "abs_gain": "absolute_gain",
            "rel_gain_pct": "relative_gain_pct",
        },
        inplace=True,
    )
    for column in ["stage1_mrr", "test_mrr", "absolute_gain", "relative_gain_pct"]:
        out[column] = out[column].map(lambda value: format(float(value), ".5f"))
    out["test_triple_count"] = out["test_triple_count"].astype(int)
    out["relation"] = out["relation"].astype(int)
    out.sort_values(["dataset", "relation"], inplace=True)

    out[out["relative_gain_pct"].astype(float) > 3.0].to_csv(
        REPORT / "relation_relative_gain_gt_3pct_best_structural.csv", index=False
    )
    out[out["relative_gain_pct"].astype(float) < -3.0].to_csv(
        REPORT / "relation_relative_gain_lt_minus_3pct_best_structural.csv", index=False
    )


def generate_dataset_stats() -> None:
    rows = []
    for dataset in DATASETS:
        base = ROOT / "data" / dataset
        ent, reln = count_entities_relations(dataset)
        rule_path = base / "rules" / "rule.txt"
        total_rules = b_rules = d_rules = z_rules = 0
        if rule_path.exists():
            with rule_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    parsed = parse_rule_line(line)
                    if not parsed:
                        continue
                    total_rules += 1
                    rule = parsed["rule"]
                    if is_b_rule(rule):
                        b_rules += 1
                    elif is_d_rule(rule):
                        d_rules += 1
                    elif is_z_rule(rule):
                        z_rules += 1
        uc_rules = total_rules - b_rules - d_rules - z_rules
        rows.append(
            {
                "dataset": dataset,
                "#entity": ent,
                "#relation": reln,
                "#train": line_count(base / "train.txt"),
                "#test": line_count(base / "test.txt"),
                "#valid": line_count(base / "valid.txt"),
                "#rule": total_rules,
                "#B rule": b_rules,
                "#Ud rule": d_rules,
                "#Uc rule": uc_rules,
                "#synergy": line_count(base / "rules" / "synergy.txt"),
                "#redundancy": line_count(base / "rules" / "redundancy.txt"),
                "#filtered_synergy": line_count(base / "rules" / "synergy_filtered.txt"),
                "#filtered_redundancy": line_count(base / "rules" / "redundancy_filtered.txt"),
            }
        )
    pd.DataFrame(rows).to_csv(REPORT / "dataset_size_rule_dependency_stats.csv", index=False)


def main() -> None:
    copy_relation_analysis_csvs()
    generate_relation_gain_tables()
    generate_dataset_stats()
    print(f"Wrote 0405 report artifacts to {REPORT}")


if __name__ == "__main__":
    main()
