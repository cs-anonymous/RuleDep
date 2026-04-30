#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
FEATURE_PATH = REPORT_DIR / "official_query_triple_features.csv"
CODEX_L_AGG = ROOT / "data" / "codex-l" / "aggregation"
OUT_MD = REPORT_DIR / "codex_l_calibration_diagnostic.md"
OUT_CONFIG_CSV = REPORT_DIR / "codex_l_config_gain_diagnostic.csv"

FORMULA_FEATURES = [
    "topk_synergy",
    "pos_mass",
    "effective_candidates",
    "rule_dominance_ratio",
    "topk_redundancy",
    "syn_rule_ratio",
]


def percentile_by_dataset(df: pd.DataFrame, feature: str) -> pd.Series:
    return df.groupby("dataset")[feature].rank(pct=True, method="average")


def add_high_gain_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_high_gain_score"] = (
        3.0 * percentile_by_dataset(out, "topk_synergy")
        + 1.0 * percentile_by_dataset(out, "pos_mass")
        + 3.0 * percentile_by_dataset(out, "effective_candidates")
        + 4.0 * (1.0 - percentile_by_dataset(out, "rule_dominance_ratio"))
        + 0.25 * (1.0 - percentile_by_dataset(out, "topk_redundancy"))
        + 1.0 * percentile_by_dataset(out, "syn_rule_ratio")
    ) / 12.25
    return out


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    codex = df[df["dataset"] == "codex-l"].sort_values("_high_gain_score", ascending=False, kind="mergesort")
    for coverage in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
        n = max(1, int(round(len(codex) * coverage)))
        subset = codex.iloc[:n]
        for metric, c1, c2 in [
            ("calibrated", "rr_stage1", "rr_stage2"),
            ("raw", "raw_rr_stage1", "raw_rr_stage2"),
        ]:
            mrr1 = float(subset[c1].mean())
            mrr2 = float(subset[c2].mean())
            rows.append(
                {
                    "coverage": coverage,
                    "n": n,
                    "metric": metric,
                    "mrr_stage1": mrr1,
                    "mrr_stage2": mrr2,
                    "gain_pt": (mrr2 / mrr1 - 1.0) if mrr1 > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def selected_relation_table(df: pd.DataFrame) -> pd.DataFrame:
    codex = df[df["dataset"] == "codex-l"].sort_values("_high_gain_score", ascending=False, kind="mergesort")
    subset = codex.iloc[: max(1, int(round(len(codex) * 0.10)))]
    return (
        subset.groupby("relation")
        .agg(
            n=("relation", "size"),
            raw_mrr_stage1=("raw_rr_stage1", "mean"),
            raw_mrr_stage2=("raw_rr_stage2", "mean"),
            calibrated_mrr_stage2=("rr_stage2", "mean"),
            calibration_offset=("calibration_offset", "mean"),
            official_relation_delta_mrr=("official_relation_delta_mrr", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )


def codex_l_config_table() -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in CODEX_L_AGG.iterdir() if p.is_dir()):
        rel_rows = []
        for metric_path in exp_dir.glob("metric-*.json"):
            try:
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stage1 = (metric.get("test_after_stage1") or {}).get("mrr")
            stage2 = (metric.get("test_after_stage2") or metric.get("test_after_stage1") or {}).get("mrr")
            n = metric.get("num_test_samples") or 0
            if stage1 is None or stage2 is None or not n:
                continue
            rel_rows.append((float(stage1), float(stage2), int(n)))
        if not rel_rows:
            continue
        total_n = sum(n for _, _, n in rel_rows)
        mrr1 = sum(s1 * n for s1, _, n in rel_rows) / total_n
        mrr2 = sum(s2 * n for _, s2, n in rel_rows) / total_n
        rows.append(
            {
                "experiment": exp_dir.name,
                "relations": len(rel_rows),
                "test_cases": total_n,
                "weighted_stage1_mrr": mrr1,
                "weighted_stage2_mrr": mrr2,
                "gain_pt": (mrr2 / mrr1 - 1.0) if mrr1 > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("gain_pt", ascending=False)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    usecols = [
        "dataset",
        "relation",
        "raw_rr_stage1",
        "raw_rr_stage2",
        "rr_stage1",
        "rr_stage2",
        "calibration_offset",
        "official_relation_delta_mrr",
    ] + FORMULA_FEATURES
    df = pd.read_csv(FEATURE_PATH, usecols=usecols)
    df = add_high_gain_score(df)

    cov = coverage_table(df)
    relations = selected_relation_table(df)
    configs = codex_l_config_table()
    configs.to_csv(OUT_CONFIG_CSV, index=False)

    cov_pivot = cov.pivot(index=["coverage", "n"], columns="metric", values="gain_pt").reset_index()
    lines = [
        "# CODEX-L Calibration Diagnostic",
        "",
        "This diagnostic checks whether the compact high-gain formula's CODEX-L peak is a real query-subset effect or an artifact of relation-level calibration offsets.",
        "",
        "## Compact Formula on CODEX-L",
        "",
        "| coverage | n | calibrated gain_pt | raw gain_pt |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in cov_pivot.itertuples(index=False):
        lines.append(
            f"| {row.coverage:.0%} | {int(row.n)} | {fmt_pct(row.calibrated)} | {fmt_pct(row.raw)} |"
        )

    lines.extend(
        [
            "",
            "## Top Relations in the 10% CODEX-L Subset",
            "",
            "| relation | n | raw stage1 MRR | raw stage2 MRR | calibrated stage2 MRR | calibration offset | official relation delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in relations.head(10).itertuples(index=False):
        lines.append(
            f"| {row.relation} | {int(row.n)} | {row.raw_mrr_stage1:.6f} | {row.raw_mrr_stage2:.6f} | "
            f"{row.calibrated_mrr_stage2:.6f} | {row.calibration_offset:.6f} | {row.official_relation_delta_mrr:.6f} |"
        )

    lines.extend(
        [
            "",
            "## CODEX-L Other Configs: Full-test Weighted Gain",
            "",
            "These numbers are full-test weighted MRR gains from `metric-*.json`; they are not query-subset coverage gains.",
            "",
            "| experiment | stage1 MRR | stage2 MRR | gain_pt |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in configs.head(16).itertuples(index=False):
        lines.append(
            f"| `{row.experiment}` | {row.weighted_stage1_mrr:.6f} | {row.weighted_stage2_mrr:.6f} | {fmt_pct(row.gain_pt)} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_CONFIG_CSV)


if __name__ == "__main__":
    main()
