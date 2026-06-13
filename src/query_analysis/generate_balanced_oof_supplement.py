#!/usr/bin/env python3
"""Supplementary OOF runs:
1. balanced5_rf (per-dataset, 5 features) via 5-fold random OOF.
2. balanced5_global_rf LODO (leave-one-dataset-out): train on 6 datasets,
   predict on the 7th; repeat for all 7 folds. Tests whether the unified
   global scorer transfers to a previously unseen dataset.

Appends to reports/official_query_subset/ml_selector_diverse/balanced_oof_true_rr/
  balanced_oof_macro_gain.csv                (rows added for new selectors)
  balanced_oof_per_dataset_gain.csv
  balanced_oof_vs_insample.csv
  balanced_oof_lodo_per_dataset.csv          (new)

Also updates the README with an LODO section.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_balanced5_selector_report as g  # noqa: E402
from generate_balanced5_selector_report_true_rr import load_merged_rr  # noqa: E402
from sweep_official_query_ml_selectors import stable_seed, target_values  # noqa: E402


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced_oof_true_rr"

N_SPLITS = 5


def kfold_seed(label: str) -> int:
    return stable_seed(f"5foldOOF:{label}") % (2**31)


def oof_per_dataset(df: pd.DataFrame, features: list[str], selector: str) -> pd.Series:
    out_score = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    t0 = time.time()
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index()
        x = group[features].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        seed = kfold_seed(f"{dataset}:{selector}")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        scores_local = np.full(len(group), np.nan, dtype=float)
        t1 = time.time()
        for train_idx, test_idx in kf.split(x):
            model = g.make_rf()
            model.fit(x.iloc[train_idx], y[train_idx])
            scores_local[test_idx] = model.predict(x.iloc[test_idx])
        if np.any(np.isnan(scores_local)):
            raise RuntimeError(f"{selector}/{dataset}: NaN in OOF")
        out_score.iloc[group["index"].to_numpy()] = scores_local
        print(
            f"    [{selector}] dataset={dataset} ({len(group)} rows) 5-fold OOF {time.time() - t1:.1f}s "
            f"(cumulative {time.time() - t0:.1f}s)",
            flush=True,
        )
    return out_score


def lodo_global(df: pd.DataFrame, features: list[str], selector: str) -> pd.Series:
    out_score = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    t0 = time.time()
    x_full = df[features].replace([np.inf, -np.inf], np.nan)
    y_full = target_values(df, "gain_clip")
    datasets = sorted(df["dataset"].unique())
    for held_out in datasets:
        is_test = (df["dataset"] == held_out).to_numpy()
        train_mask = ~is_test
        model = g.make_rf()
        t1 = time.time()
        model.fit(x_full[train_mask], y_full[train_mask])
        pred = model.predict(x_full[is_test])
        out_score.iloc[np.where(is_test)[0]] = pred
        print(
            f"    [{selector}] held_out={held_out} (train {train_mask.sum()}, test {is_test.sum()}) "
            f"fit+predict {time.time() - t1:.1f}s (cumulative {time.time() - t0:.1f}s)",
            flush=True,
        )
    return out_score


def gain_curves(df: pd.DataFrame, score: pd.Series | np.ndarray, selector: str) -> pd.DataFrame:
    rows = []
    work = df.reset_index(drop=True).copy()
    work["_score"] = np.asarray(score, dtype=float)
    for dataset, group in work.groupby("dataset", sort=True):
        gp = group.reset_index(drop=True)
        s = gp["_score"].to_numpy(dtype=float)
        curves = g.score_coverages(gp, s, g.COVERAGES)
        curves.insert(0, "selector", selector)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True)


def summarize_macro(curves: pd.DataFrame) -> pd.DataFrame:
    fixed = curves[curves["coverage"].isin(g.FIXED)].copy()
    macro = (
        fixed.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in g.FIXED]
    return macro


def per_dataset_table(curves: pd.DataFrame) -> pd.DataFrame:
    table = curves.pivot_table(
        index=["selector", "dataset"], columns="coverage", values="gain_pt"
    )
    table.columns = [f"gain_{int(c * 100)}" for c in table.columns]
    return table.reset_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[supp-oof] loading data ...", flush=True)
    df, _ = load_merged_rr()
    print(f"[supp-oof] {len(df)} rows", flush=True)

    all_curves = []

    print("[supp-oof] (1/2) balanced5_rf per-dataset 5-fold OOF", flush=True)
    score_b5_rf = oof_per_dataset(df, g.BALANCED5, "balanced5_rf")
    curves_b5_rf = gain_curves(df, score_b5_rf.values, "balanced5_rf")
    all_curves.append(curves_b5_rf)

    print("[supp-oof] (2/2) balanced5_global_rf LODO (leave-one-dataset-out)", flush=True)
    score_lodo = lodo_global(df, g.BALANCED5, "balanced5_global_rf_lodo")
    curves_lodo = gain_curves(df, score_lodo.values, "balanced5_global_rf_lodo")
    all_curves.append(curves_lodo)

    new_curves = pd.concat(all_curves, ignore_index=True)
    new_macro = summarize_macro(new_curves)
    new_per_dataset = per_dataset_table(new_curves)

    existing_curves_path = OUT_DIR / "balanced_oof_gain_curves.csv"
    existing_curves = pd.read_csv(existing_curves_path)
    existing_curves = existing_curves[
        ~existing_curves["selector"].isin(new_curves["selector"].unique())
    ]
    pd.concat([existing_curves, new_curves], ignore_index=True).to_csv(
        existing_curves_path, index=False
    )

    existing_macro_path = OUT_DIR / "balanced_oof_macro_gain.csv"
    existing_macro = pd.read_csv(existing_macro_path)
    existing_macro = existing_macro[
        ~existing_macro["selector"].isin(new_macro["selector"].unique())
    ]
    combined_macro = pd.concat([existing_macro, new_macro], ignore_index=True)
    combined_macro.to_csv(existing_macro_path, index=False)

    existing_per_dataset_path = OUT_DIR / "balanced_oof_per_dataset_gain.csv"
    existing_per_dataset = pd.read_csv(existing_per_dataset_path)
    existing_per_dataset = existing_per_dataset[
        ~existing_per_dataset["selector"].isin(new_per_dataset["selector"].unique())
    ]
    pd.concat([existing_per_dataset, new_per_dataset], ignore_index=True).to_csv(
        existing_per_dataset_path, index=False
    )

    lodo_per_dataset = new_per_dataset[new_per_dataset["selector"] == "balanced5_global_rf_lodo"].copy()
    lodo_per_dataset.to_csv(OUT_DIR / "balanced_oof_lodo_per_dataset.csv", index=False)

    # Update vs_insample for balanced5_rf (in-sample from balanced5 report).
    ins_b5 = pd.read_csv(
        REPORT_DIR
        / "ml_selector_diverse"
        / "balanced5_report_true_rr"
        / "balanced5_macro_gain_table.csv"
    )
    ins_row = ins_b5[ins_b5["selector"] == "balanced5_rf"]
    if not ins_row.empty:
        ir = ins_row.iloc[0]
        oof_row = new_macro[new_macro["selector"] == "balanced5_rf"].iloc[0]
        delta10 = float(oof_row["gain_10"]) - float(ir["gain_10"])
        delta20 = float(oof_row["gain_20"]) - float(ir["gain_20"])
        delta50 = float(oof_row["gain_50"]) - float(ir["gain_50"])
        print(
            f"[supp-oof] balanced5_rf in-sample gain@10={float(ir['gain_10']) * 100:.2f}% "
            f"OOF={float(oof_row['gain_10']) * 100:.2f}% delta={delta10 * 100:+.2f}pt",
            flush=True,
        )
        h2h_path = OUT_DIR / "balanced_oof_vs_insample.csv"
        h2h = pd.read_csv(h2h_path)
        h2h = h2h[h2h["selector"] != "balanced5_rf"]
        h2h = pd.concat(
            [
                h2h,
                pd.DataFrame(
                    [
                        {
                            "selector": "balanced5_rf",
                            "insample@10": float(ir["gain_10"]),
                            "oof@10": float(oof_row["gain_10"]),
                            "insample@20": float(ir["gain_20"]),
                            "oof@20": float(oof_row["gain_20"]),
                            "insample@50": float(ir["gain_50"]),
                            "oof@50": float(oof_row["gain_50"]),
                            "delta@10": delta10,
                            "delta@20": delta20,
                            "delta@50": delta50,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        h2h.to_csv(h2h_path, index=False)

    print("[supp-oof] writing README sections ...", flush=True)
    update_readme(combined_macro, new_per_dataset)
    print(OUT_DIR / "README.md", flush=True)


def update_readme(macro: pd.DataFrame, new_per_dataset: pd.DataFrame) -> None:
    def fmt_pct(v):
        if pd.isna(v):
            return ""
        return f"{v * 100:.2f}%"

    readme_path = OUT_DIR / "README.md"
    body = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    marker = "<!-- SUPP_OOF_SECTION -->"
    if marker in body:
        body = body.split(marker)[0].rstrip() + "\n"

    macro_disp = macro.copy()
    for col in ["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]:
        macro_disp[col] = macro_disp[col].map(fmt_pct)
    macro_md = macro_disp.to_markdown(index=False)

    lodo_table = new_per_dataset[new_per_dataset["selector"] == "balanced5_global_rf_lodo"].copy()
    lodo_disp = lodo_table.copy()
    for col in [c for c in lodo_disp.columns if c.startswith("gain_")]:
        lodo_disp[col] = lodo_disp[col].map(fmt_pct)
    lodo_md = lodo_disp.drop(columns=["selector"]).to_markdown(index=False)

    supp = f"""
{marker}

## Supplementary: balanced5_rf OOF and global RF LODO

Added after the initial 12-selector run:

- `balanced5_rf` per-dataset 5-fold OOF: completes the 5-feature ceiling
  comparison.
- `balanced5_global_rf_lodo`: leave-one-dataset-out. Train a single global
  RandomForest on 6 datasets, predict on the 7th; repeat for all 7 held-out
  datasets. Tests whether the unified Balanced5 scorer transfers to a dataset
  never seen during training.

### Full macro OOF table (rebuilt to include supplementary rows)

{macro_md}

### LODO per-dataset (which dataset is hardest to transfer to?)

For each row below, the RF was trained on the *other 6 datasets* and
evaluated on the row's dataset only. `gain_X` is the coverage-X gain under
this LODO prediction.

{lodo_md}
"""
    readme_path.write_text(body + supp, encoding="utf-8")


if __name__ == "__main__":
    main()
