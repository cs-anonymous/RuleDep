#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import analyze_official_query_subsets as aq  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build official query feature plots from existing CSV")
    parser.add_argument("--out-dir", default="/home/sy/RuleDep/reports/0421/official_query_subset")
    args = parser.parse_args()

    features_csv = os.path.join(args.out_dir, "official_query_triple_features.csv")
    with open(features_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key, value in list(row.items()):
            try:
                if value != "":
                    row[key] = float(value)
            except Exception:
                pass

    coverage_steps = [round(x / 100.0, 2) for x in range(100, 0, -1)]
    curves = aq.build_threshold_curves(rows, coverage_steps)
    aq.write_csv(os.path.join(args.out_dir, "feature_threshold_curves.csv"), curves)
    aq.plot_feature_curves(curves, args.out_dir)

    summary = aq.build_best_summary(curves)
    aq.write_csv(os.path.join(args.out_dir, "best_feature_threshold_summary.csv"), summary)
    aq.write_markdown(os.path.join(args.out_dir, "README.md"), summary, curves)

    print(f"rows={len(rows)}")
    print(f"curves={len(curves)}")
    print(f"plots_dir={os.path.join(args.out_dir, 'feature_plots')}")


if __name__ == "__main__":
    main()