#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import analyze_official_query_subsets as aq  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build official query feature CSV only")
    parser.add_argument("--example-root", default=aq.DEFAULT_EXAMPLE_ROOT)
    parser.add_argument("--out-dir", default="/home/sy/RuleDep/reports/0421/official_query_subset")
    args = parser.parse_args()

    rows = aq.compute_rows(args.example_root)
    rows = aq.add_composite_features(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "official_query_triple_features.csv")
    aq.write_csv(out_path, rows)

    print(f"rows={len(rows)}")
    print(f"features={len(aq.numeric_feature_names(rows))}")
    print(f"csv={out_path}")


if __name__ == "__main__":
    main()