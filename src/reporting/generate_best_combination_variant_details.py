#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PREFERRED_DATASET_ORDER = [
    "KG20C",
    "codex-m",
    "WN18RR",
    "FB15k-237",
    "codex-l",
    "YAGO3-10",
    "hetionet",
    "wikidata5m",
]

EXCLUDED_PREFIXES = ["best_combination"]
EXCLUDED_EXACT = {"eval-maxplus", "eval-noisyor", "canonical", "ensemble_best_valid", "ensemble_safe_valid", "ensemble_best_test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate best_combination variant details report.")
    parser.add_argument("--root", default=None, help="Repository root. Defaults to script parent directory.")
    parser.add_argument("--report-dir", default=None, help="Report output directory. Defaults to <root>/reports/0407.")
    return parser.parse_args()


def dataset_sort_key(name: str) -> int:
    return PREFERRED_DATASET_ORDER.index(name) if name in PREFERRED_DATASET_ORDER else 999


def to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{float(value):.6f}"


def load_rows(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ds = str(row.get("dataset") or "")
            agg = str(row.get("aggregation") or "")
            mrr = to_float(str(row.get("MRR") or ""))
            if not ds or not agg or mrr is None:
                continue
            out[ds][agg] = mrr
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_candidate_config(name: str) -> bool:
    if name in EXCLUDED_EXACT or name.endswith("__stage1"):
        return False
    if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    return True


def plot_delta(rows: List[Dict[str, object]], out_path: Path) -> None:
    if plt is None or not rows:
        return
    datasets = [str(row["dataset"]) for row in rows]
    deltas = [float(row.get("ensemble_valid_delta_vs_best") or 0.0) for row in rows]
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in deltas]
    x = list(range(len(datasets)))
    plt.figure(figsize=(10, 4))
    plt.bar(x, deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("MRR delta")
    plt.title("Ensemble-valid minus best remaining config")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_md(rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Best-combination Variant Details")
    lines.append("")
    lines.append("This section counts according to the current reporting caliber: first remove all `best_combination*`, Then select the optimal configuration for each data set among the remaining configurations and give two sets relation-level ensemble. ")
    lines.append("")
    lines.append("Related documents:")
    lines.append("")
    lines.append("- `best_combination_variant_details.csv`")
    lines.append("- `plot_best_combination_variant_delta.png`")
    lines.append("")
    lines.append("| Dataset | Best remaining config | Best MRR | Ensemble-valid | Ensemble-safe | Ensemble-test | Δ(valid-best) | Δ(safe-best) | Δ(test-best) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {dataset} | {best} | {best_mrr} | {ens_v} | {ens_s} | {ens_t} | {d_v} | {d_s} | {d_t} |".format(
                dataset=row["dataset"],
                best=row.get("best_remaining_config") or "",
                best_mrr=fmt(row.get("best_remaining_mrr")),
                ens_v=fmt(row.get("ensemble_best_valid_mrr")),
                ens_s=fmt(row.get("ensemble_safe_valid_mrr")),
                ens_t=fmt(row.get("ensemble_best_test_mrr")),
                d_v=fmt(row.get("ensemble_valid_delta_vs_best")),
                d_s=fmt(row.get("ensemble_safe_delta_vs_best")),
                d_t=fmt(row.get("ensemble_test_delta_vs_best")),
            )
        )
    lines.append("")
    lines.append('<p align="center"><img src="plot_best_combination_variant_delta.png" alt="ensemble-valid minus best remaining" width="60%"></p>')
    lines.append("")
    lines.append('<p align="center"><em>Figure: relation-level ensemble-valid gain relative to the best single remaining config.</em></p>')
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    report_dir = Path(args.report_dir).resolve() if args.report_dir else root / "reports" / "0407"
    report_dir.mkdir(parents=True, exist_ok=True)

    src_csv = report_dir / "all_results_summary.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing required file: {src_csv}")

    by_dataset = load_rows(src_csv)
    rows: List[Dict[str, object]] = []
    for dataset in sorted(by_dataset.keys(), key=dataset_sort_key):
        agg_map = by_dataset[dataset]
        candidates = {k: v for k, v in agg_map.items() if is_candidate_config(k)}
        best_cfg = ""
        best_mrr = None
        if candidates:
            best_cfg = max(candidates.items(), key=lambda item: float(item[1]))[0]
            best_mrr = candidates[best_cfg]
        ens_valid = agg_map.get("ensemble_best_valid")
        ens_safe = agg_map.get("ensemble_safe_valid")
        ens_test = agg_map.get("ensemble_best_test")

        row: Dict[str, object] = {
            "dataset": dataset,
            "best_remaining_config": best_cfg,
            "best_remaining_mrr": best_mrr,
            "ensemble_best_valid_mrr": ens_valid,
            "ensemble_safe_valid_mrr": ens_safe,
            "ensemble_best_test_mrr": ens_test,
            "ensemble_valid_delta_vs_best": None if ens_valid is None or best_mrr is None else float(ens_valid) - float(best_mrr),
            "ensemble_safe_delta_vs_best": None if ens_safe is None or best_mrr is None else float(ens_safe) - float(best_mrr),
            "ensemble_test_delta_vs_best": None if ens_test is None or best_mrr is None else float(ens_test) - float(best_mrr),
        }
        rows.append(row)

    csv_fields = [
        "dataset",
        "best_remaining_config",
        "best_remaining_mrr",
        "ensemble_best_valid_mrr",
        "ensemble_safe_valid_mrr",
        "ensemble_best_test_mrr",
        "ensemble_valid_delta_vs_best",
        "ensemble_safe_delta_vs_best",
        "ensemble_test_delta_vs_best",
    ]
    write_csv(report_dir / "best_combination_variant_details.csv", rows, csv_fields)
    plot_delta(rows, report_dir / "plot_best_combination_variant_delta.png")
    (report_dir / "best_combination_variant_details.md").write_text(build_md(rows), encoding="utf-8")

    print(f"Wrote {report_dir / 'best_combination_variant_details.csv'}")
    print(f"Wrote {report_dir / 'best_combination_variant_details.md'}")
    if plt is not None:
        print(f"Wrote {report_dir / 'plot_best_combination_variant_delta.png'}")


if __name__ == "__main__":
    main()
