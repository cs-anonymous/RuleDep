#!/usr/bin/env python3
import json
from pathlib import Path


DATASETS = ["FB15k-237", "KG20C", "WN18RR", "YAGO3-10", "codex-l", "codex-m", "hetionet"]
TYPE_EXPERIMENTS = {
    "structural_none": "none",
    "structural_rd": "rd",
    "structural_r2d3": "r2d3",
    "structural_r3d6": "r3d6",
}
DEP_FILTER_EXPERIMENTS = {
    "structural_none": "default",
    "structural_none_lift_k1": "lift_k1",
    "structural_none_lift_k4": "lift_k4",
    "structural_none_ratio_k1": "ratio_k1",
    "structural_none_ratio_k2": "ratio_k2",
    "structural_none_ratio_k4": "ratio_k4",
    "structural_none_mix_k1": "mix_k1",
    "structural_none_mix_k2": "mix_k2",
    "structural_none_mix_k4": "mix_k4",
}


def load_test_mrr(path: Path):
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    if "summary" in obj and "test" in obj["summary"]:
        return float(obj["summary"]["test"]["mrr"])
    if "all_metrics" in obj and "test" in obj["all_metrics"]:
        return float(obj["all_metrics"]["test"]["mrr"])
    return None


def best_by_map(agg_root: Path, experiment_map):
    best_name = None
    best_value = None
    for exp_name, mapped in experiment_map.items():
        value = load_test_mrr(agg_root / exp_name / "metrics-final.json")
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = mapped
            best_value = value
    return best_name, best_value


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()

    dataset = args.dataset
    agg_root = Path(args.data_root) / dataset / "aggregation"
    rules_root = Path(args.data_root) / dataset / "rules"

    baseline = load_test_mrr(agg_root / "structural_none" / "metrics-final.json")
    if baseline is None:
        raise SystemExit(f"Missing structural_none metrics for {dataset}")

    best_type_grouping, _ = best_by_map(agg_root, TYPE_EXPERIMENTS)
    best_dep_filter, _ = best_by_map(agg_root, DEP_FILTER_EXPERIMENTS)

    surprisal_value = load_test_mrr(agg_root / "structural_surprisal_init" / "metrics-final.json")
    use_surprisal = surprisal_value is not None and surprisal_value > baseline

    pos_auto_ratio_value = load_test_mrr(agg_root / "pos_auto_ratio" / "metrics-final.json")
    use_pos_auto_ratio = pos_auto_ratio_value is not None and pos_auto_ratio_value > baseline

    synergy_file = ""
    redundancy_file = ""
    if best_dep_filter and best_dep_filter != "default":
        synergy_file = str(rules_root / f"synergy_filtered_{best_dep_filter}.txt")
        redundancy_file = str(rules_root / f"redundancy_filtered_{best_dep_filter}.txt")

    payload = {
        "dataset": dataset,
        "baseline_mrr": baseline,
        "type_grouping": best_type_grouping or "none",
        "dep_filter_variant": best_dep_filter or "default",
        "use_surprisal_init": bool(use_surprisal),
        "use_pos_auto_ratio": bool(use_pos_auto_ratio),
        "synergy_file": synergy_file,
        "redundancy_file": redundancy_file,
    }
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
