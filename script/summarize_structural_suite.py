#!/usr/bin/env python3
import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Experiment root, e.g. data/codex-m/aggregation")
    parser.add_argument("--out", required=True, help="Summary output json path")
    args = parser.parse_args()

    experiment_names = ["rd", "r2d3", "r3d3", "r3d6"]
    results = {}
    for name in experiment_names:
        metrics_path = os.path.join(args.root, f"structural_{name}", "metrics-final.json")
        if not os.path.exists(metrics_path):
            results[name] = {"status": "missing", "metrics_path": metrics_path}
            continue
        payload = load_json(metrics_path)
        summary = payload.get("summary") or {}
        results[name] = {
            "status": "ok",
            "metrics_path": metrics_path,
            "test_after_stage1": summary.get("test_after_stage1"),
            "test": summary.get("test"),
        }

    rd_stage1 = (((results.get("rd") or {}).get("test_after_stage1") or {}).get("mrr"))
    output = {
        "root": args.root,
        "r_baseline_mrr": rd_stage1,
        "experiments": results,
        "improvement_vs_r_baseline": {},
    }

    if rd_stage1 is not None:
        rd_stage1 = float(rd_stage1)
        for name in experiment_names:
            final_test = (((results.get(name) or {}).get("test") or {}).get("mrr"))
            if final_test is None:
                continue
            output["improvement_vs_r_baseline"][name] = float(final_test) - rd_stage1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
