import __main__
import csv
import importlib.util
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "0407"
BEST_CONFIG_CSV = REPORT_DIR / "best_config_by_dataset.csv"
OUT_CSV = REPORT_DIR / "final_without_dependency_vs_structural_none_stage1.csv"
OUT_MD = REPORT_DIR / "final_without_dependency_vs_structural_none_stage1.md"
TMP_ROOT = ROOT / "tmp_eval_final_without_dependency"


def load_best_configs():
    rows = []
    with open(BEST_CONFIG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "wikidata5m":
                continue
            rows.append(row)
    return rows


def import_aggregation(argv, module_name):
    sys.argv = argv
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "aggregation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_argv(cfg, relation, tmp_exp):
    batch_size = cfg.get("batch_size", 4096)
    lr = cfg.get("lr", 0.01)
    max_epoch = cfg.get("max_epoch", 40)
    evaluate_every = cfg.get("evaluate_every", 4)
    early_stopping = cfg.get("early_stopping", 12)
    pos = cfg.get("pos", "auto_sqrt")
    rule_init_mode = cfg.get("rule_init_mode", "conf")
    dependency_scale_mode = cfg.get("dependency_scale_mode", "none")
    eval_key_batch_size = cfg.get("eval_key_batch_size", 64)
    dependency_chunk_size = cfg.get("dependency_chunk_size", 4096)
    type_grouping = cfg.get("type_grouping", "none")
    argv = [
        "aggregation.py",
        "-d",
        cfg["dataset"],
        "--device",
        "cuda",
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--max_epoch",
        str(max_epoch),
        "--evaluate_every",
        str(evaluate_every),
        "--early_stopping",
        str(early_stopping),
        "--pos",
        str(pos),
        "--rule_init_mode",
        str(rule_init_mode),
        "--dependency_scale_mode",
        str(dependency_scale_mode),
        "--multiprocess",
        "0",
        "--eval_key_batch_size",
        str(eval_key_batch_size),
        "--dependency_chunk_size",
        str(dependency_chunk_size),
        "--rule_file",
        str(cfg["rule_file"]),
        "--relation",
        str(relation),
        "--max_worker_dataloader",
        "0",
        "--type_grouping",
        str(type_grouping),
    ]
    if not cfg.get("sign_constraint", True):
        argv.append("--no_sign_constraint")
    if cfg.get("sign_constraint_dependency", False):
        argv.append("--sign_constraint_dependency")
    if cfg.get("init_dep_with_lift", False):
        argv.append("--init_dep_with_lift")
    if cfg.get("train_rule_in_dependency_stage", False):
        argv.append("--train_rule_in_dependency_stage")
    if cfg.get("dependency_mask_low_rule_weight", False):
        argv.append("--dependency_mask_low_rule_weight")
    if cfg.get("synergy", False):
        argv.extend(
            [
                "--synergy",
                "--synergy_file",
                str(cfg.get("synergy_file", f"data/{cfg['dataset']}/rules/synergy_filtered.txt")),
            ]
        )
    if cfg.get("redundancy", False):
        argv.extend(
            [
                "--redundancy",
                "--redundancy_file",
                str(cfg.get("redundancy_file", f"data/{cfg['dataset']}/rules/redundancy_filtered.txt")),
            ]
        )
    os.environ["EXPERIMENT_DIR"] = str(tmp_exp)
    return argv


def load_directional_states(mod, relation, experiment_dir):
    sys.modules["aggregation"] = mod
    __main__.MRR = mod.MRR
    __main__.build_model_for_relation = mod.build_model_for_relation
    __main__.build_rule_only_model_for_relation = mod.build_rule_only_model_for_relation
    __main__.build_dependency_model_for_relation = mod.build_model_for_relation
    with open(Path(experiment_dir) / f"mrr-{relation}.pkl", "rb") as f:
        head_mrr, tail_mrr = pickle.load(f)
    return head_mrr.nnm, tail_mrr.nnm


def filter_state_for_rule_only(model, state_dict):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    filtered = {}
    for k, v in state_dict.items():
        if k not in model_keys:
            continue
        target = model_state[k]
        if getattr(v, "shape", None) != getattr(target, "shape", None):
            continue
        filtered[k] = v
    missing = sorted(model_keys - set(filtered.keys()))
    # Dependency-free model should still receive all rule/bias/type/global-scale params.
    unexpected = sorted(set(state_dict.keys()) - model_keys)
    return filtered, missing, unexpected


def evaluate_dataset(row):
    dataset = row["dataset"]
    best_config = row["best_config"]
    cfg_path = ROOT / "data" / dataset / "aggregation" / best_config / "config.json"
    cfg = json.load(open(cfg_path))
    relation_dir = ROOT / "data" / dataset / "aggregation" / best_config
    metrics_final = json.load(open(relation_dir / "metrics-final.json"))
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    tmp_exp = TMP_ROOT / dataset / best_config
    tmp_exp.mkdir(parents=True, exist_ok=True)

    sample_relation = 0
    argv = build_argv(cfg, sample_relation, tmp_exp)
    mod = import_aggregation(argv, f"agg_eval_without_dep_{dataset}_{best_config}".replace("-", "_"))
    mod.save = lambda *args, **kwargs: None

    weighted = {
        "mrr_sum": 0.0,
        "h1_sum": 0.0,
        "h10_sum": 0.0,
        "count_sum": 0.0,
    }
    relation_rows = []

    for metric_path in sorted(relation_dir.glob("metric-*.json")):
        relation = int(metric_path.stem.split("-")[1])
        m = json.load(open(metric_path))
        count = float(m.get("num_test_samples", m.get("test_count", 0)) or 0)
        if count <= 0:
            continue

        mod.args.relation = relation
        mod.clear_relation_processed_cache(relation)
        head_state, tail_state = load_directional_states(mod, relation, relation_dir)

        rule_only_builder = mod.build_rule_only_model_for_relation

        head_model = rule_only_builder(relation).to(mod.args.device)
        head_filtered, _head_missing, _head_unexpected = filter_state_for_rule_only(head_model, head_state)
        head_model.load_state_dict(head_filtered, strict=False)

        tail_model = rule_only_builder(relation).to(mod.args.device)
        tail_filtered, _tail_missing, _tail_unexpected = filter_state_for_rule_only(tail_model, tail_state)
        tail_model.load_state_dict(tail_filtered, strict=False)

        head_mrr = mod.MRR(relation=relation, direction="s", model_builder=rule_only_builder)
        tail_mrr = mod.MRR(relation=relation, direction="o", model_builder=rule_only_builder)
        head_raw = head_mrr.calc_metrics(head_model, head_mrr.test_sp_to_o, head_mrr.test_processed, direction="s")
        tail_raw = tail_mrr.calc_metrics(tail_model, tail_mrr.test_sp_to_o, tail_mrr.test_processed, direction="o")
        rel_metrics = mod.build_test_metrics_from_raw(head_raw, tail_raw)
        weighted["mrr_sum"] += rel_metrics["mrr"] * count
        weighted["h1_sum"] += rel_metrics["h1"] * count
        weighted["h10_sum"] += rel_metrics["h10"] * count
        weighted["count_sum"] += count
        relation_rows.append((relation, count, rel_metrics["mrr"], rel_metrics["h1"], rel_metrics["h10"]))

    if weighted["count_sum"] <= 0:
        raise RuntimeError(f"No relation test counts found for dataset={dataset} best_config={best_config}")

    final_without_dep = {
        "mrr": weighted["mrr_sum"] / weighted["count_sum"],
        "h1": weighted["h1_sum"] / weighted["count_sum"],
        "h10": weighted["h10_sum"] / weighted["count_sum"],
    }

    return {
        "dataset": dataset,
        "best_config": best_config,
        "best_config_test_mrr": float(row["best_config_mrr"]) if row["best_config_mrr"] else None,
        "structural_none_stage1_mrr": float(metrics_lookup(dataset, "structural_none__stage1", "MRR")),
        "structural_none_stage1_h1": float(metrics_lookup(dataset, "structural_none__stage1", "h@1")),
        "structural_none_stage1_h10": float(metrics_lookup(dataset, "structural_none__stage1", "h@10")),
        "final_without_dependency_mrr": final_without_dep["mrr"],
        "final_without_dependency_h1": final_without_dep["h1"],
        "final_without_dependency_h10": final_without_dep["h10"],
        "delta_vs_structural_none_stage1_mrr": final_without_dep["mrr"] - float(metrics_lookup(dataset, "structural_none__stage1", "MRR")),
        "delta_vs_structural_none_stage1_h1": final_without_dep["h1"] - float(metrics_lookup(dataset, "structural_none__stage1", "h@1")),
        "delta_vs_structural_none_stage1_h10": final_without_dep["h10"] - float(metrics_lookup(dataset, "structural_none__stage1", "h@10")),
        "delta_vs_best_config_final_mrr": final_without_dep["mrr"] - float(row["best_config_mrr"]) if row["best_config_mrr"] else None,
    }


_SUMMARY_ROWS = None


def metrics_lookup(dataset, aggregation, column):
    global _SUMMARY_ROWS
    if _SUMMARY_ROWS is None:
        with open(REPORT_DIR / "all_results_summary.csv", encoding="utf-8") as f:
            _SUMMARY_ROWS = list(csv.DictReader(f))
    for row in _SUMMARY_ROWS:
        if row["dataset"] == dataset and row["aggregation"] == aggregation:
            return row[column]
    raise KeyError(f"Missing {dataset}/{aggregation}/{column} in all_results_summary.csv")


def write_outputs(rows):
    fieldnames = [
        "dataset",
        "best_config",
        "best_config_test_mrr",
        "structural_none_stage1_mrr",
        "structural_none_stage1_h1",
        "structural_none_stage1_h10",
        "final_without_dependency_mrr",
        "final_without_dependency_h1",
        "final_without_dependency_h10",
        "delta_vs_structural_none_stage1_mrr",
        "delta_vs_structural_none_stage1_h1",
        "delta_vs_structural_none_stage1_h10",
        "delta_vs_best_config_final_mrr",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for k, v in list(out.items()):
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
            writer.writerow(out)

    lines = [
        "# Final Model Without Dependency vs structural_none Stage1",
        "",
        "For each dataset, load the final selected checkpoints of the dataset-level best configuration, rebuild an otherwise matching model with dependency features removed, and evaluate test metrics. Compare that to `structural_none__stage1`.",
        "",
        "| dataset | best_config | final_without_dep_mrr | structural_none_stage1_mrr | delta | final_best_mrr | delta_vs_final |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['best_config']} | {r['final_without_dependency_mrr']:.6f} | "
            f"{r['structural_none_stage1_mrr']:.6f} | {r['delta_vs_structural_none_stage1_mrr']:+.6f} | "
            f"{r['best_config_test_mrr']:.6f} | {r['delta_vs_best_config_final_mrr']:+.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows = []
    for row in load_best_configs():
        print(f"[eval-without-dep] dataset={row['dataset']} best_config={row['best_config']}")
        rows.append(evaluate_dataset(row))
    write_outputs(rows)
    print(f"[done] wrote {OUT_CSV}")
    print(f"[done] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
