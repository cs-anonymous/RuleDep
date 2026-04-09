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
POSITIVE_CASES_CSV = REPORT_DIR / "relation_positive_examples_gt3_dependency.csv"
OUT_CSV = REPORT_DIR / "relation_case_study_examples.csv"
OUT_MD = REPORT_DIR / "relation_case_study_examples.md"


def load_positive_relations():
    return list(csv.DictReader(open(POSITIVE_CASES_CSV, encoding="utf-8")))


def build_argv_from_config(cfg, relation, tmp_exp):
    argv = [
        "aggregation.py",
        "-d",
        cfg["dataset"],
        "--device",
        "cuda",
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--max_epoch",
        str(cfg["max_epoch"]),
        "--evaluate_every",
        str(cfg["evaluate_every"]),
        "--early_stopping",
        str(cfg["early_stopping"]),
        "--pos",
        str(cfg["pos"]),
        "--rule_init_mode",
        str(cfg["rule_init_mode"]),
        "--dependency_scale_mode",
        str(cfg.get("dependency_scale_mode", "none")),
        "--multiprocess",
        "0",
        "--eval_key_batch_size",
        str(cfg.get("eval_key_batch_size", 64)),
        "--dependency_chunk_size",
        str(cfg.get("dependency_chunk_size", 4096)),
        "--rule_file",
        str(cfg["rule_file"]),
        "--relation",
        str(relation),
        "--max_worker_dataloader",
        "0",
        "--type_grouping",
        str(cfg.get("type_grouping", "none")),
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


def import_aggregation_module(module_name, argv):
    sys.argv = argv
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "aggregation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_key(mod, model, processed_entry):
    candidates = torch.as_tensor(processed_entry["candidates"], dtype=torch.long, device=mod.EVAL_DEVICE)
    rule_lists = processed_entry["rules"]
    if len(rule_lists) > 0:
        rules = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor([torch.tensor(x) for x in rule_lists]),
            padding=mod.PAD_TOK,
        ).long().to(mod.EVAL_DEVICE)
    else:
        rules = torch.empty((0, 0), dtype=torch.long, device=mod.EVAL_DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(rules)).detach()
    max_conf = mod.RULE_CONF_TABLE[rules].max(dim=1, keepdim=True).values
    score = (pred * max_conf).squeeze(1)
    scores = torch.full((mod.dataset.num_entities(),), 0.0, device=mod.EVAL_DEVICE)
    scores[candidates] = score
    return scores, len(candidates), len(rule_lists)


def load_final_directional_states(mod, relation, experiment_dir):
    sys.modules["aggregation"] = mod
    __main__.MRR = mod.MRR
    with open(Path(experiment_dir) / f"mrr-{relation}.pkl", "rb") as f:
        head_mrr, tail_mrr = pickle.load(f)
    relation_dependencies = mod.dependency_map.get(relation, [])
    dependency_model_builder = lambda rel: mod.build_model_for_relation(rel, relation_dependencies=relation_dependencies)
    final_tail_model = mod.build_model_from_state_dict(relation, dependency_model_builder, tail_mrr.nnm)
    final_head_model = mod.build_model_from_state_dict(relation, dependency_model_builder, head_mrr.nnm)
    return final_head_model, final_tail_model


def train_stage1_only(mod, relation):
    dataloader, train_split = mod.load_dataloaders(mod.args.directory_preprocessed_datasets, relation)
    pos, _pos_source, _num_pos, _num_neg = mod.resolve_pos_weight(mod.args.pos, train_split, relation)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())
    lr_values = mod.parse_csv_schedule(mod.args.lr, float, "lr")
    eval_every_values = mod.parse_csv_schedule(mod.args.evaluate_every, int, "evaluate_every")
    rule_model_builder = mod.build_rule_only_model_for_relation
    rule_model = rule_model_builder(relation)
    stage1_result = mod.run_training_stage(
        relation=relation,
        model=rule_model,
        model_builder=rule_model_builder,
        dataloader=dataloader,
        loss_fn=loss_fn,
        pos=pos,
        lr_values=lr_values,
        eval_every_values=eval_every_values,
        max_epoch=mod.args.max_epoch,
        stage_name="rule",
        checkpoint_selection="combined",
    )
    return stage1_result["model"]


def build_group_runtime(row):
    dataset = row["dataset"]
    best_config = row["best_config"]
    config_path = ROOT / "data" / dataset / "aggregation" / best_config / "config.json"
    cfg = json.load(open(config_path))
    first_relation = int(row["relation"])
    tmp_exp = ROOT / "tmp_case_study" / dataset / best_config
    if tmp_exp.exists():
        shutil.rmtree(tmp_exp)
    tmp_exp.mkdir(parents=True, exist_ok=True)

    argv = build_argv_from_config(cfg, first_relation, tmp_exp)
    mod = import_aggregation_module(f"agg_case_{dataset}_{best_config}".replace("-", "_"), argv)
    # Disable persistence for the temporary rerun.
    mod.save = lambda *args, **kwargs: None
    return cfg, mod


def collect_relation_cases(mod, row):
    dataset = row["dataset"]
    relation = int(row["relation"])
    best_config = row["best_config"]

    mod.args.relation = relation
    mod.clear_relation_processed_cache(relation)
    mod.args.synergy = False
    mod.args.redundancy = False
    stage1_model = train_stage1_only(mod, relation)

    # Final dependency model: load the already trained directional checkpoints.
    final_head_model, final_tail_model = load_final_directional_states(mod, relation, ROOT / "data" / dataset / "aggregation" / best_config)

    entity_ids = mod.dataset.entity_ids()
    relation_name = mod.dataset.relation_ids()[relation]
    rows = []
    for direction, keys_name, index_obj, split_direction, final_model in [
        ("tail", "test_o", mod.test_sp_to_o, "o", final_tail_model),
        ("head", "test_s", mod.test_po_to_s, "s", final_head_model),
    ]:
        processed = mod.load_relation_processed(relation, "test", split_direction)
        keys = mod.relation_keys[keys_name].get(relation, [])
        for key in keys:
            if key not in processed:
                continue
            golds = index_obj[key].long().to(mod.EVAL_DEVICE)
            if len(golds) != 1:
                continue

            stage1_scores, candidate_count, active_rule_count = score_key(mod, stage1_model, processed[key])
            final_scores, _, _ = score_key(mod, final_model, processed[key])

            stage1_rank = float(mod._rank_from_scores_tensor(stage1_scores, golds, None)[0].item())
            final_rank = float(mod._rank_from_scores_tensor(final_scores, golds, None)[0].item())
            stage1_top1 = int(stage1_scores.argmax().item())
            final_top1 = int(final_scores.argmax().item())
            gold_id = int(golds[0].item())

            if not (stage1_rank > 1.0 and final_rank == 1.0 and final_top1 == gold_id):
                continue

            if direction == "tail":
                query_entity = entity_ids[key[0]]
                fixed_entity = ""
                query_text = f"({query_entity}, {relation_name}, ?)"
            else:
                fixed_entity = entity_ids[key[1]]
                query_entity = ""
                query_text = f"(?, {relation_name}, {fixed_entity})"

            rows.append(
                {
                    "dataset": dataset,
                    "best_config": best_config,
                    "relation": str(relation),
                    "relation_name": row["relation_name"],
                    "relation_gloss_zh": row["relation_gloss_zh"],
                    "mapping_direction": row["mapping_direction"],
                    "direction": direction,
                    "query_text": query_text,
                    "query_entity": query_entity,
                    "fixed_entity": fixed_entity,
                    "gold_entity": entity_ids[gold_id],
                    "stage1_top1": entity_ids[stage1_top1],
                    "final_top1": entity_ids[final_top1],
                    "stage1_rank": stage1_rank,
                    "final_rank": final_rank,
                    "rank_gain": stage1_rank - final_rank,
                    "candidate_count": candidate_count,
                    "active_rule_count": active_rule_count,
                    "baseline_mrr_relation": float(row["baseline_mrr"]),
                    "final_mrr_relation": float(row["final_mrr"]),
                }
            )

    rows.sort(key=lambda r: (-r["rank_gain"], r["stage1_rank"], r["dataset"], r["relation_name"], r["query_text"]))
    return rows


def write_markdown(rows, relation_count):
    lines = [
        "# Case Study Examples",
        "",
        "Examples where the stage-1 model failed (`rank > 1`) but the dependency-augmented final model succeeded (`rank = 1`).",
        "",
        f"Scanned positive relations: `{relation_count}`",
        f"Recovered query-level flip cases: `{len(rows)}`",
        "",
        "| dataset | relation | zh | direction | query | gold | stage1 top1 | final top1 | stage1 rank | final rank | rank gain |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['relation_name']} | {row['relation_gloss_zh']} | {row['direction']} | "
            f"{row['query_text']} | {row['gold_entity']} | {row['stage1_top1']} | {row['final_top1']} | "
            f"{row['stage1_rank']:.1f} | {row['final_rank']:.1f} | {row['rank_gain']:.1f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    positive_rows = load_positive_relations()
    all_rows = []
    group_runtime = {}
    for row in positive_rows:
        group_key = (row["dataset"], row["best_config"])
        if group_key not in group_runtime:
            print(
                f"[case-study] loading group dataset={row['dataset']} best_config={row['best_config']}",
                flush=True,
            )
            group_runtime[group_key] = build_group_runtime(row)
        print(
            f"[case-study] scanning dataset={row['dataset']} relation={row['relation']} "
            f"name={row['relation_name']}",
            flush=True,
        )
        _cfg, mod = group_runtime[group_key]
        all_rows.extend(collect_relation_cases(mod, row))

    fieldnames = [
        "dataset",
        "best_config",
        "relation",
        "relation_name",
        "relation_gloss_zh",
        "mapping_direction",
        "direction",
        "query_text",
        "query_entity",
        "fixed_entity",
        "gold_entity",
        "stage1_top1",
        "final_top1",
        "stage1_rank",
        "final_rank",
        "rank_gain",
        "candidate_count",
        "active_rule_count",
        "baseline_mrr_relation",
        "final_mrr_relation",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    write_markdown(all_rows, len(positive_rows))
    print(f"wrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
