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


def load_existing_rows():
    if not OUT_CSV.exists():
        return []
    return list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))


def build_argv_from_config(cfg, relation, tmp_exp):
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


def import_aggregation_module(module_name, argv):
    sys.argv = argv
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "aggregation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_key(mod, model, processed_entry):
    candidates_all = list(processed_entry["candidates"])
    rule_lists_all = list(processed_entry["rules"])
    kept_pairs = [(cand, rules) for cand, rules in zip(candidates_all, rule_lists_all) if len(rules) > 0]

    scores = torch.full((mod.dataset.num_entities(),), 0.0, device=mod.EVAL_DEVICE)
    if not kept_pairs:
        return scores, len(candidates_all), 0

    kept_candidates = torch.as_tensor([cand for cand, _rules in kept_pairs], dtype=torch.long, device=mod.EVAL_DEVICE)
    rules = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor([torch.tensor(rules) for _cand, rules in kept_pairs]),
        padding=mod.PAD_TOK,
    ).long().to(mod.EVAL_DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(rules)).detach()
    max_conf = mod.RULE_CONF_TABLE[rules].max(dim=1, keepdim=True).values
    score = (pred * max_conf).squeeze(1)
    scores[kept_candidates] = score
    return scores, len(candidates_all), len(kept_pairs)


def load_final_directional_states(mod, relation, experiment_dir):
    sys.modules["aggregation"] = mod
    __main__.MRR = mod.MRR
    __main__.build_model_for_relation = mod.build_model_for_relation
    __main__.build_rule_only_model_for_relation = mod.build_rule_only_model_for_relation
    __main__.build_dependency_model_for_relation = mod.build_model_for_relation
    with open(Path(experiment_dir) / f"mrr-{relation}.pkl", "rb") as f:
        head_mrr, tail_mrr = pickle.load(f)
    relation_dependencies = mod.dependency_map.get(relation, [])
    dependency_model_builder = lambda rel: mod.build_model_for_relation(rel, relation_dependencies=relation_dependencies)
    rule_only_model_builder = mod.build_rule_only_model_for_relation

    def _load_with_fallback(state_dict):
        try:
            return mod.build_model_from_state_dict(relation, dependency_model_builder, state_dict)
        except RuntimeError as exc:
            # Some relations are selected from stage1/rule-only checkpoints without dependency tensors.
            msg = str(exc)
            if "dependencies.weight" not in msg and "size mismatch for synergy_pair_a_local" not in msg:
                raise
            return mod.build_model_from_state_dict(relation, rule_only_model_builder, state_dict)

    final_tail_model = _load_with_fallback(tail_mrr.nnm)
    final_head_model = _load_with_fallback(head_mrr.nnm)
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
    selected_state_dict = stage1_result.get("selected_state_dict")
    if selected_state_dict is not None:
        return mod.build_model_from_state_dict(relation, rule_model_builder, selected_state_dict)
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
    def fmt_float(value):
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)

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
            f"{fmt_float(row['stage1_rank'])} | {fmt_float(row['final_rank'])} | {fmt_float(row['rank_gain'])} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(rows, relation_count):
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
        writer.writerows(rows)
    write_markdown(rows, relation_count)


def main():
    positive_rows = load_positive_relations()
    only_dataset = os.environ.get("ONLY_DATASET")
    if only_dataset:
        positive_rows = [row for row in positive_rows if row["dataset"] == only_dataset]
    start_dataset = os.environ.get("START_DATASET")
    if start_dataset and not only_dataset:
        dataset_order = []
        for row in positive_rows:
            if row["dataset"] not in dataset_order:
                dataset_order.append(row["dataset"])
        if start_dataset in dataset_order:
            start_idx = dataset_order.index(start_dataset)
            allowed = set(dataset_order[start_idx:])
            positive_rows = [row for row in positive_rows if row["dataset"] in allowed]

    append_existing = os.environ.get("APPEND_EXISTING", "0") == "1"
    all_rows = load_existing_rows() if append_existing else []
    replace_existing_dataset = os.environ.get("REPLACE_EXISTING_DATASET", "0") == "1"
    if append_existing and replace_existing_dataset and only_dataset:
        all_rows = [row for row in all_rows if row["dataset"] != only_dataset]
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
        deduped_rows = []
        seen = set()
        for case_row in all_rows:
            key = (
                case_row["dataset"],
                case_row["best_config"],
                case_row["relation"],
                case_row["direction"],
                case_row["query_text"],
                case_row["gold_entity"],
                case_row["stage1_top1"],
                case_row["final_top1"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_rows.append(case_row)
        all_rows = deduped_rows
        write_outputs(all_rows, len(positive_rows))

    write_outputs(all_rows, len(positive_rows))
    print(f"wrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
