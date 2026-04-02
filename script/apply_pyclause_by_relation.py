import argparse
import os
import tempfile

from c_clause import Loader, RankingHandler
from clause import Options

from apply_pyclause import export_direction


def read_relation_to_id_map(train_path):
    data_dir = os.path.dirname(train_path)
    rel_file = os.path.join(data_dir, "relation_ids.del")
    if not os.path.exists(rel_file):
        raise FileNotFoundError(f"relation_ids.del not found: {rel_file}")

    relation_to_id = {}
    with open(rel_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            rel = parts[1] if len(parts) >= 2 else line
            relation_to_id[rel] = idx
    return relation_to_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run apply_pyclause.py per relation and write one file per relation"
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--worker-threads", type=int, default=-1)
    parser.add_argument("--aggregation", default="maxplus")
    parser.add_argument("--filter-w-data", type=int, default=1)
    parser.add_argument("--min-correct-predictions", type=int, default=5)
    parser.add_argument("--read-cyclic-rules", type=int, default=1)
    parser.add_argument("--read-acyclic1-rules", type=int, default=1)
    parser.add_argument("--read-acyclic2-rules", type=int, default=0)
    parser.add_argument("--read-zero-rules", type=int, default=0)
    parser.add_argument("--read-uxxc-rules", type=int, default=1)
    parser.add_argument("--read-uxxd-rules", type=int, default=1)
    parser.add_argument("--b-max-length", type=int, default=-1)
    parser.add_argument("--num_top_rules", type=int, default=200)

    return parser.parse_args()


def iter_triples(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                continue
            yield parts[0], parts[1], parts[2]


def split_target_by_relation(target_path, out_dir):
    relation_to_path = {}
    handles = {}
    try:
        for h, r, t in iter_triples(target_path):
            if r not in handles:
                rel_path = os.path.join(out_dir, f"target_{r}.txt")
                relation_to_path[r] = rel_path
                handles[r] = open(rel_path, "w", encoding="utf-8")
            handles[r].write(f"{h}\t{r}\t{t}\n")
    finally:
        for fp in handles.values():
            fp.close()
    return relation_to_path


def build_options(args):
    opts = Options()
    opts.set("loader.load_b_rules", bool(args.read_cyclic_rules))
    opts.set("loader.load_u_c_rules", bool(args.read_acyclic1_rules))
    opts.set("loader.load_u_d_rules", bool(args.read_acyclic2_rules))
    opts.set("loader.load_zero_rules", bool(args.read_zero_rules))
    opts.set("loader.load_u_xxc_rules", bool(args.read_uxxc_rules))
    opts.set("loader.load_u_xxd_rules", bool(args.read_uxxd_rules))
    opts.set("loader.b_max_length", int(args.b_max_length))
    opts.set("loader.b_min_support", int(args.min_correct_predictions))
    opts.set("loader.c_min_support", int(args.min_correct_predictions))
    opts.set("loader.b_max_branching_factor", -1)
    opts.set("loader.num_threads", int(args.worker_threads))

    opts.set("ranking_handler.collect_rules", True)
    opts.set("ranking_handler.topk", args.topk)
    opts.set("ranking_handler.aggregation_function", args.aggregation)
    opts.set("ranking_handler.filter_w_data", bool(args.filter_w_data))
    opts.set("ranking_handler.num_top_rules", args.num_top_rules)
    opts.set("ranking_handler.num_threads", int(args.worker_threads))
    opts.set("ranking_handler.disc_at_least", -1)
    return opts


def run_one_relation(args, loader, ranker, relation, relation_to_id, target_path):
    rel_id = relation_to_id.get(relation)
    if rel_id is None:
        raise KeyError(f"Relation not found in relation_ids.del: {relation}")

    output_prefix = os.path.join(args.output_dir, f"{args.split}_{rel_id}")
    print(f"[apply_by_relation] relation={relation} output_prefix={output_prefix}", flush=True)

    loader.load_target(target_path)
    ranker.calculate_ranking(loader=loader)

    head_output = f"{output_prefix}_head.json"
    tail_output = f"{output_prefix}_tail.json"
    export_direction(ranker, "head", args.topk, head_output)
    export_direction(ranker, "tail", args.topk, tail_output)


def main():
    args = parse_args()
    if args.worker_threads is None or int(args.worker_threads) <= 0:
        args.worker_threads = max(int(os.cpu_count() or 1), 1)

    os.makedirs(args.output_dir, exist_ok=True)
    relation_to_id = read_relation_to_id_map(args.train)
    opts = build_options(args)
    loader = Loader(options=opts.get("loader"))
    ranker = RankingHandler(options=opts.get("ranking_handler"))

    # Load train/filter once, and keep only target reloading in the per-relation loop.
    empty_target = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    empty_target_path = empty_target.name
    empty_target.close()

    if args.valid.endswith("empty.txt"):
        loader.load_data(data=args.train, target=empty_target_path)
    else:
        loader.load_data(data=args.train, filter=args.valid, target=empty_target_path)
    loader.load_rules(rules=args.rules)

    tmp_dir = tempfile.mkdtemp(prefix="apply_by_relation_")
    try:
        relation_to_target = split_target_by_relation(args.target, tmp_dir)
        relations = sorted(relation_to_target.keys())

        print(f"[apply_by_relation] split into {len(relations)} relations", flush=True)
        for idx, relation in enumerate(relations, start=1):
            print(f"[apply_by_relation] {idx}/{len(relations)}", flush=True)
            run_one_relation(
                args,
                loader,
                ranker,
                relation,
                relation_to_id,
                relation_to_target[relation],
            )
    finally:
        try:
            os.remove(empty_target_path)
        except OSError:
            pass
        for name in os.listdir(tmp_dir):
            p = os.path.join(tmp_dir, name)
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
