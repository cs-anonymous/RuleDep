#!/usr/bin/env python
import argparse
from collections import defaultdict
import gc
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Split processed explanation pickles into per-relation files.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--application_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--splits", default="train,valid,test", help="Comma-separated splits to process.")
    return parser.parse_args()


def infer_relation_from_key(key, direction):
    return int(key[1]) if direction == "o" else int(key[0])


def split_one_file(input_path, output_root, split_name, direction):
    file_name = os.path.basename(input_path)
    print(f"Loading {input_path} ...")
    processed = pickle.load(open(input_path, "rb"))
    print(f"Loaded {file_name} with {len(processed)} keys")

    buckets = defaultdict(dict)
    for index, (key, value) in enumerate(processed.items(), start=1):
        relation = infer_relation_from_key(key, direction)
        buckets[relation][key] = value
        if index % 500000 == 0:
            print(f"  partitioned {index}/{len(processed)} keys")

    print(f"Writing {len(buckets)} relation files for {file_name}")
    for relation, relation_processed in sorted(buckets.items()):
        relation_dir = os.path.join(output_root, str(int(relation)))
        os.makedirs(relation_dir, exist_ok=True)
        output_path = os.path.join(relation_dir, file_name)
        with open(output_path, "wb") as fout:
            pickle.dump(relation_processed, fout, protocol=pickle.HIGHEST_PROTOCOL)

    del buckets
    del processed
    gc.collect()
    print(f"Finished {file_name}")


def main():
    args = parse_args()
    dataset_dir = os.path.join(args.data_root, args.dataset)
    application_dir = args.application_dir or os.path.join(dataset_dir, "application")
    output_dir = args.output_dir or os.path.join(application_dir, "relation")
    os.makedirs(output_dir, exist_ok=True)

    splits = [split.strip() for split in str(args.splits).split(",") if split.strip()]
    directions = [("o", "processed_sp"), ("s", "processed_po")]

    for split_name in splits:
        for direction, prefix in directions:
            input_path = os.path.join(application_dir, f"{prefix}_{split_name}.pkl")
            if not os.path.exists(input_path):
                print(f"Skip missing {input_path}")
                continue
            split_one_file(input_path, output_dir, split_name, direction)

    print(f"Per-relation processed explanations saved under {output_dir}")


if __name__ == "__main__":
    main()
