#!/usr/bin/env python
import argparse
from pathlib import Path

from pykeen.datasets import Hetionet


def write_split(path: Path, triples) -> None:
    with path.open("w", encoding="utf-8") as file:
        for h, r, t in triples:
            file.write(f"{h}\t{r}\t{t}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyKEEN Hetionet splits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hetionet"),
        help="Directory where train/valid/test txt files will be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Hetionet(create_inverse_triples=False)
    split_to_triples = {
        "train.txt": dataset.training.triples,
        "valid.txt": dataset.validation.triples,
        "test.txt": dataset.testing.triples,
    }

    for filename, triples in split_to_triples.items():
        write_split(output_dir / filename, triples)

    print(f"Exported Hetionet splits to {output_dir}")
    print(
        "Sizes:",
        {
            "train": dataset.training.num_triples,
            "valid": dataset.validation.num_triples,
            "test": dataset.testing.num_triples,
            "entities": dataset.num_entities,
            "relations": dataset.num_relations,
        },
    )


if __name__ == "__main__":
    main()
