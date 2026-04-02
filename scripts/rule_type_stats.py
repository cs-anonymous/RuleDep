#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


X_VARS = {"X", "Y", "Z"}


def parse_rule_line(line: str):
    parts = line.rstrip("\n").split("\t", 3)
    if len(parts) < 4:
        return None
    try:
        body_size = int(parts[0])
        support = int(parts[1])
        score = float(parts[2])
    except ValueError:
        return None
    return {
        "body_size": body_size,
        "support": support,
        "score": score,
        "rule": parts[3],
    }


def split_head_body(rule: str) -> tuple[str, str] | None:
    if "<=" not in rule:
        return None
    head, body = rule.split("<=", 1)
    return head.strip(), body.strip()


def is_b_rule(rule: str) -> bool:
    split = split_head_body(rule)
    if split is None:
        return False
    head, _ = split
    return "(X,Y)" in head


def is_ud_rule(rule: str) -> bool:
    split = split_head_body(rule)
    if split is None:
        return False
    _, body = split
    return body.count("(A,") + body.count(",A)") == 1


def is_z_rule(rule: str) -> bool:
    split = split_head_body(rule)
    if split is None:
        return False
    _, body = split
    return body == ""


def has_head_xx(rule: str) -> bool:
    split = split_head_body(rule)
    if split is None:
        return False
    head, _ = split
    if "(" not in head or ")" not in head:
        return False
    inside = head.split("(", 1)[1].rsplit(")", 1)[0]
    args = [x.strip() for x in inside.split(",")]
    return len(args) == 2 and args[0] == args[1] and args[0] in X_VARS


def has_non_ud_aux(rule: str) -> bool:
    split = split_head_body(rule)
    if split is None:
        return False
    _, body = split
    aux_a = body.count("(A,") + body.count(",A)")
    aux_b = body.count("(B,") + body.count(",B)")
    aux_c = body.count("(C,") + body.count(",C)")
    return aux_b > 0 or aux_c > 0 or aux_a > 1


@dataclass
class FileStats:
    dataset: str
    setting: str
    path: str
    total: int = 0
    b: int = 0
    uc: int = 0
    ud: int = 0
    other: int = 0
    other_z: int = 0
    other_xx: int = 0
    other_aux: int = 0
    invalid: int = 0


def classify_rule(rule: str) -> str:
    if is_b_rule(rule):
        return "B"
    if is_z_rule(rule):
        return "other_z"
    if has_head_xx(rule):
        return "other_xx"
    if is_ud_rule(rule):
        return "Ud"
    if has_non_ud_aux(rule):
        return "other_aux"
    return "Uc"


def collect_rule_files(data_root: Path, include_rule_txt: bool) -> list[Path]:
    files: list[Path] = []
    for dataset_dir in sorted(data_root.iterdir()):
        rules_dir = dataset_dir / "rules"
        if not rules_dir.is_dir():
            continue
        for path in sorted(rules_dir.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith("rules-") or (include_rule_txt and name == "rule.txt"):
                files.append(path)
    return files


def scan_file(path: Path) -> FileStats:
    stats = FileStats(
        dataset=path.parts[-3],
        setting=path.name,
        path=str(path),
    )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_rule_line(line)
            if parsed is None:
                stats.invalid += 1
                continue
            stats.total += 1
            cls = classify_rule(parsed["rule"])
            if cls == "B":
                stats.b += 1
            elif cls == "Uc":
                stats.uc += 1
            elif cls == "Ud":
                stats.ud += 1
            elif cls == "other_z":
                stats.other += 1
                stats.other_z += 1
            elif cls == "other_xx":
                stats.other += 1
                stats.other_xx += 1
            elif cls == "other_aux":
                stats.other += 1
                stats.other_aux += 1
            else:
                raise ValueError(f"Unexpected class {cls}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Count rule types across dataset rule files")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-rule-txt", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = collect_rule_files(data_root, include_rule_txt=args.include_rule_txt)
    rows = [scan_file(path) for path in files]

    header = [
        "dataset",
        "setting",
        "total",
        "B",
        "Uc",
        "Ud",
        "other",
        "other_Z",
        "other_Uxx",
        "other_aux",
        "invalid",
        "path",
    ]
    out = csv.writer(open(args.output, "w", newline="", encoding="utf-8")) if args.output else csv.writer(__import__("sys").stdout)
    out.writerow(header)
    for row in rows:
        out.writerow(
            [
                row.dataset,
                row.setting,
                row.total,
                row.b,
                row.uc,
                row.ud,
                row.other,
                row.other_z,
                row.other_xx,
                row.other_aux,
                row.invalid,
                row.path,
            ]
        )


if __name__ == "__main__":
    main()
