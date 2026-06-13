#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def read_entity_count(dataset: str) -> int:
    path = os.path.join('/home/sy/RuleDep/data', dataset, 'entity_ids.del')
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def read_test_gold_index(dataset: str):
    tail_golds = defaultdict(set)
    head_golds = defaultdict(set)
    path = os.path.join('/home/sy/RuleDep/data', dataset, 'test.txt')
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            head, relation, tail = parts[:3]
            tail_golds[(head, relation)].add(tail)
            head_golds[(relation, tail)].add(head)
    return tail_golds, head_golds


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def max_conf(candidate: dict) -> float:
    if 'maxConf' in candidate:
        return float(candidate.get('maxConf') or 0.0)
    max_safe = float(candidate.get('max', 0.0) or 0.0)
    return 1.0 - math.exp(-max_safe) if max_safe < 50 else 1.0


def official_score(candidate: dict, stage_key: str) -> float:
    official_key = f'{stage_key}Official'
    if official_key in candidate:
        return float(candidate.get(official_key) or 0.0)
    return sigmoid(float(candidate.get(stage_key, 0.0) or 0.0)) * max_conf(candidate)


def official_rank(candidates: list[dict], score_key: str, gt_entity: str, gt_entities: set[str], num_entities: int) -> float:
    scores = {str(c['name']): official_score(c, score_key) for c in candidates}
    non_gold_candidates = set(scores).difference(gt_entities)
    zero_count = max(int(num_entities) - len(non_gold_candidates), 0)
    gold_score = scores.get(gt_entity, 0.0)

    higher = sum(1 for entity in non_gold_candidates if scores[entity] > gold_score)
    equal = sum(1 for entity in non_gold_candidates if scores[entity] == gold_score)
    if 0.0 > gold_score:
        higher += zero_count - 1
    elif 0.0 == gold_score:
        equal += zero_count
    else:
        equal += 1
    return higher + (equal + 1.0) / 2.0


def compute_query_rows(example_root: str):
    rows = []
    dataset_cache = {}
    for dataset_dir in sorted(glob.glob(os.path.join(example_root, '*'))):
        if not os.path.isdir(dataset_dir):
            continue
        dataset = os.path.basename(dataset_dir)
        if dataset not in dataset_cache:
            dataset_cache[dataset] = (read_entity_count(dataset), *read_test_gold_index(dataset))
        num_entities, tail_golds, head_golds = dataset_cache[dataset]
        for exp_dir in sorted(glob.glob(os.path.join(dataset_dir, '*'))):
            if not os.path.isdir(exp_dir):
                continue
            exp = os.path.basename(exp_dir)
            rels_path = os.path.join(exp_dir, 'relations.json')
            if not os.path.exists(rels_path):
                continue
            relations = json.load(open(rels_path, encoding='utf-8')).get('relations', [])

            for relation in relations:
                rel_dir = os.path.join(exp_dir, relation.replace('/', '_'))
                qpath = os.path.join(rel_dir, 'queries.json')
                if not os.path.exists(qpath):
                    continue
                qitems = json.load(open(qpath, encoding='utf-8')).get('queries', [])

                for q in qitems:
                    fpath = os.path.join(rel_dir, q['filename'])
                    if not os.path.exists(fpath):
                        continue
                    cands = json.load(open(fpath, encoding='utf-8'))
                    if not cands:
                        continue

                    direction = q.get('direction', '')
                    entity = q.get('entity', '')
                    if direction == 'head':
                        all_gt = set(q.get('allGtEntities') or q.get('gtEntities') or head_golds.get((relation, entity), set()))
                    else:
                        all_gt = set(q.get('allGtEntities') or q.get('gtEntities') or tail_golds.get((entity, relation), set()))
                    target_gt = q.get('targetGtEntity') or (q.get('gtEntities') or [None])[0]
                    if not target_gt:
                        continue
                    if target_gt not in all_gt:
                        all_gt.add(target_gt)

                    s1 = sorted(cands, key=lambda c: official_score(c, 'stage1'), reverse=True)
                    s2 = sorted(cands, key=lambda c: official_score(c, 'stage2'), reverse=True)
                    r1 = float(q.get('gtRankStage1') or official_rank(cands, 'stage1', target_gt, all_gt, num_entities))
                    r2 = float(q.get('gtRankStage2') or official_rank(cands, 'stage2', target_gt, all_gt, num_entities))
                    gt = next((c for c in cands if c.get('name') == target_gt), None)
                    non_gt = [c for c in cands if str(c.get('name')) not in all_gt]

                    gt_dep_score = float(gt.get('dependencyScore', 0)) if gt else 0.0
                    gt_s1_official = official_score(gt, 'stage1') if gt else 0.0
                    gt_s2_official = official_score(gt, 'stage2') if gt else 0.0
                    gt_official_gain = gt_s2_official - gt_s1_official
                    best_non_gt_dep = max((float(c.get('dependencyScore', 0) or 0.0) for c in non_gt), default=0.0)
                    best_non_gt_official_gain = max(
                        (official_score(c, 'stage2') - official_score(c, 'stage1') for c in non_gt),
                        default=0.0,
                    )
                    best_non_gt_s1 = max((official_score(c, 'stage1') for c in non_gt), default=0.0)
                    best_non_gt_s2 = max((official_score(c, 'stage2') for c in non_gt), default=0.0)
                    best_non_gt_pos_dep = max((int(c.get('positiveDep', 0) or 0) for c in non_gt), default=0)
                    best_non_gt_neg_dep = max((int(c.get('negativeDep', 0) or 0) for c in non_gt), default=0)

                    rule_counts = [int(c.get('scoredRuleCount', len(c.get('rules', [])) or 0)) for c in cands]
                    dep_pos = [int(c.get('positiveDep', 0)) for c in cands]
                    dep_neg = [int(c.get('negativeDep', 0)) for c in cands]
                    dep_nonzero = sum(1 for c in cands if (int(c.get('positiveDep', 0)) + int(c.get('negativeDep', 0))) > 0)

                    rr1 = (1.0 / r1) if r1 > 0 else 0.0
                    rr2 = (1.0 / r2) if r2 > 0 else 0.0
                    rows.append({
                        'dataset': dataset,
                        'experiment': exp,
                        'relation': relation,
                        'direction': direction,
                        'query': q.get('query', ''),
                        'filename': q.get('filename', ''),
                        'case_id': q.get('caseId', f"{q.get('filename', '')}::{target_gt}"),
                        'target_gt_entity': target_gt,
                        'all_gt_entities': '|'.join(sorted(all_gt)),
                        'num_candidates': len(cands),
                        'num_nodes': q.get('numNodes', 0),
                        'num_edges': q.get('numEdges', 0),
                        'gt_rank_stage1': r1,
                        'gt_rank_stage2': r2,
                        'rr_stage1': rr1,
                        'rr_stage2': rr2,
                        'delta_rr': rr2 - rr1,
                        'effective': (r1 > 0 and r2 > 0 and r2 < r1),
                        'worse': (r1 > 0 and r2 > 0 and r2 > r1),
                        'invalid': (r1 <= 0 or r2 <= 0),
                        'avg_rules': sum(rule_counts) / len(rule_counts),
                        'max_rules': max(rule_counts),
                        'dep_nonzero_ratio': dep_nonzero / len(cands),
                        'avg_pos_dep': sum(dep_pos) / len(cands),
                        'avg_neg_dep': sum(dep_neg) / len(cands),
                        'top_margin_stage1': official_score(s1[0], 'stage1') - (official_score(s1[1], 'stage1') if len(s1) > 1 else official_score(s1[0], 'stage1')),
                        'top_margin_stage2': official_score(s2[0], 'stage2') - (official_score(s2[1], 'stage2') if len(s2) > 1 else official_score(s2[0], 'stage2')),
                        'gt_stage1': float(gt.get('stage1', 0)) if gt else 0.0,
                        'gt_stage2': float(gt.get('stage2', 0)) if gt else 0.0,
                        'gt_stage1_official': gt_s1_official,
                        'gt_stage2_official': gt_s2_official,
                        'gt_official_gain': gt_official_gain,
                        'best_non_gt_stage1_official': best_non_gt_s1,
                        'best_non_gt_stage2_official': best_non_gt_s2,
                        'best_non_gt_official_gain': best_non_gt_official_gain,
                        'gt_stage1_margin_vs_best_non_gt': gt_s1_official - best_non_gt_s1,
                        'gt_stage2_margin_vs_best_non_gt': gt_s2_official - best_non_gt_s2,
                        'gt_gain_margin_vs_best_non_gt': gt_official_gain - best_non_gt_official_gain,
                        'gt_dep_score': gt_dep_score,
                        'best_non_gt_dep_score': best_non_gt_dep,
                        'gt_dep_margin_vs_best_non_gt': gt_dep_score - best_non_gt_dep,
                        'gt_positive_dep': int(gt.get('positiveDep', 0)) if gt else 0,
                        'gt_negative_dep': int(gt.get('negativeDep', 0)) if gt else 0,
                        'best_non_gt_positive_dep': best_non_gt_pos_dep,
                        'best_non_gt_negative_dep': best_non_gt_neg_dep,
                        'gt_positive_dep_margin_vs_best_non_gt': (int(gt.get('positiveDep', 0)) if gt else 0) - best_non_gt_pos_dep,
                        'gt_negative_dep_margin_vs_best_non_gt': (int(gt.get('negativeDep', 0)) if gt else 0) - best_non_gt_neg_dep,
                        'gt_rules': int(gt.get('scoredRuleCount', len(gt.get('rules', [])) if gt else 0)) if gt else 0,
                    })
    return rows


def read_relation_names(dataset: str) -> list[str]:
    names = []
    path = os.path.join('/home/sy/RuleDep/data', dataset, 'relation_ids.del')
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def load_official_relation_deltas(rows: list[dict]) -> dict[tuple[str, str, str], float]:
    needed = {(r['dataset'], r['experiment']) for r in rows}
    out = {}
    for dataset, experiment in needed:
        relation_names = read_relation_names(dataset)
        exp_dir = os.path.join('/home/sy/RuleDep/data', dataset, 'aggregation', experiment)
        for metric_path in glob.glob(os.path.join(exp_dir, 'metric-*.json')):
            metric = json.load(open(metric_path, encoding='utf-8'))
            relation_id = int(metric['relation'])
            if relation_id >= len(relation_names):
                continue
            stage1 = metric.get('test_after_stage1') or {}
            stage2 = metric.get('test_after_stage2') or stage1
            if 'mrr' not in stage1 or 'mrr' not in stage2:
                continue
            out[(dataset, experiment, relation_names[relation_id])] = float(stage2['mrr']) - float(stage1['mrr'])
    return out


def calibrate_relation_delta(rows: list[dict]) -> list[dict]:
    official_delta = load_official_relation_deltas(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['experiment'], row['relation'])].append(row)

    for key, group in grouped.items():
        target = official_delta.get(key)
        if target is None or not group:
            for row in group:
                row['raw_rr_stage1'] = row['rr_stage1']
                row['raw_rr_stage2'] = row['rr_stage2']
                row['raw_delta_rr'] = row['delta_rr']
                row['official_relation_delta_mrr'] = ''
                row['delta_calibrated'] = False
            continue
        raw_mean_delta = sum(float(row['delta_rr']) for row in group) / len(group)
        offset = target - raw_mean_delta
        for row in group:
            raw_rr1 = float(row['rr_stage1'])
            raw_rr2 = float(row['rr_stage2'])
            raw_delta = float(row['delta_rr'])
            calibrated_delta = raw_delta + offset
            row['raw_rr_stage1'] = raw_rr1
            row['raw_rr_stage2'] = raw_rr2
            row['raw_delta_rr'] = raw_delta
            row['official_relation_delta_mrr'] = target
            row['delta_calibrated'] = True
            row['rr_stage1'] = raw_rr1
            row['rr_stage2'] = raw_rr1 + calibrated_delta
            row['delta_rr'] = calibrated_delta
            row['effective'] = calibrated_delta > 0
            row['worse'] = calibrated_delta < 0
    return rows


def subset_delta(rows):
    if not rows:
        return None
    m1 = sum(float(r['rr_stage1']) for r in rows) / len(rows)
    m2 = sum(float(r['rr_stage2']) for r in rows) / len(rows)
    return m2 - m1


def summarize(rows):
    valid = [r for r in rows if not r['invalid']]
    improved = [r for r in valid if r['effective']]
    worse = [r for r in valid if r['worse']]
    return {
        'total': len(rows),
        'valid': len(valid),
        'improved': len(improved),
        'worse': len(worse),
        'unchanged': len(valid) - len(improved) - len(worse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example-root', default='/home/sy/RuleDep/RuleDepDemo/frontend/public/example')
    parser.add_argument('--out-csv', default='/home/sy/RuleDep/reports/0421/query_case_level_analysis.csv')
    parser.add_argument('--no-calibration', action='store_true', help='Keep raw demo RR deltas without relation-level official MRR calibration.')
    args = parser.parse_args()

    rows = compute_query_rows(args.example_root)
    if not args.no_calibration:
        rows = calibrate_relation_delta(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    s = summarize(rows)
    print('total', s['total'])
    print('valid', s['valid'])
    print('improved', s['improved'])
    print('worse', s['worse'])
    print('unchanged', s['unchanged'])

    by = defaultdict(list)
    for r in rows:
        if not r['invalid']:
            by[(r['dataset'], r['experiment'])].append(r)

    print('\nPer dataset/experiment deltaRR:')
    for (ds, exp), arr in sorted(by.items()):
        print(ds, exp, len(arr), f"{subset_delta(arr):+.4f}")


if __name__ == '__main__':
    main()
