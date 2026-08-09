#!/usr/bin/env python3
"""
Calculaterule.txtThe indicators of all rules in and output asCSV (multi-process version)
"""

import os
import sys
import csv
import time
from typing import List, Dict
from multiprocessing import Pool, Manager, Lock
from collections import defaultdict

# importanalysis_rulemodule
from analysis_rule import KnowledgeGraph, RuleParser, RuleSupportCalculator


def load_dataset(filepath: str) -> KnowledgeGraph:
    """Load the data set into the knowledge graph"""
    kg = KnowledgeGraph()
    
    print(f"Loading dataset: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file does not exist: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Chapter{line_num}Row format error: {line}")
                continue
            
            head, relation, tail = parts
            kg.add_triple(head, relation, tail)
            
            if line_num % 50000 == 0:
                print(f"Loaded {line_num:,} triples...")
    
    print(f"Data set loading completed:")
    print(f"  Number of triples: {len(kg.triples):,}")
    print(f"  Number of entities: {kg._next_entity_id:,}")
    print(f"  Original number of relations: {len([r for r in kg.relations if not r.startswith('INVERSE_')]):,}")
    print(f"  Total number of relationships (including inverse relationships): {len(kg.relations):,}")
    
    return kg


def load_rules_from_file(filepath: str, filter_normal: bool = False) -> List[Dict]:
    """
    fromrule.txtLoad rules from file
    
    File format:
    bodySize\tsupport\tconfidence\trule_string
    
    Args:
        filepath: Rule file path
        filter_normal: if forTrue, Filter out simple rules and keep only complex rules
    
    Returns:
        Rule list
    """
    rules = []
    filtered_count = 0
    
    print(f"\nLoading rules file: {filepath}")
    if filter_normal:
        print("Filter mode: only load complex rules (rules containing semicolons)")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Rules file does not exist: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 4:
                print(f"Warning: Chapter{line_num}Row format error: {line}")
                continue
            
            try:
                bodySize = int(parts[0])
                support = int(parts[1])
                confidence = float(parts[2])
                rule_string = parts[3]
                
                # If filtering is enabled, simple rules are skipped
                if filter_normal and not is_complex_rule(rule_string):
                    filtered_count += 1
                    continue
                
                rules.append({
                    'bodySize': bodySize,
                    'support': support,
                    'confidence': confidence,
                    'rule_string': rule_string,
                    'line_num': line_num
                })
            except ValueError as e:
                print(f"Warning: Chapter{line_num}Line parsing error: {e}")
                continue
    
    if filter_normal:
        print(f"filter out {filtered_count} simple rule")
    print(f"loaded {len(rules)} rules")
    return rules


def is_complex_rule(rule_string: str) -> bool:
    """Determine whether it iscomplex rule (bodyincluding semicolon)"""
    if '<=' not in rule_string:
        return False
    
    _, body_part = rule_string.split('<=', 1)
    return ';' in body_part


def classify_rule_type(rule_string: str) -> str:
    """
    Classification rule type:unaryorbinary
    """
    if '<=' not in rule_string:
        return 'unknown'
    
    head_part, _ = rule_string.split('<=', 1)
    head_part = head_part.strip()
    
    # CheckheadWhether to include variables
    if '(' in head_part and ')' in head_part:
        paren_content = head_part.split('(')[1].split(')')[0]
        # If there is only one parameter in the brackets (without commas), it is a unary rule
        if ',' not in paren_content:
            return 'unary'
        else:
            # If there is a comma, check whether there are two variables (binary) or one variable and a constant (unary)
            args = [arg.strip() for arg in paren_content.split(',')]
            # Count the number of single-letter variables (real variables)
            var_count = sum(1 for arg in args if len(arg) == 1)
            return 'unary' if var_count == 1 else 'binary'
    else:
        # Without parentheses, it is a shorthand binary rule.
        return 'binary'


def get_rule_category(rule_string: str) -> str:
    """
    Get the full breakdown of rules:normal/complex + unary/binary
    """
    is_complex = is_complex_rule(rule_string)
    rule_type = classify_rule_type(rule_string)
    
    prefix = 'complex' if is_complex else 'normal'
    return f"{prefix} {rule_type}"


def calculate_rule_attributes(rule_string: str) -> dict:
    """
    Properties of calculation rules:branch, depth, length
    
    - branch: bodyNumber of branches in (separated by semicolon)
      - branch=1: normal rules
      - branch>1: complex rules
    
    - depth: 
      - normal rules: depth = length = #bodyAtoms
      - complex rules: depth = max(#bodyAtoms for each body)
    
    - length:
      - normal rules: length = #bodyAtoms
      - complex rules: length = sum(#bodyAtoms for each body)
    
    Args:
        rule_string: rule string
    
    Returns:
        contains branch, depth, length dictionary
    """
    if '<=' not in rule_string:
        return {'branch': 0, 'depth': 0, 'length': 0}
    
    _, body_part = rule_string.split('<=', 1)
    body_part = body_part.strip()
    
    # Check if it isU0rules (emptybody) 
    if not body_part or body_part == '':
        return {'branch': 0, 'depth': 0, 'length': 0}
    
    # Check if it iscomplex rule (bodyincluding semicolon)
    if ';' in body_part:
        # Complex rule: Have multiple branches
        branches = [b.strip() for b in body_part.split(';')]
        branch = len(branches)
        
        # Count the number of atoms in each branch
        branch_atom_counts = []
        for b in branches:
            # Calculatebodynumber of atoms
            if '*' in b:
                # Abbreviation format: pass*Number of connected relationships
                atom_count = b.count('*') + 1
            elif ', ' in b:
                # Full format: used between atoms", "separated (comma+space)
                atom_count = b.count(', ') + 1
            else:
                # single atom
                atom_count = 1
            branch_atom_counts.append(atom_count)
        
        depth = max(branch_atom_counts)
        length = sum(branch_atom_counts)
    else:
        # Normal rule: only one branch
        branch = 1
        
        # Calculatebodynumber of atoms
        if '*' in body_part:
            # Abbreviation format: pass*Number of connected relationships
            atom_count = body_part.count('*') + 1
        elif ', ' in body_part:
            # Full format: used between atoms", "separated (comma+space)
            atom_count = body_part.count(', ') + 1
        else:
            # single atom
            atom_count = 1
        
        depth = atom_count
        length = atom_count
    
    return {'branch': branch, 'depth': depth, 'length': length}


def calculate_match_level(original_conf: float, calculated_conf: float) -> int:
    """
    Calculate matching degree
    
    Args:
        original_conf: original confidence
        calculated_conf: Calculated confidence
        
    Returns:
        0: identical (completely consistent)
        1: close (difference < 10%)
        2: similar (difference < 20%)
        3: unmatched (difference >= 20%)
    """
    diff = abs(calculated_conf - original_conf)
    
    if diff == 0:
        return 0
    elif diff < 0.1:
        return 1
    elif diff < 0.2:
        return 2
    else:
        return 3


def calculate_rule_metrics(rule: Dict, kg: KnowledgeGraph) -> Dict:
    """
    Calculate metrics for a single rule
    
    Returns:
        The result dictionary contains:
        {
            'originalRule': str,
            'simplifiedRule': str,
            'originalMetric': dict,
            'calculatedMetric': dict,
            'match': int,
            'category': str,
            'success': bool,
            'error': str (if failed)
        }
    """
    rule_string = rule['rule_string']
    original_metric = {
        'bodySize': rule['bodySize'],
        'support': rule['support'],
        'confidence': rule['confidence']
    }
    
    # Get rule classification
    category = get_rule_category(rule_string)
    
    # Compute rule properties
    attributes = calculate_rule_attributes(rule_string)
    
    try:
        # parsing rules
        head_relation, body_relations, variable_count, rule_info = RuleParser.parse_rule(rule_string)
        
        simplified_rule = rule_info.get('normalized_rule', rule_string)
        
        # Create a calculator
        calculator = RuleSupportCalculator(kg)
        
        # Calculate actual indicators
        result = calculator.calculate_rule_support_join(rule_info)
        
        calculated_metric = {
            'headSize': result['headSize'],
            'bodySize': result['bodySize'],
            'support': result['support'],
            'confidence': result['confidence']
        }
        
        # Calculate matching degree
        match = calculate_match_level(original_metric['confidence'], calculated_metric['confidence'])
        
        # Calculate deviation value
        deviation = original_metric['confidence'] - calculated_metric['confidence']
        
        return {
            'originalRule': rule_string,
            'simplifiedRule': simplified_rule,
            'originalMetric': str(original_metric),
            'calculatedMetric': str(calculated_metric),
            'match': match,
            'category': category,
            'branch': attributes['branch'],
            'depth': attributes['depth'],
            'length': attributes['length'],
            'deviation': deviation,
            'success': True,
            'line_num': rule.get('line_num', 0)
        }
        
    except Exception as e:
        return {
            'originalRule': rule_string,
            'simplifiedRule': 'ERROR',
            'originalMetric': str(original_metric),
            'calculatedMetric': 'ERROR',
            'match': 3,
            'category': category,
            'branch': attributes['branch'],
            'depth': attributes['depth'],
            'length': attributes['length'],
            'deviation': 0,
            'success': False,
            'error': str(e),
            'line_num': rule.get('line_num', 0)
        }


# Global variables, used in multiple processes
_global_kg = None

def init_worker(dataset_path):
    """Initialize the work process and load the knowledge graph"""
    global _global_kg
    _global_kg = load_dataset(dataset_path)


def process_rule_wrapper(rule):
    """Wrapper function for multi-process processing rules"""
    global _global_kg
    return calculate_rule_metrics(rule, _global_kg)


def print_statistics_matrix(results: List[Dict]):
    """Print statistical matrix"""
    categories = ['normal unary', 'normal binary', 'complex unary', 'complex binary']
    
    stats = {}
    for category in categories:
        category_results = [r for r in results if r.get('category') == category]
        
        total = len(category_results)
        if total == 0:
            stats[category] = {
                'identical': 0,
                'close': 0,
                'similar': 0,
                'unmatch': 0,
                'total': 0
            }
            continue
        
        # identical: match=0
        identical = sum(1 for r in category_results if r['match'] == 0)
        
        # close: match=1
        close = sum(1 for r in category_results if r['match'] == 1)
        
        # similar: match=2
        similar = sum(1 for r in category_results if r['match'] == 2)
        
        # unmatch: match=3
        unmatch = sum(1 for r in category_results if r['match'] == 3)
        
        stats[category] = {
            'identical': identical,
            'close': close,
            'similar': similar,
            'unmatch': unmatch,
            'total': total
        }
    
    # Print statistical matrix
    print("\n" + "="*100)
    print("statistical matrix")
    print("="*100)
    print()
    print(f"{'Category':<20} {'Identical':<12} {'Close':<12} {'Similar':<12} {'Unmatch':<12} {'Total':<12}")
    print("-" * 100)
    
    for category in categories:
        s = stats[category]
        print(f"{category:<20} {s['identical']:<12} {s['close']:<12} {s['similar']:<12} {s['unmatch']:<12} {s['total']:<12}")
    
    # print total
    total_identical = sum(s['identical'] for s in stats.values())
    total_close = sum(s['close'] for s in stats.values())
    total_similar = sum(s['similar'] for s in stats.values())
    total_unmatch = sum(s['unmatch'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    
    print("-" * 100)
    print(f"{'Total':<20} {total_identical:<12} {total_close:<12} {total_similar:<12} {total_unmatch:<12} {total_all:<12}")
    print()
    
    if total_all > 0:
        print(f"Identicalrate: {total_identical/total_all*100:.1f}%")
        print(f"Closerate: {total_close/total_all*100:.1f}%")
        print(f"Similarrate: {total_similar/total_all*100:.1f}%")
        print(f"Unmatchrate: {total_unmatch/total_all*100:.1f}%")
    print("="*100)


def process_all_rules(rules: List[Dict], dataset_path: str, output_csv: str, num_processes: int = 20):
    """
    Use multiprocessing to process all rules and output toCSV
    
    Args:
        rules: Rule list
        dataset_path: Dataset path
        output_csv: outputCSVfile path
        num_processes: Number of processes
    """
    print(f"\nStart processing {len(rules)} rules, use {num_processes} processes...")
    
    all_results = []
    save_interval = 500  # per treatment500Rules are saved once
    
    start_time = time.time()
    
    # Create process pool
    with Pool(processes=num_processes, initializer=init_worker, initargs=(dataset_path,)) as pool:
        # Useimap_unorderedto get the results so you can process them one by one without waiting for all the results
        for i, result in enumerate(pool.imap_unordered(process_rule_wrapper, rules, chunksize=10), 1):
            all_results.append(result)
            
            # every100bar showing progress
            if i % 100 == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed
                remaining = (len(rules) - i) / speed if speed > 0 else 0
                print(f"Progress: {i}/{len(rules)} ({i/len(rules)*100:.1f}%) - "
                      f"speed: {speed:.1f} rules/s - "
                      f"Estimated remaining: {remaining/60:.1f} minutes")
            
            # everysave_intervalSave results once
            if i % save_interval == 0:
                save_results_to_csv(all_results, output_csv)
                print(f"\nIntermediate results saved ({i} rules)")
                print_statistics_matrix(all_results)
    
    # Save final result
    save_results_to_csv(all_results, output_csv)
    
    elapsed = time.time() - start_time
    print(f"\nProcessing completed! Total time spent: {elapsed/60:.1f} minutes")
    print(f"Results have been saved to: {output_csv}")
    
    # Print final statistics
    print_statistics_matrix(all_results)
    
    # Print detailed statistics
    print_detailed_statistics(all_results)


def save_results_to_csv(results: List[Dict], output_csv: str):
    """Save results toCSVFile"""
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['originalRule', 'simplifiedRule', 'originalMetric', 'calculatedMetric', 
                      'match', 'ruleType', 'branch', 'depth', 'length', 'deviation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'originalRule': result['originalRule'],
                'simplifiedRule': result['simplifiedRule'],
                'originalMetric': result['originalMetric'],
                'calculatedMetric': result['calculatedMetric'],
                'match': result['match'],
                'ruleType': result.get('category', 'unknown'),
                'branch': result.get('branch', 0),
                'depth': result.get('depth', 0),
                'length': result.get('length', 0),
                'deviation': result.get('deviation', 0)
            })


def print_detailed_statistics(results: List[Dict]):
    """Print detailed statistics"""
    stats = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'match_0': sum(1 for r in results if r['match'] == 0),
        'match_1': sum(1 for r in results if r['match'] == 1),
        'match_2': sum(1 for r in results if r['match'] == 2),
        'match_3': sum(1 for r in results if r['match'] == 3)
    }
    
    print(f"\nDetailed statistics:")
    print(f"  Total number of rules: {stats['total']}")
    print(f"  Processed successfully: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"  Processing failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"\nMatch degree distribution:")
    print(f"  Identical (0): {stats['match_0']} ({stats['match_0']/stats['total']*100:.1f}%)")
    print(f"  Close (1, <10%): {stats['match_1']} ({stats['match_1']/stats['total']*100:.1f}%)")
    print(f"  Similar (2, <20%): {stats['match_2']} ({stats['match_2']/stats['total']*100:.1f}%)")
    print(f"  Unmatched (3, >=20%): {stats['match_3']} ({stats['match_3']/stats['total']*100:.1f}%)")


def main():
    """main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculaterule.txtMetrics for all rules in')
    parser.add_argument('--rule_file', type=str, help='Rule file path')
    parser.add_argument('--dataset', type=str, help='Dataset path')
    parser.add_argument('--output', type=str, help='outputCSVfile path')
    parser.add_argument('--num_processes', type=int, default=28, help='Number of processes, default20')
    parser.add_argument('--filter_normal', action='store_true', help='Filter out simple rules and only process complex rules')
    
    args = parser.parse_args()
    
    # If no parameters are provided, default values are used
    dataset_path = args.dataset if args.dataset else "data/FB15k-237/train.txt"
    rules_path = args.rule_file if args.rule_file else "out/FB15k-237/rule.txt"
    output_csv = args.output if args.output else "out/FB15k-237/rule_metrics.csv"
    num_processes = args.num_processes
    filter_normal = args.filter_normal
    
    print("="*100)
    print("Rule indicator calculation script (multi-process version)")
    print("="*100)
    print(f"Dataset: {dataset_path}")
    print(f"rules file: {rules_path}")
    print(f"outputCSV: {output_csv}")
    print(f"Number of processes: {num_processes}")
    print(f"Filter simple rules: {filter_normal}")
    print("="*100)
    
    try:
        # Load rules (optional filter simple rules)
        rules = load_rules_from_file(rules_path, filter_normal)
        
        # Randomly shuffle the order of rules
        import random
        random.shuffle(rules)
        print(f"The order of rules has been randomly shuffled")
        
        # Use multiprocessing to process all rules and outputCSV
        process_all_rules(rules, dataset_path, output_csv, num_processes)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
