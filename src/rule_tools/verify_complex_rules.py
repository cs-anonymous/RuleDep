#!/usr/bin/env python3
"""
Complex RulesValidation script
fromrule.txtloading incomplex rules, Take a sample and verify that its confidence level is correct
set PYTHONIOENCODING=utf-8
"""

import os
import sys
import random
from typing import List, Dict, Tuple

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


def load_rules_from_file(filepath: str) -> List[Dict]:
    """
    fromrule.txtLoad rules from file
    
    File format:
    bodySize\tsupport\tconfidence\trule_string
    
    Returns:
        List of rules, each rule contains:{
            'bodySize': int,
            'support': int,
            'confidence': float,
            'rule_string': str
        }
    """
    rules = []
    
    print(f"\nLoading rules file: {filepath}")
    
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
    
    print(f"loaded {len(rules)} rules")
    return rules


def is_complex_rule(rule_string: str) -> bool:
    """Determine whether it iscomplex rule (bodyincluding semicolon)"""
    if '<=' not in rule_string:
        return False
    
    _, body_part = rule_string.split('<=', 1)
    return ';' in body_part


def classify_rule(rule_string: str) -> str:
    """
    Classification rule type:unaryorbinary
    
    by parsing rulesheadpartial judgment
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


def sample_rules(rules: List[Dict], n_unary: int = 50, n_binary: int = 50, 
                 is_complex: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Sample from a list of rulesrules
    
    Args:
        rules: Rule list
        n_unary: Number of unary rules to sample
        n_binary: Number of binary rules sampled
        is_complex: Truerepresents samplingcomplex rules, Falserepresents samplingnormal rules
        
    Returns:
        (unary_samples, binary_samples)
    """
    rule_type_name = "complex" if is_complex else "normal"
    
    # Filter corresponding typesrules
    if is_complex:
        filtered_rules = [rule for rule in rules if is_complex_rule(rule['rule_string'])]
    else:
        filtered_rules = [rule for rule in rules if not is_complex_rule(rule['rule_string'])]
    
    print(f"\nfound {len(filtered_rules)} Article{rule_type_name} rules")
    
    # Classified by type
    unary_rules = [rule for rule in filtered_rules if classify_rule(rule['rule_string']) == 'unary']
    binary_rules = [rule for rule in filtered_rules if classify_rule(rule['rule_string']) == 'binary']
    
    print(f"  one yuan{rule_type_name} rules: {len(unary_rules)}")
    print(f"  Binary{rule_type_name} rules: {len(binary_rules)}")
    
    # sampling
    unary_samples = random.sample(unary_rules, min(n_unary, len(unary_rules)))
    binary_samples = random.sample(binary_rules, min(n_binary, len(binary_rules)))
    
    # Tag rule type
    for rule in unary_samples:
        rule['is_complex'] = is_complex
        rule['rule_category'] = f"{rule_type_name} unary"
    
    for rule in binary_samples:
        rule['is_complex'] = is_complex
        rule['rule_category'] = f"{rule_type_name} binary"
    
    print(f"\nSampling results:")
    print(f"  unary rule: {len(unary_samples)}")
    print(f"  binary rule: {len(binary_samples)}")
    
    return unary_samples, binary_samples


def verify_rule(rule: Dict, kg: KnowledgeGraph, debug_instances: bool = False) -> Dict:
    """
    Verify the confidence of a single rule
    
    Args:
        debug_instances: Whether to print detailedinstanceInformation (for debugging)
    
    Returns:
        Verification result dictionary, including:{
            'rule_string': str,
            'expected_bodySize': int,
            'expected_support': int,
            'expected_confidence': float,
            'actual_bodySize': int,
            'actual_support': int,
            'actual_confidence': float,
            'bodySize_match': bool,
            'support_match': bool,
            'confidence_match': bool,
            'confidence_diff': float,
            'has_error': bool
        }
    """
    rule_string = rule['rule_string']
    expected_bodySize = rule['bodySize']
    expected_support = rule['support']
    expected_confidence = rule['confidence']
    
    try:
        # parsing rules
        head_relation, body_relations, variable_count, rule_info = RuleParser.parse_rule(rule_string)
        
        # Create a calculator
        calculator = RuleSupportCalculator(kg)
        
        # getheadandbody instancesfor detailed debugging
        head_instances = calculator._get_head_instances(rule_info)
        body_instances = calculator._get_body_instances(rule_info)
        
        # Calculate actual indicators
        result = calculator.calculate_rule_support_join(rule_info)
        
        actual_headSize = result['headSize']
        actual_bodySize = result['bodySize']
        actual_support = result['support']
        actual_confidence = result['confidence']
        
        # If it is a unary rule and needs to be debugged, print detailed information
        if debug_instances and variable_count == 1:
            print(f"\n  [DEBUG] Detailed analysis of unary rules:")
            print(f"  [DEBUG] normalization rules: {rule_info.get('normalized_rule', 'N/A')}")
            print(f"  [DEBUG] Head relation: {rule_info.get('head_relation')}")
            print(f"  [DEBUG] Head constant: {rule_info.get('head_constant')}")
            print(f"  [DEBUG] Head instancesQuantity: {len(head_instances)}")
            
            # Printhead instancesSample
            if head_instances:
                head_sample = list(head_instances)[:10]
                print(f"  [DEBUG] Head instancesSample (before10a):")
                for entity_id in head_sample:
                    entity_str = kg.get_entity_str(entity_id)
                    print(f"    - {entity_str} (id={entity_id})")
            
            print(f"\n  [DEBUG] Body instancesQuantity: {len(body_instances)}")
            
            # Printbody instancesSample
            if body_instances:
                body_sample = list(body_instances)[:10]
                print(f"  [DEBUG] Body instancesSample (before10a):")
                for entity_id in body_sample:
                    entity_str = kg.get_entity_str(entity_id)
                    print(f"    - {entity_str} (id={entity_id})")
            
            # Check intersection
            if head_instances and body_instances:
                intersection = head_instances.intersection(body_instances)
                print(f"\n  [DEBUG] intersectioninstancesQuantity: {len(intersection)}")
                if intersection:
                    intersection_sample = list(intersection)[:10]
                    print(f"  [DEBUG] intersectioninstancesSample (before10a):")
                    for entity_id in intersection_sample:
                        entity_str = kg.get_entity_str(entity_id)
                        print(f"    - {entity_str} (id={entity_id})")
                else:
                    print(f"  [DEBUG] No intersection! Check if there isbody instancesinheadin:")
                    # Check the first fewbody instancesIs thereheadin
                    body_sample = list(body_instances)[:5]
                    for entity_id in body_sample:
                        entity_str = kg.get_entity_str(entity_id)
                        in_head = entity_id in head_instances
                        print(f"    - {entity_str} (id={entity_id}): {'inheadin' if in_head else 'Not hereheadin'}")
        
        # Calculate difference
        confidence_diff = abs(actual_confidence - expected_confidence)
        
        # Determine whether it matches (confidence allows±10%error)
        confidence_match = confidence_diff <= 0.1
        bodySize_match = (actual_bodySize == expected_bodySize)
        support_match = (actual_support == expected_support)
        
        has_error = not (confidence_match and bodySize_match and support_match)
        
        return {
            'rule_string': rule_string,
            'normalized_rule': rule_info.get('normalized_rule', rule_string),
            'line_num': rule.get('line_num', 0),
            'variable_count': variable_count,
            'expected_bodySize': expected_bodySize,
            'expected_support': expected_support,
            'expected_confidence': expected_confidence,
            'actual_headSize': actual_headSize,
            'actual_bodySize': actual_bodySize,
            'actual_support': actual_support,
            'actual_confidence': actual_confidence,
            'bodySize_match': bodySize_match,
            'support_match': support_match,
            'confidence_match': confidence_match,
            'confidence_diff': confidence_diff,
            'has_error': has_error,
            'rule_category': rule.get('rule_category', 'unknown')
        }
        
    except Exception as e:
        print(f"  Error: Validation rule failed - {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'rule_string': rule_string,
            'normalized_rule': rule_string,
            'line_num': rule.get('line_num', 0),
            'variable_count': 0,
            'expected_bodySize': expected_bodySize,
            'expected_support': expected_support,
            'expected_confidence': expected_confidence,
            'actual_headSize': 0,
            'actual_bodySize': 0,
            'actual_support': 0,
            'actual_confidence': 0.0,
            'bodySize_match': False,
            'support_match': False,
            'confidence_match': False,
            'confidence_diff': 1.0,
            'has_error': True,
            'error': str(e),
            'rule_category': rule.get('rule_category', 'unknown')
        }


def verify_rules(rules: List[Dict], kg: KnowledgeGraph, rule_type: str, 
                 enable_debug: bool = False) -> List[Dict]:
    """
    Validate a set of rules
    
    Args:
        rules: Rule list
        kg: Knowledge graph
        rule_type: rule type ('unary' or 'binary') 
        enable_debug: Whether to enable verbose debugging (defaultFalse, Only facing forward3rules enabled)
        
    Returns:
        Verification result list
    """
    results = []
    
    print(f"\nStart verification{len(rules)}Article{rule_type}rules...")
    
    for i, rule in enumerate(rules, 1):
        print(f"\n{'='*100}")
        print(f"[{i}/{len(rules)}] Validation rules (OK{rule.get('line_num', 0)}) - {rule.get('rule_category', 'unknown')}")
        print(f"  original rules: {rule['rule_string']}")
        print(f"  Expectation: bodySize={rule['bodySize']}, support={rule['support']}, confidence={rule['confidence']:.4f}")
        
        # forunaryRules, only for the front3Enable verbose debugging
        debug_instances = (rule_type == 'unary' and enable_debug and i <= 3)
        
        result = verify_rule(rule, kg, debug_instances=debug_instances)
        results.append(result)
        
        if 'error' not in result:
            print(f"\n  Abbreviation rules: {result['normalized_rule']}")
            print(f"  actual: headSize={result.get('actual_headSize', 'N/A')}, bodySize={result['actual_bodySize']}, support={result['actual_support']}, confidence={result['actual_confidence']:.4f}")
            print(f"  match: bodySize={result['bodySize_match']}, support={result['support_match']}, confidence={result['confidence_match']} (diff={result['confidence_diff']:.4f})")
            
            if result['has_error']:
                print(f"  [WARNING] There is a problem with the rules!")
            else:
                print(f"  [OK] Rule verification passed")
        else:
            print(f"  [ERROR] Authentication failed: {result.get('error', 'Unknown error')}")
    
    return results


def print_summary(all_results: List[Dict]):
    """Print validation summary and statistics matrix"""
    print("\n" + "="*100)
    print("Verification summary")
    print("="*100)
    
    # Statistics grouped by category
    categories = ['normal unary', 'normal binary', 'complex unary', 'complex binary']
    
    stats = {}
    for category in categories:
        category_results = [r for r in all_results if r.get('rule_category') == category]
        
        total = len(category_results)
        if total == 0:
            stats[category] = {
                'identical': 0,
                'similar': 0,
                'unmatch': 0,
                'total': 0
            }
            continue
        
        # identical: bodySize, support, confidenceall match
        identical = sum(1 for r in category_results 
                       if r['bodySize_match'] and r['support_match'] and r['confidence_match'])
        
        # similar: confidencematch(±10%within), butbodySizeorsupportMay not match
        similar = sum(1 for r in category_results 
                     if r['confidence_match'] and not (r['bodySize_match'] and r['support_match']))
        
        # unmatch: confidencemismatch (difference>10%) 
        unmatch = sum(1 for r in category_results 
                     if not r['confidence_match'])
        
        stats[category] = {
            'identical': identical,
            'similar': similar,
            'unmatch': unmatch,
            'total': total
        }
        
        print(f"\n{category} ({total}Article):")
        print(f"  Identical (exact match): {identical} ({identical/total*100:.1f}%)")
        print(f"  Similar (similar): {similar} ({similar/total*100:.1f}%)")
        print(f"  Unmatch (no match): {unmatch} ({unmatch/total*100:.1f}%)")
    
    # Print statistical matrix
    print("\n" + "="*100)
    print("statistical matrix")
    print("="*100)
    print()
    print(f"{'Category':<20} {'Identical':<12} {'Similar':<12} {'Unmatch':<12} {'Total':<12}")
    print("-" * 100)
    
    for category in categories:
        s = stats[category]
        print(f"{category:<20} {s['identical']:<12} {s['similar']:<12} {s['unmatch']:<12} {s['total']:<12}")
    
    # print total
    total_identical = sum(s['identical'] for s in stats.values())
    total_similar = sum(s['similar'] for s in stats.values())
    total_unmatch = sum(s['unmatch'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    
    print("-" * 100)
    print(f"{'Total':<20} {total_identical:<12} {total_similar:<12} {total_unmatch:<12} {total_all:<12}")
    print()
    
    if total_all > 0:
        print(f"overall accuracy (Identical): {total_identical/total_all*100:.1f}%")
        print(f"overall acceptability rate (Identical + Similar): {(total_identical + total_similar)/total_all*100:.1f}%")
    

def main():
    """main function"""
    # Configuration path
    dataset_path = "data/FB15k-237/train.txt"
    rules_path = "out/FB15k-237/rule.txt"
    
    print("="*100)
    print("Complex RulesValidation script")
    print("="*100)
    
    try:
        # Load dataset
        kg = load_dataset(dataset_path)
        
        # Load rules
        rules = load_rules_from_file(rules_path)
        
        # samplingnormal rules (50 unary + 50 binary)
        print("\n" + "="*100)
        print("sampling Normal Rules")
        print("="*100)
        normal_unary_samples, normal_binary_samples = sample_rules(
            rules, n_unary=50, n_binary=50, is_complex=False
        )
        
        # samplingcomplex rules (50 unary + 50 binary)
        print("\n" + "="*100)
        print("sampling Complex Rules")
        print("="*100)
        complex_unary_samples, complex_binary_samples = sample_rules(
            rules, n_unary=50, n_binary=50, is_complex=True
        )
        
        # Validation rules
        all_results = []
        
        # Verifynormal unary rules
        print("\n" + "="*100)
        print("Verify Normal Unary Rules")
        print("="*100)
        normal_unary_results = verify_rules(normal_unary_samples, kg, 'unary', enable_debug=True)
        all_results.extend(normal_unary_results)
        
        # Verifynormal binary rules
        print("\n" + "="*100)
        print("Verify Normal Binary Rules")
        print("="*100)
        normal_binary_results = verify_rules(normal_binary_samples, kg, 'binary', enable_debug=False)
        all_results.extend(normal_binary_results)
        
        # Verifycomplex unary rules
        print("\n" + "="*100)
        print("Verify Complex Unary Rules")
        print("="*100)
        complex_unary_results = verify_rules(complex_unary_samples, kg, 'unary', enable_debug=True)
        all_results.extend(complex_unary_results)
        
        # Verifycomplex binary rules
        print("\n" + "="*100)
        print("Verify Complex Binary Rules")
        print("="*100)
        complex_binary_results = verify_rules(complex_binary_samples, kg, 'binary', enable_debug=False)
        all_results.extend(complex_binary_results)
        
        # Print statistical matrix
        print_summary(all_results)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
