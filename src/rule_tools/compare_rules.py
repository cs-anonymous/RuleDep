#!/usr/bin/env python3
"""
Rule comparison script
Compares two rule files and analyzes all rules with a specific relationship as a rule header
"""

import re
import os
import sys
import csv
import argparse
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict

# importanalysis_rulemodule
from analysis_rule import load_dataset, analyze_rule_from_string, RuleParser

def parse_rule_line(line: str) -> Tuple[str, Dict, str]:
    """
    parse rule line
    Return: (Complete rules, indicator dictionary, original row)
    """
    parts = line.strip().split('\t')
    if len(parts) < 4:
        return None, None, line
    
    try:
        count1 = int(parts[0])  # bodySize
        count2 = int(parts[1])  # support
        confidence = float(parts[2])
        rule = parts[3]
        
        # Build indicator dictionary
        metrics = {
            'bodySize': count1,
            'support': count2,
            'confidence': confidence
        }
        
        return rule, metrics, line
    except (ValueError, IndexError):
        pass
    
    return None, None, line

def load_rules_with_target_relation(file_path: str, args) -> Dict[str, List[Tuple[str, Dict, str]]]:
    """
    Load rules from file
    iftarget_relationforNone, Load all rules
    iftarget_relationNot forNone, Only load rules that contain the target relationship as a header
    Return: {standardized rules: [(original rules, indicator dictionary, original row)]}
    """
    rules_dict = defaultdict(list)
    
    if args.target_relation is None:
        print(f"Loading all rules from: {file_path}")
    else:
        print(f"Loading rules from: {file_path}")
        print(f"Target relation: {args.target_relation}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            target_rule_count = 0
            
            for line in f:
                if args.only_normal:
                    if '&&' in line:
                        continue
                line_count += 1
                if line_count % 100000 == 0:
                    if args.target_relation is None:
                        print(f"  Processed {line_count} lines, loaded {target_rule_count} rules")
                    else:
                        print(f"  Processed {line_count} lines, found {target_rule_count} target rules")
                
                rule, metrics, original_line = parse_rule_line(line)
                
                # iftarget_relationforNone, Load all rules; otherwise only load matching rules
                if rule and (args.target_relation is None or rule.startswith(args.target_relation)):
                    target_rule_count += 1
                    # Standardization rules (remove confidence differences for easier comparison)
                    normalized_rule = normalize_rule(rule)
                    rules_dict[normalized_rule].append((rule, metrics, original_line.strip()))
            
            print(f"  Total lines: {line_count}")
            if args.target_relation is None:
                print(f"  Loaded {target_rule_count} total rules")
            else:
                print(f"  Found {target_rule_count} rules with target relation")
            print(f"  Unique normalized rules: {len(rules_dict)}")
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return dict(rules_dict)

def normalize_rule(rule: str) -> str:
    """
    Standardize rules to remove differences in variable bindings to facilitate comparison
    Preserve the structure and relationships of rules but ignore specific entitiesIDand variable name difference
    
    Important: Convert to abbreviated format first to ensure that the same rules in different representations can be recognized as the same
    """
    # First convert to abbreviated format (unified representation)
    simplified = convert_to_simplified_format(rule)
    
    # Then normalize: remove whitespace differences
    if '<=' in simplified:
        head, body = simplified.split('<=', 1)
        return f"{head.strip()} <= {body.strip()}"
    return simplified

def filter_rules_by_length(rules_dict: Dict[str, List[Tuple[str, Dict, str]]], 
                           max_length: int) -> Dict[str, List[Tuple[str, Dict, str]]]:
    """
    Filter rules based on rule length
    
    Args:
        rules_dict: original rule dictionary
        max_length: Maximum rule length (1, 2, or 3) 
    
    Returns:
        Filtered rule dictionary
    """
    filtered_rules = {}
    
    for normalized_rule, rule_list in rules_dict.items():
        if not rule_list:
            continue
        
        original_rule = rule_list[0][0]
        length_type = get_rule_length_type(original_rule)
        
        # Extract number length
        if length_type.startswith('L'):
            try:
                length = int(length_type[1:])
                if length <= max_length:
                    filtered_rules[normalized_rule] = rule_list
            except ValueError:
                # Unable to parse length, skipping
                continue
    
    return filtered_rules

def filter_rules_by_type(rules_dict: Dict[str, List[Tuple[str, Dict, str]]], 
                         only_binary: bool = False, 
                         only_unary_c: bool = False, 
                         only_unary_d: bool = False) -> Dict[str, List[Tuple[str, Dict, str]]]:
    """
    Filter rules based on rule type
    
    Args:
        rules_dict: original rule dictionary
        only_binary: Only keepbinaryRules (the rule header contains(X,Y)) 
        only_unary_c: Only keepunaryrules, andbodycannot appear inrp(A,X)andrp(X,A)
        only_unary_d: Only keepunaryrules, andbodyin can only berp(A,X)andrp(X,A)
    
    Returns:
        Filtered rule dictionary
    """
    if not only_binary and not only_unary_c and not only_unary_d:
        return rules_dict
    
    filtered_rules = {}
    
    for normalized_rule, rule_list in rules_dict.items():
        # Use the first rule for judgment (all rules that are the same after normalization should have the same type)
        if not rule_list:
            continue
        
        original_rule = rule_list[0][0]
        
        try:
            # Determine whether it isbinaryrules
            is_binary = is_binary_rule(original_rule)
            
            # if onlybinaryrules
            if only_binary:
                if is_binary:
                    filtered_rules[normalized_rule] = rule_list
                continue
            
            # if onlyunaryrules
            if only_unary_c or only_unary_d:
                if is_binary:
                    continue
                
                # parsebodyAtomic types in
                if '<=' not in original_rule:
                    continue
                
                body = original_rule.split('<=', 1)[1].strip()
                body_atoms = parse_body_atoms(body)
                
                # CheckbodyAtomic types in
                has_rp_xa = False  # Does it containrp(X,A)orrp(A,X)
                has_other = False  # Does it contain other types
                
                for atom in body_atoms:
                    try:
                        atom_type = get_body_atom_type(atom)
                        if atom_type in ['rp(X,A)', 'rp(A,X)']:
                            has_rp_xa = True
                        else:
                            has_other = True
                    except ValueError:
                        # Unresolvable atom, skipped
                        continue
                
                # only_unary_c: bodycannot appear inrp(A,X)andrp(X,A)
                if only_unary_c:
                    if not has_rp_xa:
                        filtered_rules[normalized_rule] = rule_list
                
                # only_unary_d: bodyin can only berp(A,X)andrp(X,A)
                if only_unary_d:
                    if has_rp_xa and not has_other:
                        filtered_rules[normalized_rule] = rule_list
        
        except Exception as e:
            # If parsing error occurs, skip the rule
            print(f"Warning: Failed to parse rule type for: {original_rule}, error: {e}")
            continue
    
    return filtered_rules

def convert_to_simplified_format(rule: str) -> str:
    """
    Convert rules in bracketed format to abbreviated format
    Useanalysis_rule.pyinRuleParser._normalize_to_simplifiedmethod
    
    For example:
    /award/award_category/winners./award/award_honor/ceremony(X,Y) <=
    /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,A)
    
    Convert to:
    /award/award_category/winners./award/award_honor/ceremony <=
    /award/award_category/winners./award/award_honor/award_winner * INVERSE_/award/award_ceremony/awards_presented./award/award_honor/award_winner
    """
    if '<=' not in rule:
        return rule
    
    head_part, body_part = rule.split('<=', 1)
    head_part = head_part.strip()
    body_part = body_part.strip()
    
    # Check if it is already in abbreviated format
    # Abbreviation format features: There are no parentheses in the header of a binary rule, or there is only one parameter in the header of a unary rule.
    if '(' not in head_part or ')' not in head_part:
        # It is already in abbreviated format, but spaces need to be added to beautify it.
        return beautify_simplified_rule(rule)
    
    paren_content = head_part.split('(')[1].split(')')[0]
    if ',' not in paren_content:
        # Uniary abbreviation format, but spaces need to be added to beautify it
        return beautify_simplified_rule(rule)
    
    # UseRuleParserConvert
    try:
        normalized = RuleParser._normalize_to_simplified(head_part, body_part)
        # Beautify output: in*Add spaces around connectors
        normalized = beautify_simplified_rule(normalized)
        return normalized
    except Exception as e:
        # If the conversion fails, return the original rule
        print(f"Warning: Failed to convert rule to simplified format: {rule}, error: {e}")
        return rule

def beautify_simplified_rule(rule: str) -> str:
    return rule


def is_binary_rule(rule: str) -> bool:
    """
    Determine whether it isBinaryRules (the rule header contains(X,Y)) 
    """
    if '<=' not in rule:
        return False
    head = rule.split('<=', 1)[0].strip()
    return '(X,Y)' in head

def get_head_variable_type(rule: str) -> str:
    """
    Get the variable type of the rule header
    Return: 'r(X,c)', 'r(c,X)', 'r(X,X)', 'binary'
    """
    if '<=' not in rule:
        raise ValueError(f"Rule format is wrong, missing'<=': {rule}")
    
    head = rule.split('<=', 1)[0].strip()
    
    # Extract content in parentheses
    match = re.search(r'\(([^)]+)\)', head)
    if not match:
        raise ValueError(f"The rule header format is incorrect and the bracket content cannot be extracted.: {head}")
    
    args = match.group(1).split(',')
    if len(args) != 2:
        raise ValueError(f"Wrong number of rule header parameters, expected2parameters: {args}")
    
    arg1, arg2 = args[0].strip(), args[1].strip()
    
    # Define variable type:Xclass variables andAclass variable
    x_vars = {'X', 'Y', 'Z'}  # Xclass variable
    a_vars = {'A', 'B', 'C'}  # Aclass variable
    
    # Determine variable type
    is_x_var1 = arg1 in x_vars
    is_x_var2 = arg2 in x_vars
    is_a_var1 = arg1 in a_vars
    is_a_var2 = arg2 in a_vars
    is_var1 = is_x_var1 or is_a_var1
    is_var2 = is_x_var2 or is_a_var2
    
    if arg1 == 'X' and arg2 == 'Y':
        return 'binary'
    elif is_x_var1 and is_x_var2 and arg1 == arg2:
        return 'r(X,X)'  # two identicalXclass variable
    elif is_x_var1 and not is_var2:
        return 'r(X,c)'  # Xclass variable,constant
    elif not is_var1 and is_x_var2:
        return 'r(c,X)'  # constant,Xclass variable
    else:
        raise ValueError(f"Unsupported rule header type: ({arg1},{arg2}) in {rule}")

def parse_body_atoms(body: str) -> List[str]:
    """
    Parse correctlybodyAtoms in , considering that the atoms may contain commas inside
    For example: "/award/award_category/winners./award/award_honor/ceremony(/m/0gs9p,X), other_relation(Y,Z)"
    """
    atoms = []
    current_atom = ""
    paren_count = 0
    i = 0
    while i < len(body):
        char = body[i]
        
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == ',' and paren_count == 0:
            # Only commas outside parentheses are atom delimiters
            if current_atom.strip():
                atoms.append(current_atom.strip())
            current_atom = ""
            i += 1
            continue
        
        current_atom += char
        i += 1
    
    # add last atom
    if current_atom.strip():
        atoms.append(current_atom.strip())
    
    return atoms

def get_body_atom_type(atom: str) -> str:
    """
    getbodyatomic variable type
    Return: 'rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)'
    """
    # Extract content in parentheses
    match = re.search(r'\(([^)]+)\)', atom.strip())
    if not match:
        print("Wrong format:", atom)
        raise ValueError(f"BodyAtom format error, unable to extract bracket contents: {atom}")
    
    args = match.group(1).split(',')
    if len(args) != 2:
        raise ValueError(f"BodyWrong number of atomic arguments, expected2parameters: {args} in {atom}")
    
    arg1, arg2 = args[0].strip(), args[1].strip()
    
    # Define variable type:Xclass variables andAclass variable
    x_vars = {'X', 'Y', 'Z'}  # Xclass variable
    a_vars = {'A', 'B', 'C'}  # Aclass variable
    
    # Determine variable type
    is_x_var1 = arg1 in x_vars
    is_x_var2 = arg2 in x_vars
    is_a_var1 = arg1 in a_vars
    is_a_var2 = arg2 in a_vars
    is_var1 = is_x_var1 or is_a_var1
    is_var2 = is_x_var2 or is_a_var2
    
    if is_x_var1 and is_x_var2 and arg1 == arg2:
        return 'rp(X,X)'  # two identicalXclass variable
    elif is_x_var1 and is_a_var2:
        return 'rp(X,A)'  # Xclass variables andAclass variable
    elif is_a_var1 and is_x_var2:
        return 'rp(A,X)'  # Aclass variables andXclass variable
    elif is_x_var1 and not is_var2:
        return 'rp(X,c)'  # XClass variables and constants
    elif not is_var1 and is_x_var2:
        return 'rp(c,X)'  # constant sumXclass variable
    else:
        raise ValueError(f"Not supportedBodyAtomic type: ({arg1},{arg2}) in {atom}")

def get_relation_path_length(atom: str) -> int:
    """
    Get the length of the relationship path in the atom (the number of relationships)
    pass count"*"number of+1to confirm
    
    NOTE: This function is now used for atoms in the abbreviated form
    The abbreviated form of an atom is a relationship path, which may contain multiple relationships.*connect
    """
    # For the abbreviated format,atomMay not contain parentheses, it is a pure relational path
    # or is relation_path(constant) Format
    
    # Remove brackets (if any)
    if '(' in atom:
        relation_path = atom.split('(')[0].strip()
    else:
        relation_path = atom.strip()
    
    # Calculate"*"The number of (ignoring spaces)
    # Remove all spaces before counting
    relation_path_no_space = relation_path.replace(' ', '')
    dot_count = relation_path_no_space.count('*')
    # Relationship path length = "*"number of + 1
    return dot_count + 1

def get_rule_length_type(rule: str) -> str:
    """
    Get the length type of the rule
    Return: 'L1', 'L2', 'L3', or 'other'
    
    Note: First convert the rule into abbreviated format, and then determine the length type based on the abbreviated format
    This can correctly distinguishL1/L2/L3, avoidL3Misjudged asL1
    """
    if '<=' not in rule:
        return 'other'
    
    # First convert to abbreviated format
    simplified_rule = convert_to_simplified_format(rule)
    
    # Extract from abbreviated formatbody
    body = simplified_rule.split('<=', 1)[1].strip()
    
    # For binary rules in abbreviated form,bodyis a relational path (which may contain*Connect multiple relationships)
    # For unary rules in abbreviated form,bodyYes relation_path(constant) Format
    # There may be multiple such atoms (although usually unary rules have only onebodyatoms)
    
    # parsebodyatom
    # Binary rules for shorthand format: bodyIt is just a relationship path, without commas separating it.
    # Unary rules for abbreviated formats: bodymay be rel(c) or rel1*rel2(c)
    
    # Check whether it is a binary rule (in abbreviated form, binary rulebodyno parentheses, orheadno brackets)
    head = simplified_rule.split('<=', 1)[0].strip()
    is_binary = '(' not in head or '(X,Y)' in head
    
    if is_binary and ',' not in body:
        # Binary rule abbreviation format:bodyis a single relationship path
        length = get_relation_path_length(body)
        return f'L{length}'
    else:
        # unary rule or with commabody (The conversion may have failed)
        # Try to parse according to the abbreviated format
        atoms = parse_body_atoms(body)
        
        if len(atoms) == 1:
            # Single atom, directly calculate the length
            length = get_relation_path_length(atoms[0])
            return f'L{length}'
        else:
            # Multiple atoms, take the maximum length
            lengths = []
            for atom in atoms:
                length = get_relation_path_length(atom)
                if length > 0:
                    lengths.append(length)
            
            if not lengths:
                # print('No valid atoms in rule body:', rule)
                return 'L0'
                
            # Determine rule length type
            max_length = max(lengths)
            assert max_length <= 3, f"Unexpected relation path length: {max_length} in rule: {rule}"
            return f'L{max_length}'

def analyze_rule_statistics(rules_dict: Dict[str, List]) -> Dict:
    """
    Detailed statistics for analysis rules
    """
    stats = {
        'total_rules': len(rules_dict),
        'binary_rules': 0,
        'unary_rules': 0,
        'L0_unary': 0,
        'L1_unary': 0,
        'L2_unary': 0,
        'L0_binary': 0,
        'L1_binary': 0,
        'L2_binary': 0,
        'L3_binary': 0,
        'atom_relation_lengths': defaultdict(int),
    }
    
    # UnaryRule matrix:head_type × body_type
    head_types = ['r(X,c)', 'r(c,X)', 'r(X,X)']
    body_types = ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)']
    
    # Initialization matrix
    stats['unary_matrix'] = {}
    for head_type in head_types:
        stats['unary_matrix'][head_type] = {}
        for body_type in body_types:
            stats['unary_matrix'][head_type][body_type] = 0
    
    for normalized_rule, rule_list in rules_dict.items():
        rule = rule_list[0][0]
        length_type = get_rule_length_type(rule)
        # judgeBinaryStillUnary
        if is_binary_rule(rule):
            stats['binary_rules'] += 1
            if length_type in ['L0', 'L1', 'L2', 'L3']:
                stats[f'{length_type}_binary'] += 1
        else:
            stats['unary_rules'] += 1
            if length_type in ['L0', 'L1', 'L2']:
                stats[f'{length_type}_unary'] += 1
            # analysisUnaryRule header andbodyType
            try:
                head_type = get_head_variable_type(rule)
                if head_type in head_types:
                    body = rule.split('<=', 1)[1].strip()
                    atoms = parse_body_atoms(body)
                    for atom in atoms:
                        try:
                            body_type = get_body_atom_type(atom)
                            stats['unary_matrix'][head_type][body_type] += 1
                        except ValueError as e:
                            print(f"Warning: Unable to resolve atom '{atom}' in rule '{rule}': {e}")
                            continue
                else:
                    print(f"Warning: UnaryRule's header type is not within the expected range: {head_type} in {rule}")
            except ValueError as e:
                print(f"Warning: Unable to parse rule header '{rule}': {e}")
                continue
        # analysisbodyRelational path length of atoms in
        if '<=' in rule:
            body = rule.split('<=', 1)[1].strip()
            atoms = parse_body_atoms(body)
            for atom in atoms:
                length = get_relation_path_length(atom)
                if length > 0:
                    stats['atom_relation_lengths'][length] += 1
        # print(stats)
    return stats

def write_rule_section(writer, rules_set: Set, rules_dict: Dict, section_title: str, kg=None):
    """
    Write rule example section (helper function)
    
    Args:
        writer: CSV writerobject
        rules_set: rule set
        rules_dict: Rule Dictionary
        section_title: section title
        kg: Knowledge graph (optional)
    """
    if not rules_set:
        return
    
    # separationBinaryandUnaryrules and categorized by lengthBinaryrules
    binary_rules = [rule for rule in rules_set if is_binary_rule(rule)]
    unary_rules = [rule for rule in rules_set if not is_binary_rule(rule)]
    
    # willBinaryRules sorted by length
    l1_binary = [r for r in binary_rules if get_rule_length_type(r) == 'L1']
    l2_binary = [r for r in binary_rules if get_rule_length_type(r) == 'L2']
    l3_binary = [r for r in binary_rules if get_rule_length_type(r) == 'L3']
    other_binary = [r for r in binary_rules if get_rule_length_type(r) not in ['L1', 'L2', 'L3']]
    
    # ChooseBinaryRule: at least3aL1, 3aL2, The rest is supplemented
    selected_binary = []
    selected_binary.extend(l1_binary[:3])  # at least3aL1
    selected_binary.extend(l2_binary[:3])  # at least3aL2
    
    # added to10aBinaryrules
    remaining_needed = 10 - len(selected_binary)
    if remaining_needed > 0:
        # Supplement from remaining rules
        remaining_rules = l1_binary[3:] + l2_binary[3:] + l3_binary + other_binary
        selected_binary.extend(remaining_rules[:remaining_needed])
    
    # Choose10aUnaryrules
    selected_unary = unary_rules[:10]
    
    # Combine all selected rules
    selected_rules = selected_binary + selected_unary
    
    # write title
    writer.writerow([f'{section_title} (total{len(rules_set)}strip, before display{len(selected_rules)}Article: before10ArticleBinary(at least3aL1+3aL2), after10ArticleUnary)'])
    
    # Write header and data
    if kg is not None:
        writer.writerow(['Post-conversion rules', 'indicator', 'real results'])
        for rule in selected_rules:
            simplified_rule = convert_to_simplified_format(rule)
            metrics = rules_dict[rule][0][1] if rules_dict[rule] else {}
            real_result = analyze_rule_from_string(rule, kg) if kg else None
            real_result_str = str(real_result['join_result']) if real_result else 'N/A'
            writer.writerow([simplified_rule, str(metrics), real_result_str])
    else:
        writer.writerow(['Post-conversion rules', 'indicator'])
        for rule in selected_rules:
            simplified_rule = convert_to_simplified_format(rule)
            metrics = rules_dict[rule][0][1] if rules_dict[rule] else {}
            writer.writerow([simplified_rule, str(metrics)])
    writer.writerow([])

def save_statistics_to_csv(stats1: Dict, stats2: Dict, file1_name: str, file2_name: str, 
                          set1: Set, set2: Set, rules1: Dict, rules2: Dict, output_file: str, kg=None, list_only: int = 0):
    """
    Save statistical results toCSVFile
    """
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        
        # if yeslist_onlymode, skip all statistical information and directly output the unique rule list
        if list_only != 0:
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1
            
            if list_only == 1:
                # List onlyfile1unique rules
                writer.writerow([f'only in{file1_name}rules in (total{len(only_in_1)}Article)'])
                writer.writerow([])
                if kg:
                    writer.writerow(['Post-conversion rules', f'{file1_name}indicator', 'real indicator', 'confDifference(File-true)', 'real conf', 'length'])
                    for rule in only_in_1:
                        simplified_rule = convert_to_simplified_format(rule)
                        metrics1 = rules1[rule][0][1] if rules1[rule] else {}
                        real_result = analyze_rule_from_string(rule, kg) if kg else None
                        real_result_str = str(real_result['join_result']) if real_result else 'N/A'
                        conf_diff = 'N/A'
                        real_conf = 'N/A'
                        if real_result and 'join_result' in real_result and 'confidence' in metrics1:
                            try:
                                real_conf_val = real_result['join_result'].get('confidence', 0)
                                file_conf = metrics1['confidence']
                                conf_diff = f"{file_conf - real_conf_val:.4f}"
                                real_conf = f"{real_conf_val:.4f}"
                            except:
                                pass
                        length = get_rule_length_type(rule)
                        writer.writerow([simplified_rule, str(metrics1), real_result_str, conf_diff, real_conf, length])
                else:
                    writer.writerow(['Post-conversion rules', f'{file1_name}indicator', 'length'])
                    for rule in only_in_1:
                        simplified_rule = convert_to_simplified_format(rule)
                        metrics1 = rules1[rule][0][1] if rules1[rule] else {}
                        length = get_rule_length_type(rule)
                        writer.writerow([simplified_rule, str(metrics1), length])
            elif list_only == 2:
                # List onlyfile2unique rules
                writer.writerow([f'only in{file2_name}rules in (total{len(only_in_2)}Article)'])
                writer.writerow([])
                if kg:
                    writer.writerow(['Post-conversion rules', f'{file2_name}indicator', 'real indicator', 'confDifference(File-true)', 'real conf', 'length'])
                    for rule in only_in_2:
                        simplified_rule = convert_to_simplified_format(rule)
                        metrics2 = rules2[rule][0][1] if rules2[rule] else {}
                        real_result = analyze_rule_from_string(rule, kg) if kg else None
                        real_result_str = str(real_result['join_result']) if real_result else 'N/A'
                        conf_diff = 'N/A'
                        real_conf = 'N/A'
                        if real_result and 'join_result' in real_result and 'confidence' in metrics2:
                            try:
                                real_conf_val = real_result['join_result'].get('confidence', 0)
                                file_conf = metrics2['confidence']
                                conf_diff = f"{file_conf - real_conf_val:.4f}"
                                real_conf = f"{real_conf_val:.4f}"
                            except:
                                pass
                        length = get_rule_length_type(rule)
                        writer.writerow([simplified_rule, str(metrics2), real_result_str, conf_diff, real_conf, length])
                else:
                    writer.writerow(['Post-conversion rules', f'{file2_name}indicator', 'length'])
                    for rule in only_in_2:
                        simplified_rule = convert_to_simplified_format(rule)
                        metrics2 = rules2[rule][0][1] if rules2[rule] else {}
                        length = get_rule_length_type(rule)
                        writer.writerow([simplified_rule, str(metrics2), length])
            
            print(f"\nThe statistical results have been saved to: {output_file}")
            return
        
        # ========== Default mode: Output complete statistics ==========
        # ========== basic statistics ==========
        writer.writerow(['basic statistics'])
        writer.writerow(['Statistical items', file1_name, file2_name, 'difference'])
        writer.writerow(['Total number of rules', stats1['total_rules'], stats2['total_rules'], 
                        stats2['total_rules'] - stats1['total_rules']])
        writer.writerow([])
        
        # ========== Rule type distribution ==========
        writer.writerow(['Rule type distribution'])
        writer.writerow(['Type', file1_name, file2_name, 'difference'])
        
        # Define rule type
        rule_types = [
            ('L0 Unary', 'L0_unary'),
            ('L1 Unary', 'L1_unary'), 
            ('L2 Unary', 'L2_unary'),
            ('Unary', 'unary_rules'),
            ('L0 Binary', 'L0_binary'),
            ('L1 Binary', 'L1_binary'),
            ('L2 Binary', 'L2_binary'),
            ('L3 Binary', 'L3_binary'),
            ('Binary', 'binary_rules')
        ]
        
        # Traverse output rule type statistics
        for display_name, stat_key in rule_types:
            count1 = stats1[stat_key]
            count2 = stats2[stat_key]
            pct1 = count1 / stats1['total_rules'] * 100
            pct2 = count2 / stats2['total_rules'] * 100
            writer.writerow([display_name,
                            f"{count1} ({pct1:.2f}%)",
                            f"{count2} ({pct2:.2f}%)",
                            count2 - count1])
        writer.writerow([])
        
        # ========== Unaryregular matrix distribution ==========
        if stats1['unary_rules'] > 0 or stats2['unary_rules'] > 0:
            head_types = ['r(X,c)', 'r(c,X)', 'r(X,X)', 'sum']
            body_types = ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)', 'sum']
            
            # outputfile1matrix
            writer.writerow([f'{file1_name} Unaryrule matrix'])
            # Header
            header = ['head\\body'] + body_types
            writer.writerow(header)
            
            # Calculate the sum of each row and column
            matrix1 = stats1.get('unary_matrix', {})
            for head_type in ['r(X,c)', 'r(c,X)', 'r(X,X)']:
                row = [head_type]
                row_sum = 0
                for body_type in ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)']:
                    count = matrix1.get(head_type, {}).get(body_type, 0)
                    row.append(count)
                    row_sum += count
                row.append(row_sum)  # row sum
                writer.writerow(row)
            
            # Calculate column sum
            sum_row = ['sum']
            total_sum = 0
            for body_type in ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)']:
                col_sum = sum(matrix1.get(head_type, {}).get(body_type, 0) 
                             for head_type in ['r(X,c)', 'r(c,X)', 'r(X,X)'])
                sum_row.append(col_sum)
                total_sum += col_sum
            sum_row.append(total_sum)  # total sum
            writer.writerow(sum_row)
            writer.writerow([])
            
            # outputfile2matrix
            writer.writerow([f'{file2_name} Unaryrule matrix'])
            # Header
            writer.writerow(header)
            
            # Calculate the sum of each row and column
            matrix2 = stats2.get('unary_matrix', {})
            for head_type in ['r(X,c)', 'r(c,X)', 'r(X,X)']:
                row = [head_type]
                row_sum = 0
                for body_type in ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)']:
                    count = matrix2.get(head_type, {}).get(body_type, 0)
                    row.append(count)
                    row_sum += count
                row.append(row_sum)  # row sum
                writer.writerow(row)
            
            # Calculate column sum
            sum_row = ['sum']
            total_sum = 0
            for body_type in ['rp(X,c)', 'rp(c,X)', 'rp(X,A)', 'rp(A,X)', 'rp(X,X)']:
                col_sum = sum(matrix2.get(head_type, {}).get(body_type, 0) 
                             for head_type in ['r(X,c)', 'r(c,X)', 'r(X,X)'])
                sum_row.append(col_sum)
                total_sum += col_sum
            sum_row.append(total_sum)  # total sum
            writer.writerow(sum_row)
            writer.writerow([])

        # ========== Atompath length distribution ==========
        writer.writerow(['Atompath length distribution'])
        writer.writerow(['length', file1_name, file2_name, 'difference'])
        
        total_atoms1 = sum(stats1['atom_relation_lengths'].values())
        total_atoms2 = sum(stats2['atom_relation_lengths'].values())
        
        # Get all occurrences of length
        all_lengths = sorted(set(stats1['atom_relation_lengths'].keys()) | set(stats2['atom_relation_lengths'].keys()))
        
        for length in all_lengths:
            count1 = stats1['atom_relation_lengths'].get(length, 0)
            count2 = stats2['atom_relation_lengths'].get(length, 0)
            pct1 = f"{count1/total_atoms1*100:.2f}%" if total_atoms1 > 0 else "0.00%"
            pct2 = f"{count2/total_atoms2*100:.2f}%" if total_atoms2 > 0 else "0.00%"
            writer.writerow([f'length{length} (L{length})', 
                            f"{count1} ({pct1})",
                            f"{count2} ({pct2})",
                            count2 - count1])
        
        writer.writerow(['total number of atoms', total_atoms1, total_atoms2, total_atoms2 - total_atoms1])
        writer.writerow([])
        
        # ========== Rule coverage ==========
        common_rules = set1 & set2
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        coverage = len(common_rules) / len(set1) if len(set1) > 0 else 0
        
        writer.writerow(['Rule coverage'])
        writer.writerow(['Statistical items', 'numerical value'])
        writer.writerow(['number of common rules', len(common_rules)])
        writer.writerow([f'only in{file1_name}number of rules in', len(only_in_1)])
        writer.writerow([f'only in{file2_name}number of rules in', len(only_in_2)])
        writer.writerow([f'{file2_name}Yes{file1_name}coverage', f'{coverage*100:.2f}%'])
        writer.writerow([])
        
        # ========== Examples of common rules ==========
        topN = 20
        if common_rules:
            # separationBinaryandUnaryrules
            binary_rules = [rule for rule in common_rules if is_binary_rule(rule)]
            unary_rules = [rule for rule in common_rules if not is_binary_rule(rule)]
            
            # Calculate how many items of each type should be taken
            half_n = topN // 2
            selected_rules = binary_rules[:half_n] + unary_rules[:half_n]
            
            writer.writerow([f'Examples of common rules (total{len(common_rules)}strip, before display{topN}Article: before{half_n}ArticleBinary, after{half_n}ArticleUnary)'])
            if kg is not None:
                writer.writerow(['Post-conversion rules', f'{file1_name}indicator', f'{file2_name}indicator', 'real results'])
                for rule in selected_rules:
                    simplified_rule = convert_to_simplified_format(rule)
                    metrics1 = rules1[rule][0][1] if rules1[rule] else {}
                    metrics2 = rules2[rule][0][1] if rules2[rule] else {}
                    real_result = analyze_rule_from_string(rule, kg) if kg else None
                    real_result_str = str(real_result['join_result']) if real_result else 'N/A'
                    writer.writerow([simplified_rule, str(metrics1), str(metrics2), real_result_str])
            else:
                writer.writerow(['Post-conversion rules', f'{file1_name}indicator', f'{file2_name}indicator'])
                for rule in selected_rules:
                    simplified_rule = convert_to_simplified_format(rule)
                    metrics1 = rules1[rule][0][1] if rules1[rule] else {}
                    metrics2 = rules2[rule][0][1] if rules2[rule] else {}
                    writer.writerow([simplified_rule, str(metrics1), str(metrics2)])
            writer.writerow([])
        
        # ========== Rules only in a file ==========
        # Use helper functions to handle both parts
        write_rule_section(writer, only_in_1, rules1, f'only in{file1_name}rules in', kg)
        write_rule_section(writer, only_in_2, rules2, f'only in{file2_name}rules in', kg)

    print(f"\nThe statistical results have been saved to: {output_file}")

def main(args):
    """
    main function
    
    Args:
        args: Command line parameter object, containing the following properties:
            - file1: First rule file name
            - file2: Second rule file name
            - dataset: Data set name
            - target_relation: target relationship
            - only_b: Compare onlybinaryrules
            - only_u_c: Compare onlyunaryrules (excludingrp(A,X)andrp(X,A)) 
            - only_u_d: Compare onlyunaryRules (contains onlyrp(A,X)andrp(X,A)) 
            - list_only: enumeration mode
            - max_length: Maximum rule length
    """
    # file path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Processing path: If it is a complete path, use it directly; otherwise, it is considered relative toout/{dataset}/file name
    if os.path.isabs(args.file1) or '\\' in args.file1 or '/' in args.file1:
        file1_path = args.file1
        file1_name = os.path.basename(args.file1)
    else:
        file1_name = args.file1
        file1_path = os.path.join(base_dir, "out", args.dataset, args.file1)
    
    if os.path.isabs(args.file2) or '\\' in args.file2 or '/' in args.file2:
        file2_path = args.file2
        file2_name = os.path.basename(args.file2)
    else:
        file2_name = args.file2
        file2_path = os.path.join(base_dir, "out", args.dataset, args.file2)
    
    dataset_path = os.path.join(base_dir, "data", args.dataset, "train.txt")
    
    print("=== Rule file comparison tool ===")
    print(f"Compare files:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    if args.target_relation:
        print(f"target relationship: {args.target_relation}")
    else:
        print("analysis mode: Full rule analysis")
    
    # Check if the file exists
    if not os.path.exists(file1_path):
        print(f"Error: File {file1_path} does not exist")
        return
    
    if not os.path.exists(file2_path):
        print(f"Error: File {file2_path} does not exist")
        return
    
    # Load rules
    print(f"\n=== Load rules ===")
    rules1 = load_rules_with_target_relation(file1_path, args)
    rules2 = load_rules_with_target_relation(file2_path, args)
    print(f"Loaded {file1_name}: {len(rules1)} rules")
    print(f"Loaded {file2_name}: {len(rules2)} rules")
    
    # Filtering rules: filter by length first, then filter by type
    if args.max_length < 3:
        print(f"\n=== length filter ===")
        print(f"filter mode: Only keep the length <= {args.max_length} rules")
        rules1 = filter_rules_by_length(rules1, args.max_length)
        rules2 = filter_rules_by_length(rules2, args.max_length)
        print(f"After length filtering {file1_name}: {len(rules1)} rules")
        print(f"After length filtering {file2_name}: {len(rules2)} rules")
    
    if args.only_b or args.only_u_c or args.only_u_d:
        print(f"\n=== Type filtering ===")
        if args.only_b:
            print("filter mode: Only keepbinaryrules")
        elif args.only_u_c:
            print("filter mode: Only keepunaryrules(Not includedrp(A,X)andrp(X,A))")
        elif args.only_u_d:
            print("filter mode: Only keepunaryrules(Contains onlyrp(A,X)andrp(X,A))")
        
        rules1 = filter_rules_by_type(rules1, args.only_b, args.only_u_c, args.only_u_d)
        rules2 = filter_rules_by_type(rules2, args.only_b, args.only_u_c, args.only_u_d)
        print(f"After type filtering {file1_name}: {len(rules1)} rules")
        print(f"After type filtering {file2_name}: {len(rules2)} rules")
    
    # Load knowledge graph
    print(f"\n=== Load knowledge graph ===")
    kg = None
    if os.path.exists(dataset_path):
        try:
            kg = load_dataset(dataset_path)
            print(f"Knowledge graph loaded successfully")
        except Exception as e:
            print(f"Failed to load knowledge graph: {e}")
            print(f"Will not contain actual results information")
    else:
        print(f"Dataset file does not exist: {dataset_path}")
        print(f"Will not contain actual results information")
    
    # Compare rules and generate statistics
    stats1 = analyze_rule_statistics(rules1)
    stats2 = analyze_rule_statistics(rules2)
    
    set1 = set(rules1.keys())
    set2 = set(rules2.keys())
    
    # Export toCSV
    output_suffix = "all_rule" if args.target_relation is None else "rule_" + args.target_relation.split('/')[-1]
    csv_output_path = os.path.join(base_dir, "out", args.dataset, f"{output_suffix}_comparison.csv")
    save_statistics_to_csv(stats1, stats2, file1_name, file2_name, set1, set2, rules1, rules2, csv_output_path, kg, args.list_only)

if __name__ == "__main__":
    # Command line parameter parsing
    parser = argparse.ArgumentParser(
        description='Rule file comparison tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
python src/rule_tools/compare_rules.py --dataset FB15k-237 --file1 rule.txt --file2 rule_comparison.txt
python src/rule_tools/compare_rules.py --file1 rule.txt --file2 rule_comparison.txt
python src/rule_tools/compare_rules.py

# List onlyfile1uniqueL1andL2rules
python src/rule_tools/compare_rules.py --list_only 1 --max_length 2

# List onlyfile2uniqueBinaryrules
python src/rule_tools/compare_rules.py --list_only 2 --only_b

# compareL1Differences in rules
python src/rule_tools/compare_rules.py --max_length 1
        ''')
    
    parser.add_argument('--dataset', type=str, default='FB15k-237',
                        help='Data set name (Default: FB15k-237)')
    parser.add_argument('--file1', type=str, default='rules-100-3',
                        help='First rule file name, relative toout/{dataset}/file name (Default: rules-100-10)')
    parser.add_argument('--file2', type=str, default='rule.txt',
                        help='The second rule file name, relative toout/{dataset}/file name (Default: rule.txt)')
    parser.add_argument('--target-relation', type=str, default=None,
                        help='Target relationship, if not specified all rules will be analyzed')
    parser.add_argument('--only_b', action='store_true',
                        help='Compare onlybinaryrules')
    parser.add_argument('--only_u_c', action='store_true',
                        help='Compare onlyunaryrules, andbodycannot appear inrp(A,X)andrp(X,A)')
    parser.add_argument('--only_u_d', action='store_true',
                        help='Compare onlyunaryrules, andbodyin can only berp(A,X)andrp(X,A)')
    parser.add_argument('--only_normal', action='store_true',
                        help='Compare onlynormalrules')
    parser.add_argument('--list_only', type=int, default=0, choices=[0, 1, 2],
                        help='enumeration mode: 0=Default(Show differences), 1=List onlyfile1Unique rules, 2=List onlyfile2Unique rules (Default: 0)')
    parser.add_argument('--max_length', type=int, default=3, choices=[1, 2, 3],
                        help='Maximum rule length, only the length is retained<=rules for this value (Default: 3)')
    
    args = parser.parse_args()
    
    # Check mutually exclusive options
    filter_options = sum([args.only_b, args.only_u_c, args.only_u_d])
    if filter_options > 1:
        print("Error: --only_b, --only_u_c, --only_u_d The options are mutually exclusive, only one of them can be selected")
        sys.exit(1)
    
    # call main function
    main(args)
