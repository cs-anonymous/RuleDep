#!/usr/bin/env python3
"""
Rule support calculator (enhanced version)
Computes a given rule headSize, bodySize and support

Supported rule formats:
1. Abbreviated format:
   /award/award_category/winners./award/award_honor/ceremony <= 
   /award/award_category/winners./award/award_honor/ceremony*/award/award_ceremony/awards_presented./award/award_honor/award_winner*INVERSE_/award/award_ceremony/awards_presented./award/award_honor/award_winner

2. Bracketed format:
   /award/award_category/winners./award/award_honor/ceremony(X,Y) <= 
   /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,A)

3. Supports single variable rules (univariate) and double variable rules (binary)

Algorithm features:
1. Based onr2h2tIndex structure to provide efficient relational query
2. automatically createdinverserelationship index
3. Compute compound relationship paths using the step-by-step join algorithm
4. Avoid unnecessary Cartesian product calculations through connection node optimization
5. Store intermediate results inr2h2tIn the index, reuse is supported
6. Provides two calculation methods: efficient connection-based algorithm and brute force search algorithm.
"""

import os
import json
import re
from collections import defaultdict, Counter
import sys
from typing import Set, Tuple, Dict, List, Optional
from itertools import product

# DEBUGcontrol switch
DEBUG = __name__ == "__main__"

def debug(*args, **kwargs):
    """only inDEBUGPrint debugging information in mode"""
    if DEBUG:
        print(*args, **kwargs)

class KnowledgeGraph:
    """Knowledge graph class, used to store and query triples, based onr2h2tIndex"""
    
    def __init__(self):
        # r2h2tIndex: relation -> {head_id: set of tail_ids}
        self.r2h2t = defaultdict(lambda: defaultdict(set))
        # all triples (Use entitiesID)
        self.triples = set()
        # Entity encoding: entity_str -> entity_id
        self.entity2id = {}
        # Entity decoding: entity_id -> entity_str
        self.id2entity = {}
        # next available entityID
        self._next_entity_id = 0
        # All relationships (including original relationships andinverserelationship)
        self.relations = set()
        # Original relation collection (used to distinguish between base relations and cached composite relations)
        self.base_relations = set()
        # Entity limit check
        self.MAX_ENTITIES = 2**16
    
    def _get_or_create_entity_id(self, entity: str) -> int:
        """Get or create an entityIDencoding"""
        if entity not in self.entity2id:
            if self._next_entity_id >= self.MAX_ENTITIES:
                raise ValueError(f"The number of entities exceeds the upper limit {self.MAX_ENTITIES}! current entity: {entity}")
            self.entity2id[entity] = self._next_entity_id
            self.id2entity[self._next_entity_id] = entity
            self._next_entity_id += 1
        return self.entity2id[entity]
    
    @property
    def entities(self) -> Set[int]:
        """Return all entitiesIDcollection of"""
        return set(range(self._next_entity_id))
    
    def get_entity_str(self, entity_id: int) -> str:
        """Get entityIDcorresponding string"""
        return self.id2entity.get(entity_id, f"<UNKNOWN_ID_{entity_id}>")
    
    def get_entity_id(self, entity_str: str) -> Optional[int]:
        """Get the entity string corresponding toID"""
        return self.entity2id.get(entity_str)
    
    @staticmethod
    def encode_pair(head: int, tail: int) -> int:
        """will(head, tail)Encode the pair as a single integer:head << 16 | tail"""
        return (head << 16) | tail
    
    @staticmethod
    def decode_pair(encoded: int) -> Tuple[int, int]:
        """Decode encoded integer to(head, tail)Yes"""
        head = encoded >> 16
        tail = encoded & 0xFFFF
        return head, tail
    
    def add_triple(self, head: str, relation: str, tail: str):
        """Add triples to the knowledge graph"""
        # Get entityID
        head_id = self._get_or_create_entity_id(head)
        tail_id = self._get_or_create_entity_id(tail)
        
        # Store triples (using encoded pairs)
        self.triples.add(self.encode_pair(head_id, tail_id))
        self.relations.add(relation)
        self.base_relations.add(relation)
        
        # Creater2h2tIndex
        self.r2h2t[relation][head_id].add(tail_id)
        
        # Createinverserelationship index
        inverse_relation = f"INVERSE_{relation}"
        self.relations.add(inverse_relation)
        self.base_relations.add(inverse_relation)
        self.r2h2t[inverse_relation][tail_id].add(head_id)
    
    def clear_cached_relations(self):
        """Clear cached composite relationships, retaining underlying relationships"""
        # Find all non-base relations (i.e. cached composite relations)
        cached_relations = [r for r in self.relations if r not in self.base_relations]
        
        # fromr2h2tDelete in
        for relation in cached_relations:
            if relation in self.r2h2t:
                del self.r2h2t[relation]
            self.relations.discard(relation)
    
    def get_relation_pairs(self, relation: str) -> Set[int]:
        """Get the set of all coded pairs for a relationship"""
        pairs = set()
        if relation in self.r2h2t:
            for head, tails in self.r2h2t[relation].items():
                for tail in tails:
                    pairs.add(self.encode_pair(head, tail))
        return pairs
    
    def get_relation_instances_count(self, relation: str) -> int:
        """Get the number of instances of a relationship"""
        count = 0
        if relation in self.r2h2t:
            for head, tails in self.r2h2t[relation].items():
                count += len(tails)
        return count
    
    @staticmethod
    def get_inverse_relation(relation: str) -> str:
        """
        Obtain the inverse relationship of a relationship and support compound relationship paths
        
        Examples:
        - r1 -> INVERSE_r1
        - INVERSE_r1 -> r1
        - r1*r2*r3 -> INVERSE_r3*INVERSE_r2*INVERSE_r1
        - INVERSE_r3*INVERSE_r2*INVERSE_r1 -> r1*r2*r3
        """
        if '*' in relation:
            # compound relationship path
            parts = relation.split('*')
            inverse_parts = []
            
            for part in reversed(parts):
                if part.startswith("INVERSE_"):
                    # If it is already an inverse relationship, return to the original relationship
                    inverse_parts.append(part[8:])
                else:
                    # If it is a positive relationship, return the inverse relationship
                    inverse_parts.append(f"INVERSE_{part}")
            
            return '*'.join(inverse_parts)
        else:
            # single relationship
            if relation.startswith("INVERSE_"):
                # If it is already an inverse relationship, return to the original relationship
                return relation[8:]
            else:
                # If it is a positive relationship, return the inverse relationship
                return f"INVERSE_{relation}"
    
    def save_r2h2t_to_json(self, filepath: str):
        """willr2h2tindex saved toJSONFile"""
        debug(f"Savingr2h2tIndex to file: {filepath}")
        
        # Convertr2h2tin serializable format
        serializable_r2h2t = {}
        for relation, h2t_dict in self.r2h2t.items():
            serializable_r2h2t[relation] = {}
            for head, tails in h2t_dict.items():
                serializable_r2h2t[relation][head] = list(tails)
        
        # save toJSONFile
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_r2h2t, f, ensure_ascii=False, indent=2)
        
        debug(f"r2h2tThe index has been saved and contains {len(serializable_r2h2t)} relationship")


class RuleParser:
    """Rule parser, supporting multiple rule formats and abbreviations"""
    
    @staticmethod
    def _is_variable(arg: str) -> bool:
        """
        Determine whether a parameter is a variable
        
        Variable definition:
        1. single letter (X, Y, A, Betc.)
        2. special mark me_myself_i (Represents a self-referential variable)
        
        Args:
            arg: parameter string
            
        Returns:
            TrueIf it is a variable,Falseif it is constant
        """
        return len(arg) == 1 or arg == 'me_myself_i'
    
    @staticmethod
    def _normalize_me_myself_i(args: List[str], context: str = 'head') -> List[str]:
        """
        Normalized inclusion me_myself_i parameter list
        
        me_myself_i is a special mark indicating that a variable at that location should be the same as a variable at another location.
        For example:
        - /rel(X, me_myself_i) should be regarded as /rel(X, X)
        - /rel(me_myself_i) should be regarded as /rel(X, X) some abbreviation of
        
        Args:
            args: Parameter list
            context: context('head' or 'body') , Used to decide replacement strategies
            
        Returns:
            Normalized parameter list
        """
        if 'me_myself_i' not in args:
            return args
        
        normalized = []
        # Find the first non- me_myself_i variable
        first_var = None
        for arg in args:
            if RuleParser._is_variable(arg) and arg != 'me_myself_i':
                first_var = arg
                break
        
        # If no other variable is found, use the default variableX
        if first_var is None:
            first_var = 'X'
        
        # will all me_myself_i Replace with first variable
        for arg in args:
            if arg == 'me_myself_i':
                normalized.append(first_var)
            else:
                normalized.append(arg)
        
        return normalized
    
    @staticmethod
    def parse_rule(rule_str: str) -> Tuple[str, List[str], int, Dict]:
        """
        Parse the rule string and convert all rules into abbreviation mode for unified processing
        
        The unified rule format is abbreviated mode:
        - One dollar rule:/rel(/m/const) <= /rel1*/rel2(/m/const2)
        - Binary rules:/rel <= /rel1*INVERSE_/rel2
        - Complex rules (Complex Rule) : bodycontains&&, Indicates multiplebranch
        
        Args:
            rule_str: rule string
            
        Returns:
            (head_relation, body_relations, variable_count, rule_info)
        """
        if '<=' not in rule_str:
            raise ValueError("Rule malformed: missing '<='")
        
        head_part, body_part = rule_str.split('<=', 1)
        head_part = head_part.strip()
        body_part = body_part.strip()
        
        # Check if it iscomplex rule (bodycontains&&) 
        if '&&' in body_part:
            debug(f"[DEBUG] detectedComplex Rule (contains&&) ")
            return RuleParser._parse_complex_rule(head_part, body_part, rule_str)
        
        # First detect the rule type and convert to abbreviation mode
        normalized_rule = RuleParser._normalize_to_simplified(head_part, body_part)
        
        rule_info = {
            'original_rule': rule_str,
            'normalized_rule': normalized_rule,
            'is_simplified': True  # Unify into abbreviation mode
        }
        
        # Parse the normalized abbreviation rules
        norm_head, norm_body = normalized_rule.split('<=', 1)
        norm_head = norm_head.strip()
        norm_body = norm_body.strip()
        
        return RuleParser._parse_simplified_rule(norm_head, norm_body, rule_info)
    
    @staticmethod
    def _parse_complex_rule(head_part: str, body_part: str, rule_str: str) -> Tuple[str, List[str], int, Dict]:
        """
        parseComplex Rule (bodycontains&&rules)
        
        Complex RuleDivided into two types:
        1. Complex Binary Rule: headIt is a binary relationship,bodyThere are multiplebranch (use&&separated)
           For example:/location/country/form_of_government(X,Y) <= branch1&&branch2&&branch3
           
        2. Complex Unary Rule: headIt is a uniary relationship,bodyThere are multiplebranch (use&&separated)
           For example:INVERSE_/government/legislative_session/members./government/government_position_held/legislative_sessions(/m/01gsvb) 
                 <= branch1&&branch2
        
        Processing method:
        - will eachbranchConvert to abbreviated form if not already abbreviated
        - save allbranchesInformation arrivesrule_info, used for subsequent calculations
        - body instances = branch1_instances ∩ branch2_instances ∩ ...
        
        Args:
            head_part: header string
            body_part: body string (contains&&) 
            rule_str: Original rule string
            
        Returns:
            (head_relation, body_relations_list, variable_count, rule_info)
        """
        debug(f"[DEBUG] parseComplex Rule:")
        debug(f"[DEBUG]   Head: {head_part}")
        debug(f"[DEBUG]   Body: {body_part}")
        
        # willbodypress&&split into multiplebranches
        branches = [branch.strip() for branch in body_part.split('&&')]
        debug(f"[DEBUG]   discover {len(branches)} abranches")
        
        # Determine the rule type (unary or binary)
        # CheckheadWhether to include variables
        is_unary = False
        if '(' in head_part and ')' in head_part:
            paren_content = head_part.split('(')[1].split(')')[0]
            # If there is only one parameter in the brackets, it is a unary rule
            if ',' not in paren_content:
                is_unary = True
            else:
                # If there is a comma, check whether there are two variables (binary) or one variable and a constant (unary)
                args = [arg.strip() for arg in paren_content.split(',')]
                # Standardize me_myself_i
                args = RuleParser._normalize_me_myself_i(args, 'head')
                # Count the number of single-letter variables (real variables)
                var_count = sum(1 for arg in args if RuleParser._is_variable(arg))
                is_unary = (var_count == 1)
        else:
            # Without parentheses, it is a shorthand binary rule.
            is_unary = False
        
        debug(f"[DEBUG]   Rule type: {'one yuan' if is_unary else 'Binary'}")
        
        # parseheadpart
        if is_unary:
            # unary rule - Need to be converted to abbreviated form
            # For example: /rel(/m/const, X) or /rel(X, /m/const) or already in abbreviated form /rel(/m/const) or INVERSE_/rel(/m/const)
            if ',' in head_part:
                # Full format, needs to be converted to abbreviation
                head_relation_base = head_part.split('(')[0].strip()
                args = [arg.strip() for arg in head_part.split('(')[1].split(')')[0].split(',')]
                # Find the location of constants and variables
                if len(args[0]) > 1:  # The first parameter is a constant
                    fixed_entity = args[0]
                    head_relation = f"INVERSE_{head_relation_base}"
                    var_pos = 'tail'
                else:  # The second parameter is a constant
                    fixed_entity = args[1]
                    head_relation = head_relation_base
                    var_pos = 'head'
            else:
                # Already in abbreviated format
                head_relation, fixed_entity, var_pos = RuleParser._parse_simplified_head(head_part)
            variable_count = 1
        else:
            # binary rule
            # Extracthead relation (Remove possible parentheses and arguments)
            if '(' in head_part:
                head_relation = head_part.split('(')[0].strip()
            else:
                head_relation = head_part.strip()
            fixed_entity = None
            variable_count = 2
        
        # Process eachbranch, Convert it to abbreviated form
        simplified_branches = []
        branch_info_list = []
        
        for i, branch in enumerate(branches):
            debug(f"[DEBUG]   ProcessBranch {i+1}: {branch}")
            
            # CheckbranchWhether it is already in abbreviated format
            if '(' not in branch or ',' not in branch:
                # Already in abbreviated format
                simplified_branch = branch
                branch_relations, branch_constant = RuleParser._parse_simplified_body(simplified_branch)
            else:
                # Full format, needs to be converted to abbreviated format
                # forcomplex ruleof eachbranch, Need to convert
                # For example:/rel1(X,A), /rel2(A,/m/const) -> /rel1*/rel2(/m/const)
                # Or:/rel1(X,A), /rel2(Y,A) -> /rel1*INVERSE_/rel2
                
                # parsebranchatoms in
                branch_atoms = RuleParser._parse_body_atoms(branch)
                debug(f"[DEBUG]     Branchnumber of atoms: {len(branch_atoms)}")
                
                # Convert according to rule typebranch
                if is_unary:
                    # univariate rulebranchConvert
                    # The free variable needs to be found (assumed here isX) 
                    free_var = 'X'
                    simplified_branch, branch_constant = RuleParser._convert_branch_to_simplified_unary(branch_atoms, free_var)
                    branch_relations, _ = RuleParser._parse_simplified_body(simplified_branch)
                else:
                    # binary rulebranchConvert
                    # The two variables in the header areXandY
                    free_vars = ['X', 'Y']
                    simplified_branch = RuleParser._convert_branch_to_simplified_binary(branch_atoms, free_vars)
                    branch_relations, branch_constant = RuleParser._parse_simplified_body(simplified_branch)
                    
                debug(f"[DEBUG]     full formatbranch: {branch}")
            
            simplified_branches.append(simplified_branch)
            branch_info_list.append({
                'branch_text': simplified_branch,
                'relations': branch_relations,
                'constant': branch_constant
            })
            
            debug(f"[DEBUG]     abbreviation: {simplified_branch}")
            debug(f"[DEBUG]     relationship path: {branch_relations}")
            debug(f"[DEBUG]     constant: {branch_constant}")
        
        # buildrule_info
        rule_info = {
            'original_rule': rule_str,
            'normalized_rule': f"{head_part} <= {'&&'.join(simplified_branches)}",
            'is_simplified': True,
            'is_complex': True,
            'branch_count': len(branches),
            'branches': branch_info_list,
            'head_relation': head_relation,
            'is_unary': is_unary,
            'variable_count': variable_count
        }
        
        if is_unary:
            rule_info['head_constant'] = fixed_entity
            rule_info['free_variable'] = 'X'
            rule_info['head_atom'] = {
                'relation': head_relation,
                'args': ['X', fixed_entity] if not head_relation.startswith('INVERSE_') else [fixed_entity, 'X']
            }
            rule_info['head_variables'] = ['X', fixed_entity] if not head_relation.startswith('INVERSE_') else [fixed_entity, 'X']
            rule_info['free_variables'] = ['X']
        else:
            rule_info['head_constant'] = None
            rule_info['body_constant'] = None
            rule_info['free_variables'] = ['X', 'Y']
            rule_info['head_atom'] = {
                'relation': head_relation,
                'args': ['X', 'Y']
            }
            rule_info['head_variables'] = ['X', 'Y']
        
        # Return value:head_relation, body_relations (allbranchesrelationship list),variable_count, rule_info
        # forcomplex rule, body_relationsis an all-encompassingbrancheslist of
        all_body_relations = []
        for branch_info in branch_info_list:
            all_body_relations.extend(branch_info['relations'])
        
        debug(f"[DEBUG] Complex RuleParsing completed:")
        debug(f"[DEBUG]   Head relation: {head_relation}")
        debug(f"[DEBUG]   Variable count: {variable_count}")
        debug(f"[DEBUG]   Branch count: {len(branches)}")
        
        return head_relation, all_body_relations, variable_count, rule_info
    
    @staticmethod
    def _normalize_to_simplified(head_part: str, body_part: str) -> str:
        """
        Convert full format rules to abbreviated format
        
        Conversion rules:
        1. One dollar rule:rel(X,/m/const) <= body1(X,A), body2(A,/m/const2)
           -> rel(/m/const) <= body_path(/m/const2) or INVERSE_rel(/m/const) <= body_path(/m/const2)
        
        2. Binary rules:rel(X,Y) <= body1(X,A), body2(Y,A)  
           -> rel <= body_path
        
        Args:
            head_part: header string
            body_part: body string
            
        Returns:
            Normalized abbreviation rule string
        """
        # Check if it is already in abbreviated format
        if '(' not in head_part or ')' not in head_part:
            # Binary rule abbreviation format
            return f"{head_part} <= {body_part}"
        
        # Check if there is a comma in the header
        paren_content = head_part.split('(')[1].split(')')[0]
        if ',' not in paren_content:
            # Unary rule abbreviation format
            return f"{head_part} <= {body_part}"
        
        # Full format, needs to be converted
        debug(f"[DEBUG] Converting to simplified format: {head_part} <= {body_part}")
        
        # Parse header
        head_relation = head_part.split('(')[0].strip()
        head_args = [arg.strip() for arg in paren_content.split(',')]
        
        # Analyze body atoms
        body_atoms = RuleParser._parse_body_atoms(body_part)
        
        # Analyze variables
        all_vars = set()
        constants = set()
        # Standardize first head_args in me_myself_i
        head_args = RuleParser._normalize_me_myself_i(head_args, 'head')
        for arg in head_args:
            if RuleParser._is_variable(arg):  # variable
                all_vars.add(arg)
            else:  # constant
                constants.add(arg)
        
        for atom in body_atoms:
            if '(' in atom and ')' in atom:
                atom_args = RuleParser._extract_variables(atom)
                # Standardize body in me_myself_i
                atom_args = RuleParser._normalize_me_myself_i(atom_args, 'body')
                for arg in atom_args:
                    if RuleParser._is_variable(arg):  # variable
                        all_vars.add(arg)
                    else:  # constant
                        constants.add(arg)
        
        # Determine rule type
        # Check whether it is a self-loop rule (headThe two parameters of are the same variable)
        is_self_loop = (len(head_args) == 2 and 
                       len(head_args[0]) == 1 and 
                       head_args[0] == head_args[1])
        
        # Free variables are different single-letter variables in the header parameters
        free_vars_count = len(set(arg for arg in head_args if RuleParser._is_variable(arg)))
        
        if free_vars_count == 1 or is_self_loop:
            # Unary rule: the head has a variable and a constant, or a self-loop rule
            return RuleParser._convert_unary_to_simplified(head_relation, head_args, body_atoms, is_self_loop)
        else:
            # Binary rule: there are two different variables in the header
            return RuleParser._convert_binary_to_simplified(head_relation, head_args, body_atoms)
    
    @staticmethod
    def _convert_unary_to_simplified(head_relation: str, head_args: List[str], 
                                   body_atoms: List[str], is_self_loop: bool = False) -> str:
        """
        Convert unary rules to shorthand format
        
        For example:rel(X,/m/const) <= body1(X,A), body2(A,/m/const2)
        Converts to:rel(/m/const) <= body_path(/m/const2) or INVERSE_rel(/m/const) <= body_path(/m/const2)
        
        Self-loop rules:rel(X,X) <= body1(/m/const,X)
        Converts to:rel(X) <= INVERSE_body1(/m/const)
        
        The free variable fixation of the unary rule isX (the only single-letter variable in the header)
        """
        # Handle self-loop rules
        if is_self_loop:
            debug(f"[DEBUG] Self-loop unary conversion: {head_relation}({head_args[0]},{head_args[1]})")
            free_var = head_args[0]  # original variable name
            # buildbodypath
            body_path, body_constant = RuleParser._build_unary_body_path(body_atoms, free_var)
            # The abbreviated form of the self-loop rule:/rel(X) <= body_path(constant)
            # Note: The header of the self-loop rule is written as /rel(X) Indicates calculation X -rel-> X
            # Use uniformlyXas a variable name instead of retaining the original variable name (e.g.Y) 
            simplified_head = f"{head_relation}(X)"
            if body_constant:
                simplified_body = f"{body_path}({body_constant})"
            else:
                # Check if there are intermediate variables
                has_intermediate_var = len(body_atoms) > 1
                if not has_intermediate_var and len(body_atoms) == 1:
                    atom = body_atoms[0]
                    args = RuleParser._extract_variables(atom)
                    args = RuleParser._normalize_me_myself_i(args, 'body')
                    has_intermediate_var = len(args) == 2 and all(RuleParser._is_variable(arg) for arg in args)
                
                if has_intermediate_var:
                    simplified_body = f"{body_path}(*)"
                else:
                    simplified_body = body_path
            result = f"{simplified_head} <= {simplified_body}"
            debug(f"[DEBUG] Self-loop simplified result: {result}")
            return result
        
        # Find free variables and constants
        free_var = None
        head_constant = None
        free_var_pos_in_head = -1
        
        for i, arg in enumerate(head_args):
            if RuleParser._is_variable(arg):  # variable
                if free_var is None:  # Only take the first variable
                    free_var = arg
                    free_var_pos_in_head = i
            else:  # constant
                head_constant = arg
        
        debug(f"[DEBUG] Unary conversion: free_var={free_var}, pos={free_var_pos_in_head}, constant={head_constant}")
        
        # buildbodypath
        body_path, body_constant = RuleParser._build_unary_body_path(body_atoms, free_var)
        
        # Determine the short form of the head
        # Note: For unary rules, if there is only one variable in the header, it needs to be replaced uniformly withX
        # For example:/location/hud_county_place/place(me_myself_i,Y) should be simplified to /location/hud_county_place/place(X)
        if head_constant:
            # There are constant situations
            if free_var_pos_in_head == 0:
                # rel(X, /m/const) -> rel(/m/const)
                simplified_head = f"{head_relation}({head_constant})"
            else:
                # rel(/m/const, X) -> INVERSE_rel(/m/const)
                simplified_head = f"INVERSE_{head_relation}({head_constant})"
        else:
            # There is no constant, indicating that the header is rel(X, Y) But it is actually a univariate rule (XandYare the same variable or one of them isme_myself_i) 
            # This situation should be uniformly displayed as rel(X)
            simplified_head = f"{head_relation}(X)"
        
        # Build a complete abbreviation rule
        # univariate rulebodySome parts require parentheses:
        # - if there isbodyConstant, in the format body_path(constant)
        # - if notbodyConstants (only intermediate variables), in the format body_path(*), Indicates that there are intermediate variables
        if body_constant:
            simplified_body = f"{body_path}({body_constant})"
        else:
            # Check if there are intermediate variables (bodynumber of atoms > 1, or there are non-free variables in a single atom)
            has_intermediate_var = len(body_atoms) > 1
            if not has_intermediate_var and len(body_atoms) == 1:
                # Single atom, check if there is an intermediate variable
                atom = body_atoms[0]
                args = RuleParser._extract_variables(atom)
                # Standardize me_myself_i
                args = RuleParser._normalize_me_myself_i(args, 'body')
                # If there are two parameters and both are variables (single letters), it means there is an intermediate variable
                has_intermediate_var = len(args) == 2 and all(RuleParser._is_variable(arg) for arg in args)
            
            if has_intermediate_var:
                simplified_body = f"{body_path}(*)"
            else:
                simplified_body = body_path
        
        result = f"{simplified_head} <= {simplified_body}"
        debug(f"[DEBUG] Unary simplified result: {result}")
        return result
    
    @staticmethod
    def _build_unary_body_path(body_atoms: List[str], free_var: str) -> Tuple[str, Optional[str]]:
        """
        Constructing unary rulesbodypath
        
        analysisbodyHow connections are made in atoms, determining correct relationship paths and constants
        
        For example:body1(X,A), body2(A,/m/const) 
        -> XPassAConnect to/m/const, The path is body1*body2, The constant is/m/const
        
        Args:
            body_atoms: bodyatom list
            free_var: free variable
            
        Returns:
            (body_path, body_constant)
        """
        if not body_atoms:
            return "", None
        
        # parse each atom
        parsed_atoms = []
        body_constant = None
        
        for atom in body_atoms:
            relation = RuleParser._extract_relation_from_atom(atom)
            args = RuleParser._extract_variables(atom)
            parsed_atoms.append({'relation': relation, 'args': args})
            
            # Find constants
            for arg in args:
                if len(arg) > 1:  # constant
                    body_constant = arg
        
        debug(f"[DEBUG] Building unary body path: atoms={[(a['relation'], a['args']) for a in parsed_atoms]}")
        debug(f"[DEBUG] Free var: {free_var}, Body constant: {body_constant}")
        
        if len(parsed_atoms) == 1:
            # single atom
            atom = parsed_atoms[0]
            args = atom['args']
            
            # Determine the location of free variables
            free_var_pos = -1
            for i, arg in enumerate(args):
                if arg == free_var:
                    free_var_pos = i
                    break
            
            if free_var_pos == 0:
                # The free variable is inheadlocation, direct usage relationship
                return atom['relation'], body_constant
            else:
                # The free variable is intailposition, using the inverse relationship
                return f"INVERSE_{atom['relation']}", body_constant
        
        # Multiple atoms need to be analyzed and connected
        return RuleParser._analyze_unary_connection(parsed_atoms, free_var, body_constant)
    
    @staticmethod
    def _analyze_unary_connection(parsed_atoms: List[Dict], free_var: str, body_constant: str) -> Tuple[str, str]:
        """
        Analyze the connection methods of multiple atoms in unary rules
        
        Goal: Construct a path from free variables to constants
        
        For example:body1(X,A), body2(A,/m/const)
        - Xinbody1location0, Aat location1
        - Ainbody2location0, constant in position1  
        - Connection:X -> A -> /m/const
        - Path:body1 ∘ body2
        
        Args:
            parsed_atoms: parsed atom list
            free_var: free variable
            body_constant: bodyconstants in
            
        Returns:
            (relation_path, constant)
        """
        if len(parsed_atoms) <= 1:
            atom = parsed_atoms[0]
            return atom['relation'], body_constant
        
        # Find the atom containing the free variable (starting atom)
        start_atom_idx = -1
        for i, atom in enumerate(parsed_atoms):
            if free_var in atom['args']:
                start_atom_idx = i
                break
        
        if start_atom_idx == -1:
            # No atoms containing free variables found, use the first
            start_atom_idx = 0
        
        # Build path starting from starting atom
        path_relations = []
        current_atom = parsed_atoms[start_atom_idx]
        used_atoms = {start_atom_idx}
        
        # Determine the position of the free variable in the starting atom
        free_var_pos = -1
        for i, arg in enumerate(current_atom['args']):
            if arg == free_var:
                free_var_pos = i
                break
        
        # Determine whether an inverse relationship is needed based on the position of the free variable
        if free_var_pos == 0:
            # The free variable is inheadlocation, direct usage relationship
            path_relations.append(current_atom['relation'])
            current_var = current_atom['args'][1]  # connection variables
        else:
            # The free variable is intailposition, using the inverse relationship
            path_relations.append(f"INVERSE_{current_atom['relation']}")
            current_var = current_atom['args'][0]  # connection variables
        
        debug(f"[DEBUG] Starting path with {path_relations[0]}, current_var={current_var}")
        
        # Continue to connect the remaining atoms
        while len(used_atoms) < len(parsed_atoms):
            found_next = False
            
            for i, atom in enumerate(parsed_atoms):
                if i in used_atoms:
                    continue
                
                if current_var in atom['args']:
                    used_atoms.add(i)
                    
                    # Determine the position of the connection variable in the current atom
                    var_pos = atom['args'].index(current_var)
                    
                    if var_pos == 0:
                        # The connection variable is inheadlocation, direct usage relationship
                        path_relations.append(atom['relation'])
                        # The next variable istailposition variable
                        next_var = atom['args'][1] if len(atom['args']) > 1 else None
                    else:
                        # The connection variable is intailposition, using the inverse relationship
                        path_relations.append(f"INVERSE_{atom['relation']}")
                        # The next variable isheadposition variable
                        next_var = atom['args'][0]
                    
                    debug(f"[DEBUG] Added {path_relations[-1]}, next_var={next_var}")
                    current_var = next_var
                    found_next = True
                    break
            
            if not found_next:
                break
        
        # Build final path
        final_path = '*'.join(path_relations)
        debug(f"[DEBUG] Final unary path: {final_path}")
        
        return final_path, body_constant
    
    @staticmethod
    def _convert_binary_to_simplified(head_relation: str, head_args: List[str], 
                                    body_atoms: List[str]) -> str:
        """
        Convert binary rules to shorthand format
        
        For example:rel(X,Y) <= body1(X,A), body2(Y,A)
        Converts to:rel <= body1*INVERSE_body2
        
        The free variable fixation of the binary rule is head_args (That is X, Y, order determined)
        """
        # Extract free variables (single letter variables in header)
        head_args = RuleParser._normalize_me_myself_i(head_args, 'head')
        free_vars = [arg for arg in head_args if RuleParser._is_variable(arg)]
        
        if len(free_vars) != 2:
            # If it is not a strict binary rule, return the original format
            return f"{head_relation}({','.join(head_args)}) <= {', '.join(body_atoms)}"
        
        # buildbodyPath, passing in an ordered list of free variables
        body_path = RuleParser._build_binary_body_path(body_atoms, free_vars)
        
        result = f"{head_relation} <= {body_path}"
        debug(f"[DEBUG] Binary simplified result: {result}")
        return result
    
    @staticmethod
    def _build_binary_body_path(body_atoms: List[str], free_vars: List[str]) -> str:
        """
        Constructing binary rulesbodypath
        
        Analyze connection patterns and build correct relationship paths
        For example:
        - body1(X,A), body2(Y,A) -> body1*INVERSE_body2 (X->A<-Y)
        - body1(X,A), body2(A,B), body3(Y,B) -> body1*body2*INVERSE_body3 (X->A->B<-Y)
        """
        if not body_atoms:
            return ""
        
        # Analyze atoms
        parsed_atoms = []
        for atom in body_atoms:
            relation = RuleParser._extract_relation_from_atom(atom)
            args = RuleParser._extract_variables(atom)
            parsed_atoms.append({'relation': relation, 'args': args})
        
        debug(f"[DEBUG] Building binary body path: atoms={[(a['relation'], a['args']) for a in parsed_atoms]}")
        debug(f"[DEBUG] Free vars: {free_vars}")
        
        if len(parsed_atoms) == 1:
            # Single atom, extract relational part
            atom = parsed_atoms[0]
            # A single atom of a binary rule returns the relationship directly (you may need to addINVERSE) 
            if len(free_vars) == 2:
                X, Y = free_vars[0], free_vars[1]
                args = atom['args']
                # Check if parameter order matches
                if args[0] == X and args[1] == Y:
                    # Sequential matching, direct use of relationships
                    return atom['relation']
                elif args[0] == Y and args[1] == X:
                    # In reverse order, use the inverse relationship
                    return f"INVERSE_{atom['relation']}"
                else:
                    # In other cases, return the relationship directly
                    return atom['relation']
            return atom['relation']
        
        if len(free_vars) != 2:
            # Simplified processing: connect all relationships directly
            return '*'.join([atom['relation'] for atom in parsed_atoms])
        
        # Analyze connection mode: fromXstart, findYpath
        X, Y = free_vars[0], free_vars[1]
        
        # found containingXatom as starting point
        start_atom_idx = -1
        for i, atom in enumerate(parsed_atoms):
            if X in atom['args']:
                start_atom_idx = i
                break
        
        if start_atom_idx == -1:
            # Not found containingXatoms, using simplified processing
            return '*'.join([atom['relation'] for atom in parsed_atoms])
        
        # Build fromXArriveYconnection path
        path_relations = []
        current_atom = parsed_atoms[start_atom_idx]
        used_atoms = {start_atom_idx}
        
        # OKXposition in starting atom
        x_pos = current_atom['args'].index(X)
        
        if x_pos == 0:
            # Xinheadlocation, direct usage relationship
            path_relations.append(current_atom['relation'])
            current_var = current_atom['args'][1]  # connection variables
        else:
            # Xintailposition, using the inverse relationship
            path_relations.append(f"INVERSE_{current_atom['relation']}")
            current_var = current_atom['args'][0]  # connection variables
        
        debug(f"[DEBUG] Starting from {X} with {path_relations[0]}, current_var={current_var}")
        
        # Keep connecting until you findY
        while current_var != Y and len(used_atoms) < len(parsed_atoms):
            found_next = False
            
            for i, atom in enumerate(parsed_atoms):
                if i in used_atoms:
                    continue
                
                if current_var in atom['args']:
                    used_atoms.add(i)
                    
                    # Determine the position of the connection variable in the current atom
                    var_pos = atom['args'].index(current_var)
                    
                    if var_pos == 0:
                        # The connection variable is inheadlocation, direct usage relationship
                        path_relations.append(atom['relation'])
                        next_var = atom['args'][1]
                    else:
                        # The connection variable is intailposition, using the inverse relationship
                        path_relations.append(f"INVERSE_{atom['relation']}")
                        next_var = atom['args'][0]
                    
                    debug(f"[DEBUG] {current_var} -> {next_var} via {path_relations[-1]}")
                    current_var = next_var
                    found_next = True
                    break
            
            if not found_next:
                debug(f"[DEBUG] Cannot find connection from {current_var}")
                break
        
        final_path = '*'.join(path_relations)
        debug(f"[DEBUG] Final binary path: {final_path}")
        
        return final_path

    @staticmethod
    def _convert_branch_to_simplified_unary(branch_atoms: List[str], free_var: str) -> Tuple[str, Optional[str]]:
        """
        convert the unary rulebranchConvert from full format to abbreviated format
        
        For example:/rel1(X,A), /rel2(A,/m/const) -> /rel1*/rel2(/m/const)
        
        Args:
            branch_atoms: branchlist of atoms
            free_var: free variable (usually'X') 
            
        Returns:
            (simplified_branch, branch_constant)
        """
        # buildbodyPaths and constants
        body_path, body_constant = RuleParser._build_unary_body_path(branch_atoms, free_var)
        
        # Build abbreviation
        if body_constant:
            simplified_branch = f"{body_path}({body_constant})"
        else:
            # Check if there are intermediate variables
            has_intermediate_var = len(branch_atoms) > 1
            if not has_intermediate_var and len(branch_atoms) == 1:
                atom = branch_atoms[0]
                args = RuleParser._extract_variables(atom)
                # Standardize me_myself_i
                args = RuleParser._normalize_me_myself_i(args, 'body')
                has_intermediate_var = len(args) == 2 and all(RuleParser._is_variable(arg) for arg in args)
            
            if has_intermediate_var:
                simplified_branch = f"{body_path}(*)"
            else:
                simplified_branch = body_path
        
        return simplified_branch, body_constant
    
    @staticmethod
    def _convert_branch_to_simplified_binary(branch_atoms: List[str], free_vars: List[str]) -> str:
        """
        Binary rulesbranchConvert from full format to abbreviated format
        
        For example:/rel1(X,A), /rel2(Y,A) -> /rel1*INVERSE_/rel2
        
        Args:
            branch_atoms: branchlist of atoms
            free_vars: list of free variables (usually['X', 'Y']) 
            
        Returns:
            simplified_branch
        """
        # buildbodypath
        body_path = RuleParser._convert_binary_to_simplified_body_path(branch_atoms, free_vars)
        return body_path
    
    @staticmethod
    def _convert_binary_to_simplified_body_path(branch_atoms: List[str], free_vars: List[str]) -> str:
        """
        Binary rulesbodyA list of atoms converted into a shorthand relational path
        
        Args:
            branch_atoms: bodyatom list
            free_vars: free variable list
            
        Returns:
            abbreviated relationship path
        """
        if not branch_atoms:
            return ""
        
        # Use directly_build_binary_body_pathmethod
        return RuleParser._build_binary_body_path(branch_atoms, free_vars)

    @staticmethod
    def _parse_simplified_rule(head_part: str, body_part: str, rule_info: Dict) -> Tuple[str, List[str], int, Dict]:
        """Parse abbreviation format rules and use unified data structure"""
        # Extract header relationships and possible fixed entities
        head_relation, fixed_entity, var_pos = RuleParser._parse_simplified_head(head_part)
        
        # Resolve body relationships, which may include entity constraints
        body_relations, body_constant = RuleParser._parse_simplified_body(body_part)
        
        # Check whether it is a self-loop rule
        is_self_loop = (var_pos == 'self_loop')
        
        # Determine rule type
        if is_self_loop:
            # Self-loop rules:/rel(X) express X -rel-> X
            variable_count = 1
            rule_info.update({
                'is_unary': True,
                'is_self_loop': True,
                'variable_count': 1,
                'head_relation': head_relation,
                'head_constant': None,  # Self-loop rules have no fixed entity
                'body_relations': body_relations,
                'body_constant': body_constant,
                'free_variable': fixed_entity  # herefixed_entityActually it is the variable name (such asX) 
            })
            
            # build standardhead_atomstructure - The self-loop rule is expressed as(X, X)
            rule_info['head_atom'] = {
                'relation': head_relation,
                'args': [fixed_entity, fixed_entity]  # (X, X)
            }
            rule_info['head_variables'] = [fixed_entity, fixed_entity]
            rule_info['free_variables'] = [fixed_entity]
        elif fixed_entity or body_constant:
            # Unary rule: there is a fixed entity
            variable_count = 1
            rule_info.update({
                'is_unary': True,
                'is_self_loop': False,
                'variable_count': 1,
                'head_relation': head_relation,
                'head_constant': fixed_entity,
                'body_relations': body_relations,
                'body_constant': body_constant,
                'free_variable': 'X'  # Use uniformlyXas a free variable name
            })
            
            # build standardhead_atomstructure
            rule_info['head_atom'] = {
                'relation': head_relation,
                'args': ['X', fixed_entity] if not head_relation.startswith('INVERSE_') else [fixed_entity, 'X']
            }
            rule_info['head_variables'] = ['X', fixed_entity] if not head_relation.startswith('INVERSE_') else [fixed_entity, 'X']
            rule_info['free_variables'] = ['X']
        else:
            # Binary rule: no fixed entities
            variable_count = 2
            rule_info.update({
                'is_unary': False,
                'is_self_loop': False,
                'variable_count': 2,
                'head_relation': head_relation,
                'head_constant': None,
                'body_relations': body_relations,
                'body_constant': None,
                'free_variables': ['X', 'Y']
            })
            
            rule_info['head_atom'] = {
                'relation': head_relation,
                'args': ['X', 'Y']
            }
            rule_info['head_variables'] = ['X', 'Y']
        
        return head_relation, body_relations, variable_count, rule_info
    
    @staticmethod
    def _parse_simplified_body(body_part: str) -> Tuple[List[str], Optional[str]]:
        """
        parse abbreviation formatbodypart
        
        For example:/rel1*/rel2(/m/entity) 
        Return:(["/rel1", "/rel2"], "/m/entity")
        
        Special circumstances:/rel(*) express"any entity", Not a constant constraint
        Return:(["/rel"], None)
        
        U0rules (emptybody) : Return empty list
        Return:([], None)
        """
        body_constant = None
        
        # Check if it isU0rules (emptybody) 
        if not body_part or body_part.strip() == '':
            return [], None
        
        # Check if there are bracket constraints
        if '(' in body_part and ')' in body_part:
            # Find the last bracket and extract the constraint entity
            last_paren_start = body_part.rfind('(')
            last_paren_end = body_part.rfind(')')
            
            if last_paren_start < last_paren_end:
                entity_part = body_part[last_paren_start+1:last_paren_end].strip()
                # Check if it is"any entity"placeholder
                if entity_part == '*':
                    # (*) Represents any entity, not a constant constraint
                    # Remove the bracket part, but don't set itbody_constant
                    body_part = body_part[:last_paren_start].strip()
                elif entity_part.startswith('/m/'):
                    # True constant constraints
                    body_constant = entity_part
                    # Remove brackets
                    body_part = body_part[:last_paren_start].strip()
        
        # Parse relationship paths
        if '*' in body_part:
            body_relations = [rel.strip() for rel in body_part.split('*')]
        else:
            body_relations = [body_part.strip()]
        
        return body_relations, body_constant
    
    @staticmethod
    def _parse_full_rule(head_part: str, body_part: str, rule_info: Dict) -> Tuple[str, List[str], int, Dict]:
        """Parse complete format rules and correctly handle variable binding modes"""
        # Extract header relations and variables
        head_relation = RuleParser._extract_relation_from_atom(head_part)
        head_variables = RuleParser._extract_variables(head_part)
        
        # Analyze body atoms
        body_atoms = RuleParser._parse_body_atoms(body_part)
        
        # Analyze variable types
        all_body_variables = []
        for atom in body_atoms:
            variables = RuleParser._extract_variables(atom)
            all_body_variables.extend(variables)
        
        free_vars, constraint_vars = RuleParser._analyze_variables(head_variables, all_body_variables)
        
        # Fix variable counting: consider whether header contains constants
        has_constant_in_head = any(len(var) > 1 for var in head_variables)
        if has_constant_in_head and len(free_vars) == 1:
            # Unary rule: There is one variable and one constant in the header
            variable_count = 1
        elif len(free_vars) == 2 and not has_constant_in_head:
            # Binary Rule: There are two variables in the header
            variable_count = 2
        else:
            # In other cases, the number of free variables is used directly
            variable_count = len(free_vars)
        
        debug(f"  [DEBUG] Head variables: {head_variables}, Free vars: {free_vars}, Has constant: {has_constant_in_head}, Variable count: {variable_count}")
        
        # Store detailed atomic information
        parsed_body_atoms = []
        for atom in body_atoms:
            relation = RuleParser._extract_relation_from_atom(atom)
            variables = RuleParser._extract_variables(atom)
            parsed_body_atoms.append({
                'relation': relation,
                'args': variables,
                'original': atom
            })
        
        # buildhead_atomstructure
        head_variables = RuleParser._extract_variables(head_part)
        rule_info['head_atom'] = {
            'relation': head_relation,
            'args': head_variables
        }
        
        rule_info['body_atoms'] = parsed_body_atoms
        rule_info['head_variables'] = head_variables
        # Keep the order of the free variables consistent with the head variables (as perhead_variablesorder of appearance)
        rule_info['free_variables'] = [v for v in head_variables if v in free_vars]
        rule_info['constraint_variables'] = sorted(list(constraint_vars))  # Constraint variables are sorted alphabetically
        rule_info['variable_count'] = variable_count
        
        # For binary rules, build the connection path
        if variable_count == 2:
            body_relations = RuleParser._build_connection_path(head_variables, parsed_body_atoms)
        else:
            # Unary rules or other situations, directly extract the relationship name
            body_relations = [atom['relation'] for atom in parsed_body_atoms]
        
        # willbody_relationsstore torule_infoin
        rule_info['body_relations'] = body_relations
        
        if variable_count == 1:
            rule_info['is_unary'] = True
            # Find fixed entity and variable locations
            for i, var in enumerate(head_variables):
                if len(var) > 1:  # constant (Entity)
                    rule_info['fixed_entity'] = var
                    rule_info['variable_position'] = 'tail' if i == 1 else 'head'
                    break
        
        return head_relation, body_relations, variable_count, rule_info
    
    @staticmethod
    def _build_connection_path(head_variables: List[str], body_atoms: List[Dict]) -> List[str]:
        """
        Build connection paths based on variable binding patterns
        
        For binary rules Head(X,Y) <= Body1(...), Body2(...), ...
        need to find fromXArriveYconnection path
        """
        if len(head_variables) != 2:
            return [atom['relation'] for atom in body_atoms]
        
        X, Y = head_variables[0], head_variables[1]
        
        debug(f"[DEBUG] Building connection path for X={X}, Y={Y}")
        debug(f"[DEBUG] Body atoms: {[(atom['relation'], atom['args']) for atom in body_atoms]}")
        
        # Simplified implementation: assume chain connection
        # found containingXatom as starting point
        path_relations = []
        
        current_var = X
        used_atoms = set()
        
        while current_var != Y and len(used_atoms) < len(body_atoms):
            found_next = False
            
            for i, atom in enumerate(body_atoms):
                if i in used_atoms:
                    continue
                
                relation = atom['relation']
                args = atom['args']
                
                if current_var in args:
                    used_atoms.add(i)
                    
                    # Determine the position of a variable in a relationship, and the next variable
                    if args[0] == current_var:
                        # current_varin first position
                        next_var = args[1]
                        # Use positive relationships:current_var -> next_var
                        path_relations.append(relation)
                    else:
                        # current_varin second position
                        next_var = args[0]
                        # Use an inverse relationship:current_var <- next_var (That is next_var -> current_var)
                        path_relations.append(f"INVERSE_{relation}")
                    
                    debug(f"[DEBUG] {current_var} -> {next_var} via {path_relations[-1]}")
                    current_var = next_var
                    found_next = True
                    break
            
            if not found_next:
                debug(f"[DEBUG] Cannot find connection from {current_var}")
                break
        
        debug(f"[DEBUG] Final connection path: {path_relations}")
        return path_relations
    
    @staticmethod
    def _parse_simplified_head(head_part: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Parse the abbreviated format header
        
        forINVERSE_/rel(/m/entity), Equivalent to/rel(/m/entity, X), Entity isheadlocation
        for/rel(/m/entity), Equivalent to/rel(X, /m/entity), Entity istaillocation
        for/rel(X), is a self-loop rule, which is equivalent to/rel(X, X)
        
        Returns:
            (relation, fixed_entity, variable_position)
        """
        if '(' in head_part and ')' in head_part:
            # With parentheses, probably a unary rule:/rel(/m/123) or INVERSE_/rel(/m/123) or self-loop rule /rel(X) or /rel(me_myself_i)
            relation = head_part.split('(')[0].strip()
            entity_part = head_part.split('(')[1].split(')')[0].strip()
            
            if RuleParser._is_variable(entity_part):
                # Self-loop rules:/rel(X) or /rel(me_myself_i) express /rel(X, X)
                # For self-loop rules, there is no fixed entity, and the variable name is returned as a special token
                # if yes me_myself_i, normalized to X
                if entity_part == 'me_myself_i':
                    entity_part = 'X'
                return relation, entity_part, 'self_loop'
            elif entity_part.startswith('/m/'):
                if relation.startswith('INVERSE_'):
                    # INVERSE_/rel(/m/entity) Equivalent to /rel(/m/entity, X)
                    # Entity isheadposition, the variable is intaillocation
                    return relation, entity_part, 'head'
                else:
                    # /rel(/m/entity) Equivalent to /rel(X, /m/entity)
                    # Entity istailposition, the variable is inheadlocation
                    return relation, entity_part, 'tail'
            else:
                # There is no fixed entity, or the format is wrong
                return relation, None, None
        else:
            # Without parentheses, binary rules:/rel
            return head_part.strip(), None, None
    
    @staticmethod
    def _analyze_variables(head_variables: List[str], body_variables: List[str]) -> Tuple[Set[str], Set[str]]:
        """Analyze variable types and distinguish between free variables and constrained variables"""
        # in the normalized variable list me_myself_i
        head_variables = RuleParser._normalize_me_myself_i(head_variables, 'head')
        body_variables = RuleParser._normalize_me_myself_i(body_variables, 'body')
        
        head_var_set = set(head_variables)
        body_var_set = set(body_variables)
        
        # Free variable: single letter variable orme_myself_i (normalized) and appears in the header
        free_variables = {var for var in head_var_set if RuleParser._is_variable(var)}
        
        # Constraint variables: variables that are variables but not in the head (Such asA, B, CWait)
        constraint_variables = set()
        for var in body_var_set:
            if RuleParser._is_variable(var) and var not in head_var_set:
                constraint_variables.add(var)
        
        return free_variables, constraint_variables
    
    @staticmethod
    def _extract_relation_from_atom(atom: str) -> str:
        """Extract relation name from atom"""
        if '(' in atom:
            return atom.split('(')[0].strip()
        else:
            return atom.strip()
    
    @staticmethod
    def _extract_variables(atom: str) -> List[str]:
        """Extract variables from atoms (including normalization me_myself_i) """
        if '(' not in atom or ')' not in atom:
            return []
        
        var_part = atom.split('(')[1].split(')')[0]
        variables = [v.strip() for v in var_part.split(',')]
        # Standardize me_myself_i
        variables = RuleParser._normalize_me_myself_i(variables, 'extracted')
        return variables
    
    @staticmethod
    def _parse_body_atoms(body_part: str) -> List[str]:
        """Parse the atomic list of body parts"""
        atoms = []
        current_atom = ""
        paren_count = 0
        
        for char in body_part:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                atoms.append(current_atom.strip())
                current_atom = ""
                continue
            
            current_atom += char
        
        if current_atom.strip():
            atoms.append(current_atom.strip())
        
        return atoms


class RuleSupportCalculator:
    """Rule support calculator, based onr2h2tIndexing and step-by-step join algorithms"""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        # Instance cache, which stores a collection of instances for each path
        self.instance_cache = {}
    
    def join_relations(self, r1: str, r2: str) -> str:
        """
        Join two relations, create a composite relation and cache it toKGin
        
        Args:
            r1: first relationship
            r2: second relationship
            
        Returns:
            Composite relationship name (r1*r2)
        """
        composite_name = f"{r1}*{r2}"
        
        # If the compound relationship already exists, return directly
        if composite_name in self.kg.r2h2t:
            return composite_name
        
        # Compute examples of composite relationships
        instances = self._join_two_relations(r1, r2)
        
        # Add compound relationship toKGofr2h2tIndexing
        for head, tail in instances:
            self.kg.r2h2t[composite_name][head].add(tail)
        
        # Added to the relationship collection (but not to thebase_relations, Because this is a cached composite relationship)
        self.kg.relations.add(composite_name)
        
        debug(f"    [DEBUG] Created composite relation: {composite_name} with {len(instances)} instances")
        
        return composite_name
    
    def get_binary_instances_join(self, relation_path: List[str]) -> Set[int]:
        """
        Obtain a collection of instances of a binary rule using a join algorithm
        
        Args:
            relation_path: Relationship path list
        
        Returns:
            coded(X, Y)paired set
        """
        debug(f"    [DEBUG] get_binary_instances_join: relation_path={relation_path}")
        
        if not relation_path:
            return set()
        
        # use new compute_supp method
        result = self.compute_supp(relation_path)
        
        debug(f"    [DEBUG] Final result has {len(result)} instances")
        return result
    
    def compute_supp(self, relation_path: List[str]) -> Set[int]:
        """
        Compute the set of instances of a relationship path, ensuring that all entities on the path are not equal
        
        Args:
            relation_path: Relationship path list
            
        Returns:
            The set of encoding pairs that satisfy all constraints
        """
        path_str = '*'.join(relation_path)
        
        # If it has been calculated, return it directly
        if path_str in self.instance_cache:
            debug(f"      [DEBUG] Using cached instances for {path_str}")
            return self.instance_cache[path_str]
        
        if len(relation_path) == 1:
            # A single relationship, returning its instance directly
            instances = self.kg.get_relation_pairs(relation_path[0])
            self.instance_cache[path_str] = instances
            return instances
        
        if len(relation_path) == 2:
            # Two relationships, directly connected
            instances = self._join_two_relations(relation_path[0], relation_path[1])
            self.instance_cache[path_str] = instances
            return instances
        
        # length >= 3 path, need to consider all N-1 way of splitting
        n = len(relation_path)
        debug(f"      [DEBUG] Computing {n}-path with {n-1} split methods")
        
        # Results for all splits
        split_results = []
        
        # Iterate through all possible split positions
        for split_pos in range(1, n):
            left_path = relation_path[:split_pos]
            right_path = relation_path[split_pos:]
            
            debug(f"      [DEBUG] Split {split_pos}: {' * '.join(left_path)} | {' * '.join(right_path)}")
            
            # Recursively calculate the left and right parts
            left_instances = self.compute_supp(left_path)
            if not left_instances:
                debug(f"      [DEBUG] Left part is empty, this split gives 0 instances")
                # If the left side of a split is empty, the result of this split is the empty set
                split_results.append(set())
                continue
            
            right_instances = self.compute_supp(right_path)
            if not right_instances:
                debug(f"      [DEBUG] Right part is empty, this split gives 0 instances")
                split_results.append(set())
                continue
            
            # Use uniformly instances * instances Connect the left and right parts
            split_result = self._join_two_instance_sets(left_instances, right_instances)
            
            debug(f"      [DEBUG] Split {split_pos} result: {len(split_result)} instances")
            split_results.append(split_result)
        
        # Take the intersection of all split results
        if not split_results:
            result = set()
        else:
            result = split_results[0]
            for split_result in split_results[1:]:
                result = result.intersection(split_result)
        
        debug(f"      [DEBUG] Final {n}-path result (intersection of {len(split_results)} splits): {len(result)} instances")
        
        self.instance_cache[path_str] = result
        return result
    
    def _join_two_relations(self, r1: str, r2: str) -> Set[int]:
        """Connect two relations, ensuring X != A != Y"""
        debug(f"      [DEBUG] Joining two relations: {r1} * {r2}")
        
        r1_h2t = self.kg.r2h2t.get(r1, {})
        r2_h2t = self.kg.r2h2t.get(r2, {})
        
        # Get connection node
        inverse_r1 = self.kg.get_inverse_relation(r1)
        r1_tails = set(self.kg.r2h2t.get(inverse_r1, {}).keys())
        r2_heads = set(r2_h2t.keys())
        connection_nodes = r1_tails.intersection(r2_heads)
        
        debug(f"      [DEBUG] Connection nodes: {len(connection_nodes)}")
        
        result = set()
        for node in connection_nodes:
            r1_heads = self.kg.r2h2t.get(inverse_r1, {}).get(node, set())
            r2_tails = r2_h2t.get(node, set())
            
            for h in r1_heads:
                for t in r2_tails:
                    # ensure h != node != t and h != t
                    if h != node and node != t and h != t:
                        result.add(self.kg.encode_pair(h, t))
        
        debug(f"      [DEBUG] Join result: {len(result)} instances")
        return result
    
    def _join_instances_with_relation(self, instances: Set[int], relation: str) -> Set[int]:
        """Connect a collection of instances with a relationship:instances * relation"""
        debug(f"      [DEBUG] Joining {len(instances)} instances with relation {relation}")
        
        r_h2t = self.kg.r2h2t.get(relation, {})
        result = set()
        
        for encoded in instances:
            x, y = self.kg.decode_pair(encoded)
            # y as a relationship relation of head
            if y in r_h2t:
                for z in r_h2t[y]:
                    # ensure x != y != z and x != z
                    if x != y and y != z and x != z:
                        result.add(self.kg.encode_pair(x, z))
        
        debug(f"      [DEBUG] Result: {len(result)} instances")
        return result
    
    def _join_relation_with_instances(self, relation: str, instances: Set[int]) -> Set[int]:
        """Join a relationship with a collection of instances:relation * instances"""
        debug(f"      [DEBUG] Joining relation {relation} with {len(instances)} instances")
        
        # need to find relation to obtain the inverse relationship of tail -> head mapping
        inverse_rel = self.kg.get_inverse_relation(relation)
        inv_h2t = self.kg.r2h2t.get(inverse_rel, {})
        
        result = set()
        
        for encoded in instances:
            y, z = self.kg.decode_pair(encoded)
            # y as a relationship relation of tail, Find all reachable y of head
            if y in inv_h2t:
                for x in inv_h2t[y]:
                    # ensure x != y != z and x != z
                    if x != y and y != z and x != z:
                        result.add(self.kg.encode_pair(x, z))
        
        debug(f"      [DEBUG] Result: {len(result)} instances")
        return result
    
    def _join_two_instance_sets(self, left: Set[int], right: Set[int]) -> Set[int]:
        """Join two instance collections"""
        # Create right Index of: head -> set of tails
        right_index = defaultdict(set)
        for encoded in right:
            y2, z = self.kg.decode_pair(encoded)
            right_index[y2].add(z)
        
        result = set()
        for encoded in left:
            x, y = self.kg.decode_pair(encoded)
            if y in right_index:
                # for all connected z, Check x != z
                for z in right_index[y]:
                    if x != z:
                        result.add(self.kg.encode_pair(x, z))
        
        return result
    
    def get_unary_instances_bruteforce(self, rule_info: Dict) -> Set[str]:
        """
        Use brute force algorithm to obtain the instance set of unary rules
        
        Args:
            rule_info: Rule information dictionary
        
        Returns:
            The set of all possible values of a variable
        """
        body_atoms = rule_info['body_atoms']
        variable = rule_info['free_variables'][0]
        
        # Start with all entities
        candidates = set(self.kg.entities)
        
        # for eachbodyAtom to filter
        for atom in body_atoms:
            relation = atom['relation']
            args = atom['args']
            
            valid_values = set()
            
            # Get all instances of this relationship
            relation_instances = self.kg.get_relation_pairs(relation)
            
            for head, tail in relation_instances:
                # Check whether the constraints of the current atom are met
                if self._matches_atom_pattern(head, tail, args, variable):
                    # Extract the value of a variable
                    if args[0] == variable:
                        valid_values.add(head)
                    elif args[1] == variable:
                        valid_values.add(tail)
            
            candidates = candidates.intersection(valid_values)
        
        return candidates
    
    def get_binary_instances_bruteforce(self, rule_info: Dict) -> Set[Tuple[str, str]]:
        """
        Use brute force algorithm to obtain a collection of instances of binary rules
        
        Args:
            rule_info: Rule information dictionary
        
        Returns:
            (X, Y) set of pairs
        """
        body_atoms = rule_info['body_atoms']
        free_vars = rule_info['free_variables']
        
        if len(free_vars) != 2:
            return set()
        
        var_x, var_y = free_vars
        valid_pairs = set()
        
        # Violently enumerate all possible (X, Y) combination
        for x in self.kg.entities:
            for y in self.kg.entities:
                if x == y:
                    continue
                
                # check this (x, y) Does it satisfy allbodyatom
                satisfies_all = True
                
                for atom in body_atoms:
                    if not self._satisfies_atom(atom, {var_x: x, var_y: y}):
                        satisfies_all = False
                        break
                
                if satisfies_all:
                    valid_pairs.add((x, y))
        
        return valid_pairs
    
    def _matches_atom_pattern(self, head: str, tail: str, args: List[str], variable: str) -> bool:
        """Check if triplet matches atomic pattern"""
        for i, arg in enumerate(args):
            if arg != variable and not arg.startswith('/m/'):  # Not a variable or a constant
                continue
            
            if i == 0 and arg != variable and arg != head:
                return False
            elif i == 1 and arg != variable and arg != tail:
                return False
        
        return True
    
    def _satisfies_atom(self, atom: Dict, variable_assignment: Dict[str, str]) -> bool:
        """Check whether variable assignment satisfies atomicity"""
        relation = atom['relation']
        args = atom['args']
        
        # Substitute variables
        head_val = variable_assignment.get(args[0], args[0])
        tail_val = variable_assignment.get(args[1], args[1])
        
        # Check if this triplet exists
        return (head_val, relation, tail_val) in self.kg.triples
    
    def calculate_rule_support_join(self, rule_info: Dict) -> Dict:
        """
        Calculate rule support using join algorithm
        
        Args:
            rule_info: Rule information dictionary
        
        Returns:
            contains headSize, bodySize, support, confidence dictionary
        """
        debug(f"[DEBUG] rule_info keys: {list(rule_info.keys())}")
        debug(f"[DEBUG] variable_count: {rule_info.get('variable_count', 'NOT_SET')}")
        
        variable_count = rule_info.get('variable_count', 0)
        body_relations = rule_info.get('body_relations', [])
        is_complex = rule_info.get('is_complex', False)
        
        # unified computingheadandbodyinstance collection
        head_instances = self._get_head_instances(rule_info)
        head_size = len(head_instances)
        
        debug(f"  [DEBUG] Head instances: {head_size}")
        if head_instances:
            head_sample = list(head_instances)[:5]
            debug(f"  [DEBUG] Head sample: {head_sample}")
        
        # Check if it isU0rules (bodyis empty and is notcomplex rule) 
        if not body_relations and not is_complex:
            # U0Rules:bodySizeis the number of all facts of the relationship
            head_relation = rule_info.get('head_relation')
            
            if variable_count == 1:
                # Unary rule: number of facts needed to get the original relation
                if head_relation.startswith('INVERSE_'):
                    original_relation = head_relation[8:]
                    body_size = self.kg.get_relation_instances_count(original_relation)
                else:
                    body_size = self.kg.get_relation_instances_count(head_relation)
            else:
                # Binary Rule: Get the number of facts for a relationship
                body_size = self.kg.get_relation_instances_count(head_relation)
            
            debug(f"  [DEBUG] U0Rules:bodySize (number of relational facts)= {body_size}")
            support = head_size
            confidence = support / body_size if body_size > 0 else 0
            
            debug(f"  [DEBUG] Support: {support}, Confidence: {confidence}")
            
            return {
                'headSize': head_size,
                'bodySize': body_size,
                'support': support,
                'confidence': confidence
            }
        
        # Normal Rules: CalculatebodyExample
        body_instances = self._get_body_instances(rule_info)
        body_size = len(body_instances)
            
        debug(f"  [DEBUG] Body instances: {body_size}")
        if body_instances:
            body_sample = list(body_instances)[:5]
            debug(f"  [DEBUG] Body sample: {body_sample}")
        
        # Normal Rule: Compute Intersection
        support_instances = head_instances.intersection(body_instances)
        support = len(support_instances)
        confidence = support / body_size if body_size > 0 else 0
        
        debug(f"  [DEBUG] Support: {support}, Confidence: {confidence}")
        
        return {
            'headSize': head_size,
            'bodySize': body_size,
            'support': support,
            'confidence': confidence
        }
    
    def _get_head_instances(self, rule_info: Dict) -> Set:
        """Get the head instance collection - Use a unified abbreviation format for processing"""
        head_relation = rule_info.get('head_relation')
        variable_count = rule_info.get('variable_count', 0)
        is_self_loop = rule_info.get('is_self_loop', False)
        
        if is_self_loop:
            # Self-loop rules:rel(X) express rel(X, X)
            # find all satisfaction X -rel-> X entityX
            debug(f"  [DEBUG] Self-loop rule: {head_relation}(X)")
            result = set()
            if head_relation in self.kg.r2h2t:
                for head_id, tail_ids in self.kg.r2h2t[head_relation].items():
                    # Checkhead_idwhether in its owntail_idsin
                    if head_id in tail_ids:
                        result.add(head_id)
            debug(f"  [DEBUG] Found {len(result)} self-loop instances")
            return result
        
        if variable_count == 1:
            # Unary rule: return all possible values of a variable (entityIDcollection)
            head_constant = rule_info.get('head_constant')
            head_constant_id = self.kg.get_entity_id(head_constant)
            
            if head_constant_id is None:
                debug(f"  [DEBUG] Head constant not found: {head_constant}")
                return set()
            
            debug(f"  [DEBUG] Head relation: {head_relation}, constant: {head_constant} (id={head_constant_id})")
            
            if head_relation.startswith('INVERSE_'):
                # INVERSE_relation(constant) express relation(constant, X)
                original_relation = head_relation[8:]
                debug(f"  [DEBUG] Looking for {original_relation}[{head_constant_id}]")
                if original_relation in self.kg.r2h2t and head_constant_id in self.kg.r2h2t[original_relation]:
                    result = set(self.kg.r2h2t[original_relation][head_constant_id])
                    debug(f"  [DEBUG] Found {len(result)} head instances")
                    return result
                else:
                    debug(f"  [DEBUG] No instances found")
            else:
                # relation(constant) express relation(X, constant)
                inverse_relation = self.kg.get_inverse_relation(head_relation)
                debug(f"  [DEBUG] Looking for {inverse_relation}[{head_constant_id}]")
                if inverse_relation in self.kg.r2h2t and head_constant_id in self.kg.r2h2t[inverse_relation]:
                    result = set(self.kg.r2h2t[inverse_relation][head_constant_id])
                    debug(f"  [DEBUG] Found {len(result)} head instances")
                    return result
                else:
                    debug(f"  [DEBUG] No instances found")
            
            return set()
        else:
            # Binary rules: return a set of encoded pairs
            return self.kg.get_relation_pairs(head_relation)
    
    def _get_body_instances(self, rule_info: Dict) -> Set:
        """Get a collection of body instances - Use a unified abbreviation format for processing"""
        # Check if it iscomplex rule
        if rule_info.get('is_complex', False):
            debug(f"  [DEBUG] ProcessComplex Ruleofbody instances")
            return self._get_complex_body_instances(rule_info)
        
        variable_count = rule_info.get('variable_count', 0)
        
        if variable_count == 1:
            # unary rule
            body_relations = rule_info.get('body_relations', [])
            body_constant = rule_info.get('body_constant')
            
            # U0The rule shouldn't have gotten here (already incalculate_rule_support_joinprocessing)
            if not body_relations:
                debug(f"  [WARNING] U0Rules should not call_get_body_instances")
                return set()
            
            # Connect allbodyrelationship
            if len(body_relations) == 1:
                connected_relation = body_relations[0]
            else:
                connected_relation = body_relations[0]
                for i in range(1, len(body_relations)):
                    connected_relation = self.join_relations(connected_relation, body_relations[i])
            
            debug(f"  [DEBUG] Connected relation: {connected_relation}")
            debug(f"  [DEBUG] Body constant: {body_constant}")
            
            # Get instance
            if body_constant is not None:
                body_constant_id = self.kg.get_entity_id(body_constant)
                if body_constant_id is None:
                    debug(f"  [DEBUG] Body constant not found: {body_constant}")
                    return set()
                
                inverse_connected_relation = self.kg.get_inverse_relation(connected_relation)
                debug(f"  [DEBUG] Using inverse relation: {inverse_connected_relation}")
                
                if inverse_connected_relation in self.kg.r2h2t and body_constant_id in self.kg.r2h2t[inverse_connected_relation]:
                    result = set(self.kg.r2h2t[inverse_connected_relation][body_constant_id])
                    debug(f"  [DEBUG] Body instances: {len(result)}")
                    return result
                else:
                    debug(f"  [DEBUG] No instances found")
                    return set()
            else:
                # No constants: get all of the entire relationheadEntity
                if connected_relation in self.kg.r2h2t:
                    result = set(self.kg.r2h2t[connected_relation].keys())
                    debug(f"  [DEBUG] Body instances (all heads): {len(result)}")
                    return result
                else:
                    debug(f"  [DEBUG] No instances found")
                    return set()
        else:
            # Binary rules: return a set of encoded pairs
            body_relations = rule_info.get('body_relations', [])
            
            # U0The rule shouldn't have gotten here (already incalculate_rule_support_joinprocessing)
            if not body_relations:
                debug(f"  [WARNING] U0Rules should not call_get_body_instances")
                return set()
            
            return self.get_binary_instances_join(body_relations)
    
    def _get_complex_body_instances(self, rule_info: Dict) -> Set:
        """
        getComplex Ruleofbody instances
        
        forcomplex rule, bodyThere are multiplebranches (use&&separated):
        - Calculate each separatelybranchofinstances
        - body instances = branch1_instances ∩ branch2_instances ∩ ...
        
        Args:
            rule_info: Rule information dictionary, includingbrancheslist
            
        Returns:
            body instancesCollection (unary rule returns entityIDset, binary rules return the encoded pair set)
        """
        branches = rule_info.get('branches', [])
        is_unary = rule_info.get('is_unary', False)
        
        debug(f"  [DEBUG] CalculateComplex Ruleofbody instances")
        debug(f"  [DEBUG] BranchQuantity: {len(branches)}")
        debug(f"  [DEBUG] Rule type: {'one yuan' if is_unary else 'Binary'}")
        
        if not branches:
            debug(f"  [ERROR] No branches found in complex rule!")
            return set()
        
        # store eachbranchofinstances
        branch_instances_list = []
        
        # Calculate eachbranchofinstances
        for i, branch_info in enumerate(branches):
            debug(f"  [DEBUG] CalculateBranch {i+1}: {branch_info['branch_text']}")
            
            branch_relations = branch_info['relations']
            branch_constant = branch_info['constant']
            
            # Calculated based on rule typebranchofinstances
            if is_unary:
                # unary rule
                branch_instances = self._get_branch_instances_unary(branch_relations, branch_constant)
            else:
                # binary rule
                branch_instances = self._get_branch_instances_binary(branch_relations)
            
            debug(f"  [DEBUG] Branch {i+1} instances: {len(branch_instances)}")
            if branch_instances:
                sample = list(branch_instances)[:3]
                if is_unary:
                    debug(f"  [DEBUG] Branch {i+1} sample: {[self.kg.get_entity_str(e) for e in sample]}")
                else:
                    debug(f"  [DEBUG] Branch {i+1} sample: {[(self.kg.get_entity_str(h), self.kg.get_entity_str(t)) for h, t in [self.kg.decode_pair(p) for p in sample]]}")
            
            branch_instances_list.append(branch_instances)
        
        # Count allbranchesintersection of
        if not branch_instances_list:
            return set()
        
        result = branch_instances_list[0]
        for i in range(1, len(branch_instances_list)):
            result = result.intersection(branch_instances_list[i])
            debug(f"  [DEBUG] before{i+1}abranchesintersection of: {len(result)} instances")
        
        debug(f"  [DEBUG] Complex Ruleeventuallybody instances: {len(result)}")
        return result
    
    def _get_branch_instances_unary(self, relations: List[str], constant: Optional[str]) -> Set[int]:
        """
        Compute a single unary rulebranchofinstances
        
        Args:
            relations: Relationship path list
            constant: Constants (if any)
            
        Returns:
            EntityIDcollection
        """
        if not relations:
            return set()
        
        # Connect all relationships
        if len(relations) == 1:
            connected_relation = relations[0]
        else:
            connected_relation = relations[0]
            for i in range(1, len(relations)):
                connected_relation = self.join_relations(connected_relation, relations[i])
        
        debug(f"    [DEBUG] Branchrelationship after connection: {connected_relation}")
        debug(f"    [DEBUG] Branchconstant: {constant}")
        
        # Get instance
        if constant is not None:
            constant_id = self.kg.get_entity_id(constant)
            if constant_id is None:
                debug(f"    [DEBUG] Constant not found: {constant}")
                return set()
            
            inverse_connected_relation = self.kg.get_inverse_relation(connected_relation)
            debug(f"    [DEBUG] Use inverse relationship: {inverse_connected_relation}")
            
            if inverse_connected_relation in self.kg.r2h2t and constant_id in self.kg.r2h2t[inverse_connected_relation]:
                result = set(self.kg.r2h2t[inverse_connected_relation][constant_id])
                return result
            else:
                return set()
        else:
            # No constants: get all of the entire relationheadEntity
            if connected_relation in self.kg.r2h2t:
                result = set(self.kg.r2h2t[connected_relation].keys())
                return result
            else:
                return set()
    
    def _get_branch_instances_binary(self, relations: List[str]) -> Set[int]:
        """
        Computes a single binary rulebranchofinstances
        
        Args:
            relations: Relationship path list
            
        Returns:
            coded pair set
        """
        if not relations:
            return set()
        
        return self.get_binary_instances_join(relations)
    
    def _extract_unary_body_info(self, rule_info: Dict) -> Tuple[str, int]:
        """
        Extract from full-form unary rulesbodyConstant and variable location information
        
        For rules like:head(X,/m/05pd94v) <= body1(X,A), body2(A,/m/0m2l9)
        Need to extract constant from last atom containing constant /m/0m2l9
        
        Returns:
            (body_constant, body_variable_position)
        """
        body_atoms = rule_info.get('body_atoms', [])
        
        if not body_atoms:
            return None, 0
        
        debug(f"  [DEBUG] Extracting from body_atoms: {[(atom.get('relation'), atom.get('args')) for atom in body_atoms]}")
        
        # Find atoms containing constants
        for atom in body_atoms:
            if isinstance(atom, dict):
                args = atom.get('args', [])
            else:
                # ifbody_atomsis a list of strings, skip
                continue
            
            # Find constant (length>1parameters)
            for i, arg in enumerate(args):
                if len(arg) > 1:  # find constant
                    # Determine the location of the variable: Opposite the constant is the location of the variable
                    variable_position = 1 - i  # If the constant is in positioni, variable at position1-i
                    debug(f"  [DEBUG] Found constant {arg} at position {i}, variable at position {variable_position}")
                    return arg, variable_position
        
        # If the constant is not found, return the default value
        debug(f"  [DEBUG] No constant found in body atoms")
        return None, 0

    def calculate_rule_support_bruteforce(self, rule_info: Dict) -> Dict:
        """
        Use brute force algorithm to calculate rule support
        
        Args:
            rule_info: Rule information dictionary
        
        Returns:
            contains headSize, bodySize, support, confidence dictionary
        """
        if rule_info['variable_count'] == 1:
            return self._calculate_unary_rule_bruteforce(rule_info)
        else:
            return self._calculate_binary_rule_bruteforce(rule_info)
    
    def _calculate_unary_rule_bruteforce(self, rule_info: Dict) -> Dict:
        """Calculate unary rule support - Brute force algorithm"""
        head_atom = rule_info['head_atom']
        body_relations = rule_info.get('body_relations', [])
        
        # Calculateheadinstance collection
        head_instances = self._get_atom_instances_bruteforce(head_atom, rule_info['free_variables'])
        head_size = len(head_instances)
        
        # Check if it isU0rules (emptybody) 
        if not body_relations:
            # U0Rules:bodyis empty
            # support = headSize (allheadInstances are supported)
            # bodySize = The number of all triples of the relation
            head_relation = head_atom['relation']
            
            if head_relation.startswith('INVERSE_'):
                original_relation = head_relation[8:]
                # INVERSEtotal number of relationships = Total number of original relations
                body_size = self.kg.get_relation_instances_count(original_relation)
            else:
                body_size = self.kg.get_relation_instances_count(head_relation)
            
            support = head_size  # U0Rules:support = headSize
            confidence = support / body_size if body_size > 0 else 0
            
            debug(f"  [DEBUG] U0rules: headSize={head_size}, bodySize={body_size}, support={support}")
        else:
            # Ordinary unary rules: Yesbody
            # Calculatebodyinstance collection
            body_instances = self.get_unary_instances_bruteforce(rule_info)
            body_size = len(body_instances)
            
            # Calculate support
            support_instances = head_instances.intersection(body_instances)
            support = len(support_instances)
            confidence = support / body_size if body_size > 0 else 0
        
        return {
            'headSize': head_size,
            'bodySize': body_size,
            'support': support,
            'confidence': confidence
        }
    
    def _calculate_binary_rule_bruteforce(self, rule_info: Dict) -> Dict:
        """Calculate binary rule support - Brute force algorithm"""
        head_atom = rule_info['head_atom']
        body_relations = rule_info.get('body_relations', [])
        
        # Calculateheadinstance collection
        head_instances = self._get_atom_instances_bruteforce(head_atom, rule_info['free_variables'])
        head_size = len(head_instances)
        
        # Check if it isU0rules (emptybody) 
        if not body_relations:
            # U0Rules:bodyis empty
            # support = headSize (allheadInstances are supported)
            # bodySize = The number of all triples of the relation
            head_relation = head_atom['relation']
            
            if head_relation.startswith('INVERSE_'):
                original_relation = head_relation[8:]
                body_size = self.kg.get_relation_instances_count(original_relation)
            else:
                body_size = self.kg.get_relation_instances_count(head_relation)
            
            support = head_size  # U0Rules:support = headSize
            confidence = support / body_size if body_size > 0 else 0
            
            debug(f"  [DEBUG] U0Rules (binary): headSize={head_size}, bodySize={body_size}, support={support}")
        else:
            # Ordinary binary rules: Yesbody
            # Calculatebodyinstance collection
            body_instances = self.get_binary_instances_bruteforce(rule_info)
            body_size = len(body_instances)
            
            # Calculate support
            support_instances = head_instances.intersection(body_instances)
            support = len(support_instances)
            confidence = support / body_size if body_size > 0 else 0
        
        return {
            'headSize': head_size,
            'bodySize': body_size,
            'support': support,
            'confidence': confidence
        }
    
    def _get_atom_instances_bruteforce(self, atom: Dict, free_variables: List[str]) -> Set:
        """Get a collection of atomic instances"""
        relation = atom['relation']
        args = atom['args']
        
        if len(free_variables) == 1:
            # Unary rule: return a set of variable values
            variable = free_variables[0]
            instances = set()
            
            for head, tail in self.kg.get_relation_pairs(relation):
                if args[0] == variable and args[1] != variable:
                    if args[1] == tail:
                        instances.add(head)
                elif args[1] == variable and args[0] != variable:
                    if args[0] == head:
                        instances.add(tail)
            
            return instances
        else:
            # Binary Rule: Return (X, Y) set of pairs
            return self.kg.get_relation_pairs(relation)

    
    def find_path_instances_count(self, relation_path: List[str]) -> int:
        """
        Calculate the number of path instances using the step-by-step join algorithm
        
        Algorithm:
        1. Right to left stepwise connection relationship
        2. Each time two relations are connected, a new intermediate relation is generated and stored inr2h2t
        3. The number of instances that ultimately returns the full path
        """
        if not relation_path:
            return 0
        
        if len(relation_path) == 1:
            return self.kg.get_relation_instances_count(relation_path[0])
        
        debug(f"Calculate path example: {' * '.join(relation_path)}")
        
        # Connect step by step from right to left
        current_relation = relation_path[-1]  # rightmost relationship
        
        # Starting from the penultimate relationship, join to the left
        for i in range(len(relation_path) - 2, -1, -1):
            left_relation = relation_path[i]
            current_relation = self.join_relations(left_relation, current_relation)
        
        # Returns the number of instances of the final relationship
        return self.kg.get_relation_instances_count(current_relation)
    
    def _calculate_unary_support(self, rule_info: Dict, head_instances: Set[Tuple[str, str]], 
                               body_instances: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """Example of calculating the support of a unary rule"""
        # For unary rules, matching needs to be based on fixed entity positions
        if not rule_info['is_unary']:
            return head_instances.intersection(body_instances)
        
        fixed_entity = rule_info.get('fixed_entity')
        var_position = rule_info.get('variable_position')
        
        if not fixed_entity or not var_position:
            # If there is no fixed entity information, fall back to simple intersection
            return head_instances.intersection(body_instances)
        
        # Match based on variable position
        support_instances = set()
        
        if var_position == 'tail':
            # shaped like head(X, fixed_entity), The free variable is intaillocation
            for head, tail in head_instances:
                if tail == fixed_entity:
                    # Check if the body contains the corresponding instance
                    for b_head, b_tail in body_instances:
                        if head == b_head:  # Xvalue match
                            support_instances.add((head, tail))
                            break
        else:
            # shaped like head(fixed_entity, X), The free variable is inheadlocation
            for head, tail in head_instances:
                if head == fixed_entity:
                    for b_head, b_tail in body_instances:
                        if tail == b_tail:  # Xvalue match
                            support_instances.add((head, tail))
                            break
        
        return support_instances
    
    def _find_unary_body_instances_simplified(self, rule_info: Dict, body_relations: List[str]) -> Set[Tuple[str, str]]:
        """
        Unary rule body instance for handling abbreviated format
        
        For example:/rel(/m/123) <= /rel1*/rel2(/m/456)
        means:/rel(X,/m/123) <= /rel1(X,A), /rel2(A,/m/456)
        """
        if not body_relations:
            return set()
        
        # For abbreviated unary rules, it is necessary to build composite relationships and filter fixed entities
        if len(body_relations) == 1:
            # single relationship
            relation = body_relations[0]
            # Extract possible fixed entities from relationship names
            if '(' in relation and ')' in relation:
                # Such as:/rel2(/m/456)
                base_relation = relation.split('(')[0]
                fixed_entity = relation.split('(')[1].split(')')[0]
                
                # Get all instances of this relationship, filtering those that contain fixed entities
                all_instances = self.kg.get_relation_pairs(base_relation)
                filtered_instances = set()
                
                if rule_info['variable_position'] == 'tail':
                    # Xintailposition, the fixed entity should be inheadlocation
                    for head, tail in all_instances:
                        if tail == fixed_entity:
                            filtered_instances.add((head, tail))
                else:
                    # Xinheadposition, the fixed entity should be intaillocation
                    for head, tail in all_instances:
                        if head == fixed_entity:
                            filtered_instances.add((head, tail))
                
                return filtered_instances
            else:
                # There is no fixed entity, all instances are returned
                return self.kg.get_relation_pairs(relation)
        else:
            # Joining multiple relationships: building composite relationships
            composite_relation = body_relations[0]
            for rel in body_relations[1:]:
                composite_relation = self.join_relations(composite_relation, rel)
            
            return self.kg.get_relation_pairs(composite_relation)
    
    def _find_binary_body_instances_simple(self, body_relations: List[str]) -> Set[Tuple[str, str]]:
        """
        Handling body instances of binary rules (simplified version)
        """
        if not body_relations:
            return set()
        
        if len(body_relations) == 1:
            return self.kg.get_relation_pairs(body_relations[0])
        
        # The connection of multiple relationships
        current_pairs = self.kg.get_relation_pairs(body_relations[0])
        
        for relation in body_relations[1:]:
            next_pairs = self.kg.get_relation_pairs(relation)
            new_pairs = set()
            
            # Simplified connection: find pairs that can be connected
            for (h1, t1) in current_pairs:
                for (h2, t2) in next_pairs:
                    if t1 == h2:  # Can connect
                        new_pairs.add((h1, t2))
            
            current_pairs = new_pairs
        
        return current_pairs
    
    def _find_body_instances_by_variables(self, rule_info: Dict) -> Set[Tuple[str, str]]:
        """
        Accurately calculate body instances based on variable binding information
        
        Example of a unary rule:
        head(X,/m/02cg41) <= body1(X,/m/05pd94v)
        head(X,/m/05pd94v) <= body1(X,A), body2(A,/m/0m2l9)
        
        Binary rule example:
        head(X,Y) <= body1(X,A), body2(Y,A)
        head(X,Y) <= body1(X,A), body2(A,B), body3(Y,B)
        """
        body_atoms = rule_info['body_atoms']
        free_vars = rule_info['free_variables']
        
        if not body_atoms:
            return set()
        
        # Get an instance for each body atom
        atom_instances = {}
        for i, atom in enumerate(body_atoms):
            relation = RuleParser._extract_relation_from_atom(atom)
            variables = RuleParser._extract_variables(atom)
            atom_instances[i] = {
                'relation': relation,
                'variables': variables,
                'instances': self.kg.get_relation_pairs(relation)
            }
        
        debug(f"  Number of atoms in the body: {len(atom_instances)}")
        
        # Generate all possible variable binding combinations
        result_instances = set()
        
        if len(free_vars) == 1:
            # Unary rule: there is only one free variable
            free_var = list(free_vars)[0]
            free_var_values = self._solve_unary_rule(atom_instances, free_var)
            
            # For unary rules, instances need to be constructed based on the pattern of the header
            # Get the header constant from the rule information
            head_variables = rule_info.get('head_variables', [])
            head_constants = []
            
            for var in head_variables:
                if var.startswith('/m/'):  # This is a constant entity
                    head_constants.append(var)
                elif var != free_var:  # This may also be a constant
                    head_constants.append(var)
            
            # Construct header instance
            if len(head_variables) == 2:
                # Two-parameter header, such as head(X, constant) or head(constant, X)
                for value in free_var_values:
                    if head_variables[0] == free_var:
                        # Free variables come first
                        if len(head_constants) > 0:
                            result_instances.add((value, head_constants[0]))
                    elif head_variables[1] == free_var:
                        # free variable in second place
                        if len(head_constants) > 0:
                            result_instances.add((head_constants[0], value))
            
        else:
            # Binary Rule: There are two free variables
            result_instances = self._solve_binary_rule(atom_instances, free_vars)
        
        return result_instances
    
    def _solve_unary_rule(self, atom_instances: Dict, free_var: str) -> Set[str]:
        """
        Solve the unary rule
        
        For example:head(X,/m/02cg41) <= body1(X,/m/05pd94v)
        Need to find all satisfactionbody1ofXvalue
        
        Returns: a collection of free variable values that meet the conditions
        """
        # This needs to be implemented according to the specific atomic mode
        # For a unary rule, we need to find all possible values of the free variable
        result_values = set()
        
        if not atom_instances:
            return result_values
        
        # Starting from the first atom, collect the possible values of the free variables
        first_atom = atom_instances[0]
        first_instances = first_atom['instances']
        first_vars = first_atom['variables']
        
        # Find the position of the free variable in the first atom
        free_var_position = None
        for i, var in enumerate(first_vars):
            if var == free_var:
                free_var_position = i
                break
        
        if free_var_position is not None:
            # Collect all possible values of a free variable
            for head, tail in first_instances:
                if free_var_position == 0:
                    result_values.add(head)
                elif free_var_position == 1:
                    result_values.add(tail)
        
        # If there are multiple atoms, intersection operation is required
        for i in range(1, len(atom_instances)):
            atom = atom_instances[i]
            atom_instances_set = atom['instances']
            atom_vars = atom['variables']
            
            # Find the position of the free variable in the current atom
            free_var_position = None
            for j, var in enumerate(atom_vars):
                if var == free_var:
                    free_var_position = j
                    break
            
            if free_var_position is not None:
                # Collect possible values of free variables in the current atom
                current_values = set()
                for head, tail in atom_instances_set:
                    if free_var_position == 0:
                        current_values.add(head)
                    elif free_var_position == 1:
                        current_values.add(tail)
                
                # Intersect with previous results
                result_values = result_values.intersection(current_values)
        
        return result_values
    
    def _solve_binary_rule(self, atom_instances: Dict, free_vars: Set[str]) -> Set[Tuple[str, str]]:
        """
        Solving Binary Rules
        
        For example:head(X,Y) <= body1(X,A), body2(Y,A)
        Need to find all those that meet the connection conditions(X,Y)Yes
        """
        if len(atom_instances) == 1:
            # only one body atom
            atom = atom_instances[0]
            return atom['instances']
        
        # Multiple body atoms: need to be connected via constraint variables
        result = set()
        
        # Simplified implementation: assume it is the connection of two atoms
        if len(atom_instances) == 2:
            atom1 = atom_instances[0]
            atom2 = atom_instances[1]
            
            instances1 = atom1['instances']
            instances2 = atom2['instances']
            vars1 = atom1['variables']
            vars2 = atom2['variables']
            
            # Find connection variables (common constraint variables)
            common_vars = set(vars1) & set(vars2) - free_vars
            
            if common_vars:
                # With connection variables: matching by connection variables
                for h1, t1 in instances1:
                    for h2, t2 in instances2:
                        # Simplified connection logic
                        if self._can_join(vars1, vars2, h1, t1, h2, t2, common_vars):
                            # Construct the result pair (needs to be determined based on the position of the free variable)
                            x_val, y_val = self._extract_free_variable_values(
                                vars1, vars2, h1, t1, h2, t2, free_vars
                            )
                            if x_val and y_val:
                                result.add((x_val, y_val))
        
        return result
    
    def _can_join(self, vars1: List[str], vars2: List[str], h1: str, t1: str, 
                  h2: str, t2: str, common_vars: Set[str]) -> bool:
        """Check if two atomic instances can be connected by a constraint variable"""
        # Simplified implementation: check if the tail matches
        return t1 == h2 or t1 == t2 or h1 == h2 or h1 == t2
    
    def _extract_free_variable_values(self, vars1: List[str], vars2: List[str], 
                                    h1: str, t1: str, h2: str, t2: str, 
                                    free_vars: Set[str]) -> Tuple[str, str]:
        """Extract the value of a free variable from a connected atomic instance"""
        # Simplified implementation: assumptionsXIn the first position of the first atom,Yin the first or second position of the second atom
        x_val = h1 if len(vars1) > 0 and vars1[0] in free_vars else None
        y_val = h2 if len(vars2) > 0 and vars2[0] in free_vars else (
            t2 if len(vars2) > 1 and vars2[1] in free_vars else None
        )
        return x_val, y_val
    
    def _find_binary_body_instances_simple(self, body_relations: List[str]) -> Set[Tuple[str, str]]:
        """
        Simplified binary rule body instance calculation (for shorthand format)
        Assuming chain connection mode
        """
        if not body_relations:
            return set()
        
        if len(body_relations) == 1:
            return self.kg.get_relation_pairs(body_relations[0])
        
        # For multiple relationships, perform chain connections
        current_pairs = self.kg.get_relation_pairs(body_relations[0])
        
        for relation in body_relations[1:]:
            next_pairs = self.kg.get_relation_pairs(relation)
            new_pairs = set()
            
            # Simplified connection: find pairs that can be connected
            for (h1, t1) in current_pairs:
                for (h2, t2) in next_pairs:
                    if t1 == h2:  # Can connect
                        new_pairs.add((h1, t2))
            
            current_pairs = new_pairs
        
        return current_pairs
    
    def get_path_instances(self, relation_path: List[str]) -> Set[int]:
        """Get a collection of actual instances of a path (encoded pairing)"""
        if not relation_path:
            return set()
        
        if len(relation_path) == 1:
            return self.kg.get_relation_pairs(relation_path[0])
        
        # Construct path relationship name, withfind_path_instances_countlogical consistency in
        current_relation = relation_path[-1]  # rightmost relationship
        
        # Starting from the penultimate relation, join to the left (reusing stored intermediate results)
        for i in range(len(relation_path) - 2, -1, -1):
            left_relation = relation_path[i]
            current_relation = f"{left_relation}*{current_relation}"
        
        # If the relationship is already inr2h2t, directly obtain the instance
        if current_relation in self.kg.r2h2t:
            return self.kg.get_relation_pairs(current_relation)
        else:
            # If it doesn't exist, compute it first (this shouldn't happen becausefind_path_instances_countIt should have been calculated)
            self.find_path_instances_count(relation_path)
            return self.kg.get_relation_pairs(current_relation)


def load_dataset(filepath: str) -> KnowledgeGraph:
    """Load the data set into the knowledge graph"""
    kg = KnowledgeGraph()
    
    debug(f"Loading dataset: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file does not exist: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                debug(f"Warning: Chapter{line_num}Row format error: {line}")
                continue
            
            head, relation, tail = parts
            kg.add_triple(head, relation, tail)
            
            if line_num % 50000 == 0:
                debug(f"Loaded {line_num:,} triples...")
    
    debug(f"Data set loading completed:")
    debug(f"  Number of triples: {len(kg.triples):,}")
    debug(f"  Number of entities: {kg._next_entity_id:,}")
    debug(f"  Original number of relations: {len([r for r in kg.relations if not r.startswith('INVERSE_')]):,}")
    debug(f"  Total number of relationships (including inverse relationships): {len(kg.relations):,}")
    
    return kg


def analyze_rule_from_string(rule_str: str, kg: KnowledgeGraph) -> Dict:
    """
    Analyze rule support from rule string
    
    Args:
        rule_str: rule string
        kg: Knowledge graph
        
    Returns:
        A dictionary containing the results of both algorithms
    """
    try:
        # parsing rules
        head_relation, body_relations, variable_count, rule_info = RuleParser.parse_rule(rule_str)
        
        debug(f"\nAnalysis rules: {rule_str}")
        debug(f"normalization rules: {rule_info.get('normalized_rule', rule_str)}")
        debug(f"Rule type: {'one yuan' if variable_count == 1 else 'Binary'}")
        debug(f"head relationship: {head_relation}")
        debug(f"physical relationship: {body_relations}")
        
        # Create a calculator
        calculator = RuleSupportCalculator(kg)
        
        # Join algorithm and brute force algorithm
        join_result = None
        bruteforce_result = None
        
        try:
            # Call the join algorithm
            debug("=== Join algorithm ===")
            join_result = calculator.calculate_rule_support_join(rule_info)
            
            # Call brute force algorithm
            # debug("=== Brute force algorithm ===")
            # bruteforce_result = calculator.calculate_rule_support_bruteforce(rule_info)
            
        except Exception as e:
            debug(f"Calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # If the calculation fails, at least basic header information is returned
            head_size = kg.get_relation_instances_count(head_relation)
            join_result = {'headSize': head_size, 'bodySize': 0, 'support': 0, 'confidence': 0.0}
            bruteforce_result = {'headSize': head_size, 'bodySize': 0, 'support': 0, 'confidence': 0.0}
        
        return {
            'rule': rule_str,
            'head_relation': head_relation,
            'body_relations': body_relations,
            'variable_count': variable_count,
            'join_result': join_result,
            'bruteforce_result': bruteforce_result
        }
        
    except Exception as e:
        debug(f"Rule analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Dataset path (relative to the path of the current script)
    dataset_path = "data/FB15k-237/train.txt"
    
    # Example of rules to analyze
    test_rules = [
        "/award/award_category/winners./award/award_honor/ceremony(X,/m/01xqqp) <= /award/award_category/winners./award/award_honor/ceremony(X,A), /award/award_category/winners./award/award_honor/ceremony(/m/0257w4,A)",
        # Example of a unary rule (only one free variable)
        "/award/award_category/winners./award/award_honor/ceremony(X,/m/02cg41) <= /award/award_category/winners./award/award_honor/ceremony(X,/m/05pd94v)",
        
        "/award/award_category/winners./award/award_honor/ceremony(X,/m/05pd94v) <= /award/award_category/winners./award/award_honor/ceremony(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(A,/m/0m2l9)",
        
        # Unary rule abbreviation format
        "/award/award_category/winners./award/award_honor/ceremony(/m/05pd94v) <= /award/award_category/winners./award/award_honor/ceremony*/award/award_ceremony/awards_presented./award/award_honor/award_winner(/m/0m2l9)",
        
        "INVERSE_/award/award_category/winners./award/award_honor/ceremony(/m/0gs9p) <= INVERSE_/award/award_category/winners./award/award_honor/ceremony*/award/award_category/nominees./award/award_nomination/nominated_for(/m/0j8f09z)",
        
        # Binary rule example (with two free variablesX,Y) 
        "/award/award_category/winners./award/award_honor/ceremony(X,Y) <= /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,A)",
        
        "/award/award_category/winners./award/award_honor/ceremony(X,Y) <= /award/award_category/winners./award/award_honor/ceremony(X,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(A,B), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,B)",
        
        # Binary rules for abbreviated format (traditional format)
        "/award/award_category/winners./award/award_honor/ceremony <= /award/award_category/winners./award/award_honor/ceremony*/award/award_ceremony/awards_presented./award/award_honor/award_winner*INVERSE_/award/award_ceremony/awards_presented./award/award_honor/award_winner",

        "/award/award_category/winners./award/award_honor/ceremony(/m/0gs96,X) <= /award/award_category/winners./award/award_honor/ceremony(A,X), /award/award_category/nominees./award/award_nomination/nominated_for(A,/m/02r79_h)",
        "/award/award_category/winners./award/award_honor/ceremony(/m/0f4x7,X) <= /award/award_category/winners./award/award_honor/ceremony(A,X), /award/award_nominee/award_nominations./award/award_nomination/award(/m/02_fj,A)",
        "/award/award_category/winners./award/award_honor/ceremony(/m/01ck6v,Y) <= /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,A)",
        "/award/award_category/winners./award/award_honor/ceremony(X,/m/07z31v) <= /award/award_nominee/award_nominations./award/award_nomination/award(A,X)"
    ]

    test_rules = [
        "/award/award_category/winners./award/award_honor/ceremony(X,Y) <= /award/award_category/category_of(X,A), /time/event/instance_of_recurring_event(Y,A)",
        "/award/award_category/winners./award/award_honor/ceremony(X,Y) <= /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_winner/awards_won./award/award_honor/award_winner(B,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,B)",

        "/film/film/release_date_s./film/film_regional_release_date/film_release_region(X,/m/0b90_r) <= /film/film/release_date_s./film/film_regional_release_date/film_release_region(X,/m/07ylj)",

        "INVERSE_/music/genre/artists(/m/06by7) <= INVERSE_/music/performance_role/regular_performances./music/group_membership/group(*)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency <= /education/university/local_tuition./measurement_unit/dated_money_value/currency * INVERSE_/education/university/local_tuition./measurement_unit/dated_money_value/currency * /education/university/local_tuition./measurement_unit/dated_money_value/currency"
    
        "/award/award_category/winners./award/award_honor/ceremony(X,Y) <= /award/award_category/winners./award/award_honor/award_winner(X,A), /award/award_winner/awards_won./award/award_honor/award_winner(B,A), /award/award_ceremony/awards_presented./award/award_honor/award_winner(Y,B)",

        "/film/film/release_date_s./film/film_regional_release_date/film_release_region(X,/m/0b90_r) <= /film/film/release_date_s./film/film_regional_release_date/film_release_region(X,/m/07ylj)",

        "INVERSE_/music/genre/artists(/m/06by7) <= INVERSE_/music/performance_role/regular_performances./music/group_membership/group(*)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency <= /education/university/local_tuition./measurement_unit/dated_money_value/currency * INVERSE_/education/university/local_tuition./measurement_unit/dated_money_value/currency * /education/university/local_tuition./measurement_unit/dated_money_value/currency"
    ]
    
    # User providedComplex Rulestest case
    test_rules = [
        # typicalComplex Binary rule (M3 binary rule) : 
        # bodyYes3abranch (by2a&&separated)
        "/location/country/form_of_government(X,Y) <= /government/politician/government_positions_held./government/government_position_held/jurisdiction_of_office(A,X), /people/person/nationality(A,B), /location/country/form_of_government(B,Y)&&/location/country/form_of_government(X,A), /location/country/form_of_government(B,A), /location/country/form_of_government(B,Y)&&/location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency(X,A), /location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency(B,A), /location/country/form_of_government(B,Y)",
        
        # typicalComplex Unary rule (M2 unary rule) : 
        # bodyYes2abranch (by1a&&separated)
        "INVERSE_/government/legislative_session/members./government/government_position_held/legislative_sessions(/m/01gsvb) <= /government/legislative_session/members./government/government_position_held/district_represented(/m/05kkh)&&/government/legislative_session/members./government/government_position_held/district_represented(/m/07_f2)"
    ]

    # The followingrulesTheoreticallysuppIt should all be0, because currency They are all many-to-one relationships, so the second half INVERSE_currency*currency There can be no examples
    test_rules4 = {
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency(B,A), /location/statistical_region/gdp_real./measurement_unit/adjusted_money_value/adjustment_currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /business/business_operation/operating_income./measurement_unit/dated_money_value/currency(B,A), /organization/endowed_organization/endowment./measurement_unit/dated_money_value/currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /education/university/domestic_tuition./measurement_unit/dated_money_value/currency(B,A), /business/business_operation/operating_income./measurement_unit/dated_money_value/currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency(B,A), /location/statistical_region/gni_per_capita_in_ppp_dollars./measurement_unit/dated_money_value/currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /location/statistical_region/gdp_nominal./measurement_unit/dated_money_value/currency(B,A), /location/statistical_region/gdp_real./measurement_unit/adjusted_money_value/adjustment_currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /business/business_operation/assets./measurement_unit/dated_money_value/currency(B,A), /education/university/domestic_tuition./measurement_unit/dated_money_value/currency(B,Y)",
        "/education/university/domestic_tuition./measurement_unit/dated_money_value/currency(X,Y) <= /education/university/local_tuition./measurement_unit/dated_money_value/currency(X,A), /education/university/domestic_tuition./measurement_unit/dated_money_value/currency(B,A), /business/business_operation/revenue./measurement_unit/dated_money_value/currency(B,Y)",
        }
    
    # Note that the following are calculated separatelyrule, discoverbodySize != 0, This is a serious problem
    # test_rules = ["INVERSE_/location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency <=  INVERSE_/education/university/local_tuition./measurement_unit/dated_money_value/currency*/education/university/local_tuition./measurement_unit/dated_money_value/currency*INVERSE_/location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency"]

    # test_rules = ["/location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency <=  /location/statistical_region/rent50_2./measurement_unit/dated_money_value/currency*INVERSE_/education/university/local_tuition./measurement_unit/dated_money_value/currency*/education/university/local_tuition./measurement_unit/dated_money_value/currency"]

    if len(sys.argv) > 1:
        test_rules = sys.argv[1:]

    try:
        # Load dataset
        kg = load_dataset(dataset_path)
        
        # Analyze each rule
        results = []
        for rule_str in test_rules:
            result = analyze_rule_from_string(rule_str, kg)
            if result:
                results.append(result)
        
        # Output comprehensive results
        debug(f"\n{'='*100}")
        debug("Comprehensive analysis results")
        debug(f"{'='*100}")
        
        for i, result in enumerate(results, 1):
            debug(f"\nrules {i}: {result['rule']}")
            
            if result['join_result']:
                debug(f"  Join algorithm: {result['join_result']}")
            
            if result['bruteforce_result']:
                debug(f"  Brute force algorithm: {result['bruteforce_result']}")
        
        debug(f"\n{'='*100}")
        
    except Exception as e:
        debug(f"Error: {e}")
        import traceback
        traceback.print_exc()

