import json
import os

DATA = "/Users/a./Documents/RuleDep/reports/0407/case_study"
JSON_FILE = os.path.join(DATA, "FB15k-237_countries_spoken_in_m_0d05q4_0112_graph.json")

with open(JSON_FILE) as f:
    graph = json.load(f)

# === 1. Build rule lookup ===
rule_map = {}
for r in graph["rule_dependency_layer"]["rules"]:
    rule_map[r["rule_id"]] = {
        "rule_id": r["rule_id"],
        "rule_type": r["rule_type"],
        "rule_text": r["rule_text"],
        "stage1_weight": r["weight"]["stage1"]["effective_weight"],
        "stage2_weight": r["weight"]["stage2"]["effective_weight"],
    }

# === 2. Extract candidate data ===
candidates = {}
for ce in graph["candidate_explanations"]:
    eid = ce["entity_id"]
    label = ce["entity_label"]
    candidates[eid] = {
        "label": label,
        "role": ce["role"],
        "rank_before": ce["rank"]["before_dep"],
        "rank_after": ce["rank"]["after_dep"],
        "score_before": ce["rank_score"]["before_dep"],
        "score_after": ce["rank_score"]["after_dep"],
        "total_before": ce["score"]["before_dep"]["total_linear"],
        "total_after": ce["score"]["after_dep"]["total_linear"],
        "rule_total": ce["score"]["before_dep"]["rule_total"],
        "dependency_total": ce["score"]["after_dep"]["dependency_total"],
        "intercept": ce["score"]["before_dep"]["intercept"],
        "active_rules_sample": ce["active_rules_sample"],
        "active_rule_count": ce["active_rule_count"],
        "active_dep_count": ce["active_dependency_count"],
        "top_deps": ce.get("top_dependency_details", []),
        "rule_components_by_type": ce["score"]["before_dep"]["rule_components"]["by_rule_type"],
        "dep_components_by_type": ce["score"]["after_dep"]["dependency_components"]["by_dependency_type"],
        "dep_components_by_sign": ce["score"]["after_dep"]["dependency_components"]["by_sign"],
    }

# === 3. Print summary ===
print("=" * 80)
print("CASE STUDY: countries_spoken_in(Iraq)")
print("=" * 80)

for eid, c in candidates.items():
    print(f"\n{'─'*70}")
    print(f"  {c['label']}  ({c['role']})")
    print(f"{'─'*70}")
    print(f"  Rank:       #{int(c['rank_before'])} → #{int(c['rank_after'])}")
    print(f"  Score:      {c['score_before']:.6f} → {c['score_after']:.6f}  (Δ = {c['score_after']-c['score_before']:+.6f})")
    print(f"  total_linear: {c['total_before']:.6f} → {c['total_after']:.6f}  (Δ = {c['total_after']-c['total_before']:+.6f})")
    print(f"  rule_total: {c['rule_total']:.6f}  (types: B={c['rule_components_by_type'].get('B',0):.4f}, U={c['rule_components_by_type'].get('U',0):.4f})")
    print(f"  dependency_total: {c['dependency_total']:+.6f}")
    dep_signs = c['dep_components_by_sign']
    if dep_signs:
        print(f"    positive: {dep_signs.get('positive', 0):+.6f}")
        print(f"    negative: {dep_signs.get('negative', 0):+.6f}")
    dep_types = c['dep_components_by_type']
    if dep_types:
        for dt, dv in dep_types.items():
            print(f"    {dt}: {dv:+.6f}")
    print(f"  Active rules: {c['active_rule_count']}, Active deps: {c['active_dep_count']}")

    # Top 10 dependency details
    if c['top_deps']:
        print(f"\n  TOP DEPENDENCY INTERACTIONS (by contribution):")
        for i, dep in enumerate(c['top_deps'][:10]):
            left_id = dep['left_rule_id']
            right_id = dep['right_rule_id']
            left_text = dep.get('left_rule_text', '?')
            right_text = dep.get('right_rule_text', '?')
            contrib = dep['estimated_candidate_contribution']
            dep_type = dep['dependency_type']
            sign = dep['sign']
            frac = dep['estimated_fraction_of_dependency_total']
            print(f"    [{i+1}] {dep['dep_id']}  ({dep_type}, {sign}, {frac*100:.1f}% of dep total)")
            print(f"        contrib: {contrib:+.6f}")
            print(f"        L: {left_id}  → {left_text[:90]}")
            print(f"        R: {right_id}  → {right_text[:90]}")

print("\n" + "=" * 80)

# === 4. Focus on English and Persian only ===
print("\n\n" + "=" * 80)
print("FOCUS: English vs Persian — DEPENDENCY IMPACT ANALYSIS")
print("=" * 80)

english = candidates["/m/02h40lc"]
persian = candidates["/m/032f6"]

print(f"\n{'─'*70}")
print(f"  BEFORE dependency correction:")
print(f"    English:  score={english['score_before']:.6f} (rank #1)")
print(f"    Persian:  score={persian['score_before']:.6f} (rank #2)")
print(f"    English leads by {english['score_before'] - persian['score_before']:.6f}")

print(f"\n{'─'*70}")
print(f"  AFTER dependency correction:")
print(f"    English:  score={english['score_after']:.6f}  (dependency Δ = {english['dependency_total']:+.6f})")
print(f"    Persian:  score={persian['score_after']:.6f}  (dependency Δ = {persian['dependency_total']:+.6f})")
print(f"    Persian leads by {persian['score_after'] - english['score_after']:.6f}")

# === 5. Find the top negative dependencies for English ===
print(f"\n{'─'*70}")
print(f"  ENGLISH: Top dependency contributions that LOWER the score")
print(f"{'─'*70}")
english_neg_deps = [d for d in english['top_deps'] if d['sign'] == 'negative']
english_pos_deps = [d for d in english['top_deps'] if d['sign'] == 'positive']

print(f"\n  NEGATIVE dependencies (penalize English):")
total_neg = 0
for i, dep in enumerate(english_neg_deps[:5]):
    contrib = dep['estimated_candidate_contribution']
    total_neg += contrib
    frac = dep['estimated_fraction_of_dependency_total']
    left_id = dep['left_rule_id']
    right_id = dep['right_rule_id']
    left_text = dep.get('left_rule_text', '?')
    right_text = dep.get('right_rule_text', '?')
    print(f"    [{i+1}] {dep['dep_id']}  ({dep['dependency_type']}, {contrib:+.6f}, {frac*100:.1f}% of dep total)")
    print(f"        L: {left_id} | {left_text[:100]}")
    print(f"        R: {right_id} | {right_text[:100]}")
print(f"\n  (Top 5 negative deps account for {total_neg:.6f} of the {english['dep_components_by_sign'].get('negative', 0):.6f} total negative)")

print(f"\n  POSITIVE dependencies (boost English):")
total_pos = 0
for i, dep in enumerate(english_pos_deps[:5]):
    contrib = dep['estimated_candidate_contribution']
    total_pos += contrib
    frac = dep['estimated_fraction_of_dependency_total']
    left_id = dep['left_rule_id']
    right_id = dep['right_rule_id']
    left_text = dep.get('left_rule_text', '?')
    right_text = dep.get('right_rule_text', '?')
    print(f"    [{i+1}] {dep['dep_id']}  ({dep['dependency_type']}, {contrib:+.6f}, {frac*100:.1f}% of dep total)")
    print(f"        L: {left_id} | {left_text[:100]}")
    print(f"        R: {right_id} | {right_text[:100]}")
print(f"\n  (Top 5 positive deps account for {total_pos:.6f} of the {english['dep_components_by_sign'].get('positive', 0):.6f} total positive)")

# === 6. Find the top positive dependencies for Persian ===
print(f"\n{'─'*70}")
print(f"  PERSIAN: Top dependency contributions that RAISE the score")
print(f"{'─'*70}")
persian_pos_deps = [d for d in persian['top_deps'] if d['sign'] == 'positive']
persian_neg_deps = [d for d in persian['top_deps'] if d['sign'] == 'negative']

print(f"\n  POSITIVE dependencies (boost Persian):")
total_pos_p = 0
for i, dep in enumerate(persian_pos_deps[:5]):
    contrib = dep['estimated_candidate_contribution']
    total_pos_p += contrib
    frac = dep['estimated_fraction_of_dependency_total']
    left_id = dep['left_rule_id']
    right_id = dep['right_rule_id']
    left_text = dep.get('left_rule_text', '?')
    right_text = dep.get('right_rule_text', '?')
    print(f"    [{i+1}] {dep['dep_id']}  ({dep['dependency_type']}, {contrib:+.6f}, {frac*100:.1f}% of dep total)")
    print(f"        L: {left_id} | {left_text[:100]}")
    print(f"        R: {right_id} | {right_text[:100]}")
print(f"\n  (Top 5 positive deps account for {total_pos_p:.6f} of the {persian['dep_components_by_sign'].get('positive', 0):.6f} total positive)")

print(f"\n  NEGATIVE dependencies (penalize Persian):")
if persian_neg_deps:
    for i, dep in enumerate(persian_neg_deps[:3]):
        contrib = dep['estimated_candidate_contribution']
        frac = dep['estimated_fraction_of_dependency_total']
        left_id = dep['left_rule_id']
        right_id = dep['right_rule_id']
        left_text = dep.get('left_rule_text', '?')
        right_text = dep.get('right_rule_text', '?')
        print(f"    [{i+1}] {dep['dep_id']}  ({dep['dependency_type']}, {contrib:+.6f}, {frac*100:.1f}% of dep total)")
        print(f"        L: {left_id} | {left_text[:100]}")
        print(f"        R: {right_id} | {right_text[:100]}")
else:
    print(f"    None (all dependencies are positive)")

# === 7. Print active rules for each ===
print(f"\n{'─'*70}")
print(f"  ACTIVE RULES (samples)")
print(f"{'─'*70}")

for label, eid in [("English", "/m/02h40lc"), ("Persian", "/m/032f6")]:
    c = candidates[eid]
    print(f"\n  {label}: {c['active_rule_count']} active rules")
    for rid in c['active_rules_sample'][:15]:
        if rid in rule_map:
            r = rule_map[rid]
            w = r['stage2_weight']
            print(f"    {rid} ({r['rule_type']}, w={w:.4f}) → {r['rule_text'][:80]}")
        else:
            print(f"    {rid} (not in rule_dependency_layer)")
