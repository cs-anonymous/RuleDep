import json
import os
import re
from collections import defaultdict

DATA = "/Users/a./Documents/RuleDep/reports/0407/case_study"
JSON_FILE = os.path.join(DATA, "FB15k-237_countries_spoken_in_m_0d05q4_0112_graph.json")

with open(JSON_FILE) as f:
    graph = json.load(f)

# === Build comprehensive rule map ===
rule_map = {}
for r in graph["rule_dependency_layer"]["rules"]:
    rule_map[r["rule_id"]] = {
        "type": r["rule_type"],
        "text": r["rule_text"],
        "w1": r["weight"]["stage1"]["effective_weight"],
        "w2": r["weight"]["stage2"]["effective_weight"],
    }

# Also collect rule texts from candidate explanations (they may have rules not in rule_dependency_layer)
for ce in graph["candidate_explanations"]:
    for dep in ce.get("top_dependency_details", []):
        for side in ["left", "right"]:
            rid = dep[f"{side}_rule_id"]
            if rid not in rule_map:
                rule_map[rid] = {
                    "type": dep.get(f"{side}_rule_type", "?"),
                    "text": dep.get(f"{side}_rule_text", ""),
                    "w1": dep.get(f"{side}_rule_weight", {}).get("stage1", {}).get("effective_weight", 0),
                    "w2": dep.get(f"{side}_rule_weight", {}).get("stage2", {}).get("effective_weight", 0),
                }

def clean_rule_text(text, max_len=55):
    clean = text
    replacements = [
        ("/film/film/release_date_s./film/film_regional_release_date/film_release_region", "film_release"),
        ("/award/award_category/winners./award/award_honor/award_winner", "award_winner"),
        ("/award/award_category/nominees./award/award_nomination/nominated_for", "nominated_for"),
        ("/music/performance_role/track_performances./music/track_contribution/role", "music_role"),
        ("/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country", "olympic_country"),
        ("/people/person/languages", "person_lang"),
        ("/people/person/religion", "person_rel"),
        ("/education/educational_degree/people_with_this_degree./education/education/institution", "degree_inst"),
        ("/location/statistical_region/religions./location/religion_percentage/religion", "region_rel"),
        ("/film/film/other_crew./film/film_crew_gig/film_crew_role", "film_crew"),
        ("/film/film/music", "film_music"),
        ("/award/award_winner/awards_won./award/award_honor/award_winner", "award_win"),
        ("/award/award_nominee/award_nominations./award/award_nomination/award", "award_nom"),
        ("/award/award_nominee/award_nominations./award/award_nomination/award_nominee", "award_nomine"),
        ("/media_common/netflix_genre/titles", "netflix"),
        ("/music/record_label/artist", "label_artist"),
        ("/music/genre/artists", "genre_artists"),
        ("/music/genre/parent_genre", "parent_genre"),
        ("/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft", "draft_pick"),
        ("/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor", "tv_actor"),
        ("/location/location/adjoin_s./location/adjoining_relationship/adjoins", "adjoins"),
        ("/music/performance_role/regular_performances./music/group_membership/role", "perf_role"),
        ("/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics", "olympic_medal"),
        ("/music/performance_role/guest_performances./music/recording_contribution/performance_role", "guest_perf"),
        ("/music/artist/track_contributions./music/track_contribution/role", "artist_track"),
        ("/film/film/genre", "film_genre"),
        ("/film/film/production_companies", "film_prod"),
        ("/film/film/estimated_budget./measurement_unit/dated_money_value/currency", "film_budget"),
        ("/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/sport", "olympic_sport"),
        ("/people/person/profession", "person_prof"),
        ("/people/person/nationality", "person_nation"),
        ("/people/person/place_of_birth", "person_birth"),
        ("/award/award_category/nominees./award/award_nomination/nominated_for", "award_nom_for"),
        ("/award/award_nominee/award_nominations./award/award_nomination/award", "award_nom"),
        ("/military/military_combatant/military_conflicts./military/military_combatant_group/combatants", "military_conflict"),
    ]
    for pat, repl in replacements:
        clean = clean.replace(pat, repl)
    clean = re.sub(r'/m/\w+', '', clean)
    clean = clean.strip(' ()<=,')
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > max_len:
        clean = clean[:max_len] + "..."
    return clean

# === Get candidate info ===
candidates_info = {}
for ce in graph["candidate_explanations"]:
    eid = ce["entity_id"]
    label = ce["entity_label"]
    candidates_info[eid] = {
        "label": label,
        "active_rules": ce["active_rules_sample"],
        "active_rule_count": ce["active_rule_count"],
        "top_deps": ce.get("top_dependency_details", []),
        "rule_types": ce["score"]["before_dep"]["rule_components"]["by_rule_type"],
        "dependency_total": ce["score"]["after_dep"]["dependency_total"],
        "score_before": ce["rank_score"]["before_dep"],
        "score_after": ce["rank_score"]["after_dep"],
        "rank_before": int(ce["rank"]["before_dep"]),
        "rank_after": int(ce["rank"]["after_dep"]),
    }

english = candidates_info["/m/02h40lc"]
persian = candidates_info["/m/032f6"]

for name, cand in [("English", english), ("Persian", persian)]:
    print("=" * 100)
    print(f"  {name} ({cand['label']})   rank #{cand['rank_before']} → #{cand['rank_after']}")
    print(f"  score: {cand['score_before']:.4f} → {cand['score_after']:.4f}   dependency Δ = {cand['dependency_total']:+.4f}")
    print("=" * 100)
    print()

    # === Part 1: Top 10 rules by weight ===
    active = cand["active_rules"]
    rule_entries = []
    for rid in active:
        if rid in rule_map:
            r = rule_map[rid]
            rtext = clean_rule_text(r["text"])
            rule_entries.append((rid, r["type"], r["w1"], r["w2"], rtext))

    rule_entries.sort(key=lambda x: x[2], reverse=True)
    top10 = rule_entries[:10]

    rule_total = cand["rule_types"].get("B", 0) + cand["rule_types"].get("U", 0)

    print(f"  [A] TOP-10 ACTIVE RULES (by stage-1 weight)")
    print(f"  {'#':>3}  {'Rule ID':<12} {'T':<2} {'w(s1)':>7} {'w(s2)':>7} {'%of_total':>8}  Rule")
    print(f"  {'─'*100}")
    for i, (rid, rtype, w1, w2, rtext) in enumerate(top10):
        pct = w1 / rule_total * 100 if rule_total > 0 else 0
        print(f"  {i+1:3d}  {rid:<12} {rtype:<2} {w1:>7.4f} {w2:>7.4f} {pct:>7.1f}%  {rtext}")
    total_w_top10 = sum(r[2] for r in top10)
    print(f"\n  → Top-10 weight sum = {total_w_top10:.4f} / {rule_total:.4f} = {total_w_top10/rule_total*100:.1f}%")
    print()

    # === Part 2: Top dependency pairs ===
    dep_sign_filter = "negative" if name == "English" else "positive"
    relevant_deps = [d for d in cand["top_deps"] if d["sign"] == dep_sign_filter]
    if not relevant_deps:
        relevant_deps = cand["top_deps"]
    top5_deps = relevant_deps[:5]

    print(f"  [B] TOP-5 DEPENDENCY PAIRS ({dep_sign_filter}, driving the Δ)")
    print(f"  {'#':>3}  {'Dep ID':<8} {'Type':<6} {'Contrib':>9}  Left Rule                              Right Rule")
    print(f"  {'─'*100}")
    for i, dep in enumerate(top5_deps):
        lid = dep["left_rule_id"]
        rid = dep["right_rule_id"]
        contrib = dep["estimated_candidate_contribution"]
        dtype = dep["dependency_type"]
        ltext = clean_rule_text(dep.get("left_rule_text", ""))
        rtext = clean_rule_text(dep.get("right_rule_text", ""))
        print(f"  {i+1:3d}  {dep['dep_id']:<8} {dtype:<6} {contrib:>+9.4f}  {ltext:<40} {rtext}")
    print()

    # === Part 3: Redundancy/Synergy analysis among top-10 rules ===
    top10_ids = set(r[0] for r in top10)
    dep_rule_ids = set()
    for dep in cand["top_deps"]:
        dep_rule_ids.add(dep["left_rule_id"])
        dep_rule_ids.add(dep["right_rule_id"])

    print(f"  [C] REDUNDANCY CHECK: top-10 rule interactions")
    overlap = top10_ids & dep_rule_ids

    # Find specific pairs
    direct_pairs = []
    for dep in cand["top_deps"]:
        lid = dep["left_rule_id"]
        rid = dep["right_rule_id"]
        if lid in top10_ids and rid in top10_ids:
            direct_pairs.append((lid, rid, dep["dependency_type"], dep["sign"], dep["estimated_candidate_contribution"]))

    # Also check: top-10 rules paired with non-top-10 rules
    cross_pairs = []
    for dep in cand["top_deps"]:
        lid = dep["left_rule_id"]
        rid = dep["right_rule_id"]
        if lid in top10_ids and rid not in top10_ids:
            cross_pairs.append((lid, rid, dep["dependency_type"], dep["sign"], dep["estimated_candidate_contribution"]))
        elif rid in top10_ids and lid not in top10_ids:
            cross_pairs.append((rid, lid, dep["dependency_type"], dep["sign"], dep["estimated_candidate_contribution"]))

    if direct_pairs:
        print(f"    Direct top-10 ↔ top-10 pairs:")
        for lid, rid, dtype, sign, contrib in direct_pairs:
            icon = "✗ redundancy" if sign == "negative" else "✓ synergy"
            print(f"      {icon}: {lid} ↔ {rid} ({dtype})  contrib={contrib:+.4f}")
    else:
        print(f"    ✗ No direct pairs among top-10 rules in top dependency list.")

    if cross_pairs:
        print(f"\n    Cross pairs: top-10 rule ↔ lower-weight rule:")
        # Sort by absolute contribution
        cross_pairs.sort(key=lambda x: abs(x[4]), reverse=True)
        for lid, rid, dtype, sign, contrib in cross_pairs[:5]:
            icon = "✗" if sign == "negative" else "✓"
            ltext = clean_rule_text(rule_map.get(lid, {}).get("text", lid))[:35]
            rtext = clean_rule_text(rule_map.get(rid, {}).get("text", rid))[:35]
            print(f"      {icon} {lid} ↔ {rid} ({dtype}, {contrib:+.4f})")
            print(f"         {ltext}  ↔  {rtext}")
    else:
        print(f"    ✗ No cross pairs found in top dependency list either.")

    if overlap:
        print(f"\n    Top-10 rules appearing in any dependency: {overlap}")
        for rid in overlap:
            r = rule_map.get(rid, {})
            print(f"      {rid} ({r.get('type','?')}, w={r.get('w1',0):.4f})")
    else:
        print(f"\n    ✗ None of the top-10 rules appear in top dependencies.")
        print(f"    → The high-weight rules are essentially independent.")
        print(f"    → Redundancy/synergy is driven by mid-weight and low-weight rules.")

    print()
    print()
