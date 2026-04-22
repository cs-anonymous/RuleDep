import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA = "/Users/a./Documents/RuleDep/reports/0407/case_study"
JSON_FILE = os.path.join(DATA, "FB15k-237_countries_spoken_in_m_0d05q4_0112_graph.json")

with open(JSON_FILE) as f:
    graph = json.load(f)

# === Extract candidate data ===
candidates = {}
for ce in graph["candidate_explanations"]:
    eid = ce["entity_id"]
    candidates[eid] = {
        "label": ce["entity_label"],
        "role": ce["role"],
        "rank_before": ce["rank"]["before_dep"],
        "rank_after": ce["rank"]["after_dep"],
        "score_before": ce["rank_score"]["before_dep"],
        "score_after": ce["rank_score"]["after_dep"],
        "rule_total": ce["score"]["before_dep"]["rule_total"],
        "rule_types": ce["score"]["before_dep"]["rule_components"]["by_rule_type"],
        "dependency_total": ce["score"]["after_dep"]["dependency_total"],
        "dep_types": ce["score"]["after_dep"]["dependency_components"]["by_dependency_type"],
        "dep_signs": ce["score"]["after_dep"]["dependency_components"]["by_sign"],
        "top_deps": ce.get("top_dependency_details", []),
        "active_rule_count": ce["active_rule_count"],
    }

english = candidates["/m/02h40lc"]
persian = candidates["/m/032f6"]

# === Build rule map ===
rule_map = {}
for r in graph["rule_dependency_layer"]["rules"]:
    rule_map[r["rule_id"]] = {
        "type": r["rule_type"],
        "text": r["rule_text"],
    }

def rule_label(rid, max_len=70):
    """Get a shortened, readable rule label."""
    if rid in rule_map:
        text = rule_map[rid]["text"]
        rtype = rule_map[rid]["type"]
    else:
        text = rid
        rtype = "?"

    # Remove Freebase prefixes for readability
    clean = text
    # Shorten relation paths
    for pattern, repl in [
        ("/film/film/release_date_s./film/film_regional_release_date/film_release_region", "film_release_region"),
        ("/award/award_category/winners./award/award_honor/award_winner", "award_winner"),
        ("/award/award_category/nominees./award/award_nomination/nominated_for", "nominated_for"),
        ("/music/performance_role/track_performances./music/track_contribution/role", "music_role"),
        ("/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country", "olympic_country"),
        ("/people/person/languages", "person_languages"),
        ("/people/person/religion", "person_religion"),
        ("/education/educational_degree/people_with_this_degree./education/education/institution", "degree_institution"),
        ("/location/statistical_region/religions./location/religion_percentage/religion", "region_religion"),
        ("/film/film/other_crew./film/film_crew_gig/film_crew_role", "film_crew_role"),
        ("/film/film/music", "film_music"),
        ("/award/award_winner/awards_won./award/award_honor/award_winner", "award_winner_aw"),
        ("/award/award_nominee/award_nominations./award/award_nomination/award", "award_nom"),
        ("/award/award_nominee/award_nominations./award/award_nomination/award_nominee", "award_nominee"),
        ("/media_common/netflix_genre/titles", "netflix_titles"),
        ("/music/record_label/artist", "label_artist"),
        ("/music/genre/artists", "genre_artists"),
        ("/music/genre/parent_genre", "parent_genre"),
        ("/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft", "draft_pick"),
        ("/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor", "tv_actor"),
        ("/location/location/adjoin_s./location/adjoining_relationship/adjoins", "location_adjoins"),
        ("/music/performance_role/regular_performances./music/group_membership/role", "perf_role"),
        ("/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics", "olympic_medals"),
        ("/music/performance_role/guest_performances./music/recording_contribution/performance_role", "guest_perf"),
        ("/music/artist/track_contributions./music/track_contribution/role", "artist_track"),
    ]:
        clean = clean.replace(pattern, repl)

    # Remove Freebase IDs
    import re
    clean = re.sub(r'/m/\w+', '', clean)
    clean = clean.strip(' ()<=,')
    # Collapse multiple spaces
    clean = re.sub(r'\s+', ' ', clean).strip()

    if len(clean) > max_len:
        clean = clean[:max_len] + "..."

    return f"[{rtype}] {clean}"

# === Create the visualization ===
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_facecolor("#FAFAFA")

# Layout positions
iraq_pos = (0.5, 0.05)
eng_pos = (0.22, 0.88)
per_pos = (0.78, 0.88)

# === Iraq ===
iraq_circle = plt.Circle(iraq_pos, 0.04, facecolor="#2E8B75", zorder=5, edgecolor="white", lw=3)
ax.add_patch(iraq_circle)
ax.text(iraq_pos[0], iraq_pos[1] - 0.065, "Iraq", fontsize=12, fontweight="bold",
        ha="center", va="top", color="#1B5E20")
ax.text(iraq_pos[0], iraq_pos[1] - 0.10, "(seed entity)", fontsize=7, ha="center",
        va="top", color="#666")

# === Candidates ===
for pos, label, color, rank_info, score_before, score_after, dep_delta, active_count in [
    (eng_pos, "English", "#FF9800", "1 → 2",
     english["score_before"], english["score_after"], english["dependency_total"],
     english["active_rule_count"]),
    (per_pos, "Persian", "#E53935", "2 → 1",
     persian["score_before"], persian["score_after"], persian["dependency_total"],
     persian["active_rule_count"]),
]:
    circle = plt.Circle(pos, 0.045, facecolor=color, alpha=0.9, zorder=5, edgecolor="white", lw=3)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1] + 0.065, label, fontsize=13, fontweight="bold", ha="center", va="bottom")
    ax.text(pos[0], pos[1] - 0.065, f"rank #{rank_info}", fontsize=10, fontweight="bold",
            color=color, ha="center", va="top")

    # Score boxes
    ax.text(pos[0], pos[1] - 0.115,
            f"stage-1: {score_before:.3f}    stage-2: {score_after:.3f}",
            fontsize=8, ha="center", va="top", color="#333")

    # Dependency delta
    delta_color = "#E53935" if dep_delta < 0 else "#4CAF50"
    ax.text(pos[0], pos[1] - 0.155,
            f"dependency Δ = {dep_delta:+.3f}",
            fontsize=9, fontweight="bold", ha="center", va="top", color=delta_color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=delta_color, alpha=0.12, edgecolor="none"))

    ax.text(pos[0], pos[1] - 0.19,
            f"({active_count} active rules)",
            fontsize=7, ha="center", va="top", color="#888")

# === Dependency correction arrow ===
ax.annotate("", xy=(per_pos[0] - 0.12, per_pos[1] - 0.02),
            xytext=(eng_pos[0] + 0.12, eng_pos[1] - 0.02),
            arrowprops=dict(arrowstyle="<->", color="#9C27B0", lw=3, alpha=0.5,
                           connectionstyle="arc3,rad=0.08"))
ax.text(0.5, 0.95, "dependency correction reverses ranking",
        fontsize=10, fontweight="bold", ha="center", color="#9C27B0", style="italic")

# === English: top negative dependencies ===
eng_neg = [d for d in english["top_deps"] if d["sign"] == "negative"][:5]
y_start = 0.62

ax.text(eng_pos[0], y_start + 0.08, "Top redundancy penalties (English)",
        fontsize=9, fontweight="bold", ha="center", color="#E53935")

for i, dep in enumerate(eng_neg):
    y = y_start - i * 0.12
    contrib = dep["estimated_candidate_contribution"]
    left_id = dep["left_rule_id"]
    right_id = dep["right_rule_id"]
    dep_type = dep["dependency_type"]
    frac = dep["estimated_fraction_of_dependency_total"]

    left_x = 0.06
    right_x = 0.38

    # Draw connection line
    ax.plot([left_x, right_x], [y, y], color="#E53935", lw=1.5, alpha=0.4, zorder=2)
    # Redundancy cross
    ax.text((left_x + right_x) / 2, y, "⊗", fontsize=9, ha="center", va="center", color="#E53935", zorder=3)

    # Contribution label
    ax.text((left_x + right_x) / 2, y + 0.03,
            f"{contrib:+.4f}  ({frac*100:.1f}%)",
            fontsize=7, ha="center", va="bottom", color="#E53935", fontweight="bold")

    # Rule labels with text
    left_text = rule_label(left_id, 45)
    right_text = rule_label(right_id, 45)

    ax.text(left_x, y - 0.025, left_text, fontsize=5.5, ha="left", va="top", color="#555",
            fontfamily="monospace")
    ax.text(right_x, y - 0.025, right_text, fontsize=5.5, ha="right", va="top", color="#555",
            fontfamily="monospace")

# === Persian: top positive dependencies ===
per_pos_deps = [d for d in persian["top_deps"] if d["sign"] == "positive"][:5]

ax.text(per_pos[0], y_start + 0.08, "Top synergy boosts (Persian)",
        fontsize=9, fontweight="bold", ha="center", color="#4CAF50")

for i, dep in enumerate(per_pos_deps):
    y = y_start - i * 0.12
    contrib = dep["estimated_candidate_contribution"]
    left_id = dep["left_rule_id"]
    right_id = dep["right_rule_id"]
    dep_type = dep["dependency_type"]
    frac = dep["estimated_fraction_of_dependency_total"]

    left_x = 0.62
    right_x = 0.94

    # Draw connection line
    ax.plot([left_x, right_x], [y, y], color="#4CAF50", lw=1.5, alpha=0.4, zorder=2)
    # Synergy plus
    ax.text((left_x + right_x) / 2, y, "⊕", fontsize=9, ha="center", va="center", color="#4CAF50", zorder=3)

    # Contribution label
    ax.text((left_x + right_x) / 2, y + 0.03,
            f"{contrib:+.4f}  ({frac*100:.1f}%)",
            fontsize=7, ha="center", va="bottom", color="#4CAF50", fontweight="bold")

    # Rule labels
    left_text = rule_label(left_id, 45)
    right_text = rule_label(right_id, 45)

    ax.text(left_x, y - 0.025, left_text, fontsize=5.5, ha="left", va="top", color="#555",
            fontfamily="monospace")
    ax.text(right_x, y - 0.025, right_text, fontsize=5.5, ha="right", va="top", color="#555",
            fontfamily="monospace")

# === Central explanation box ===
ax.text(0.5, 0.28,
        "Key observations:\n\n"
        "  English: 171 active rules fire, dominated by U-type (general) evidence.\n"
        "  Heavy redundancy penalties across co-fired rules (Δ = -0.904).\n"
        "  Top penalty hub: R19792 (award/TV actor rule) appears in 4 of top-5 deps.\n\n"
        "  Persian: 109 active rules, more balanced B/U split.\n"
        "  All dependencies are positive synergy (Δ = +0.409).\n"
        "  Key synergy pairs involve olympic_country + genre_artists rules.",
        fontsize=8, ha="center", va="center", color="#444",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ddd", lw=1))

# === Legend ===
legend_elements = [
    plt.Line2D([0], [0], color="#E53935", lw=1.5, marker="x", markersize=8, label="Redundancy (negative dep)"),
    plt.Line2D([0], [0], color="#4CAF50", lw=1.5, marker="+", markersize=8, label="Synergy (positive dep)"),
    plt.Line2D([0], [0], marker="o", markersize=10, color="none", markerfacecolor="#FF9800", label="English (was #1)"),
    plt.Line2D([0], [0], marker="o", markersize=10, color="none", markerfacecolor="#E53935", label="Persian (gold, now #1)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

plt.suptitle(
    "FB15k-237: countries\_spoken\_in(Iraq) — Dependency Mechanism",
    fontsize=12, fontweight="bold", y=0.99
)

output_path = os.path.join(DATA, "dependency_mechanism_v2.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.close()
