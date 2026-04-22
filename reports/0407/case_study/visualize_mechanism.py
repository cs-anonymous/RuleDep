import json
import os
import numpy as np
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
    }

# === Build rule map ===
rule_map = {}
for r in graph["rule_dependency_layer"]["rules"]:
    rule_map[r["rule_id"]] = {
        "type": r["rule_type"],
        "text": r["rule_text"],
        "w1": r["weight"]["stage1"]["effective_weight"],
        "w2": r["weight"]["stage2"]["effective_weight"],
    }

# === Focus: English and Persian ===
english = candidates["/m/02h40lc"]
persian = candidates["/m/032f6"]

# --- Collect top dependencies for each ---
# English: top negative deps (penalize)
english_neg = [d for d in english["top_deps"] if d["sign"] == "negative"][:6]
# Persian: top positive deps (boost)
persian_pos = [d for d in persian["top_deps"] if d["sign"] == "positive"][:6]

# Collect all rule texts referenced by these deps
relevant_rules = {}
for dep in english_neg + persian_pos:
    for side in ["left", "right"]:
        rid = dep[f"{side}_rule_id"]
        if rid not in relevant_rules and rid in rule_map:
            relevant_rules[rid] = rule_map[rid]

print(f"Relevant rules from top deps: {list(relevant_rules.keys())}")

# --- Score summary ---
print(f"\nEnglish: rule={english['rule_total']:.2f}, dep={english['dependency_total']:+.3f}, total_before={english['score_before']:.3f}, after={english['score_after']:.3f}")
print(f"Persian: rule={persian['rule_total']:.2f}, dep={persian['dependency_total']:+.3f}, total_before={persian['score_before']:.3f}, after={persian['score_after']:.3f}")

# === Visualization ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]  # left: English
ax2 = axes[1]  # right: Persian

for side_idx, (cand, ax_obj, color_main) in enumerate([
    (english, ax, "#FF9800"),
    (persian, ax2, "#E53935"),
]):
    label = cand["label"]
    rank_before = int(cand["rank_before"])
    rank_after = int(cand["rank_after"])

    # Title with rank change
    ax_obj.set_title(
        f"{label}  rank #{rank_before} → #{rank_after}",
        fontsize=12, fontweight="bold", pad=12
    )

    # Score bar: before vs after
    y_pos = 0.95
    bar_height = 0.12

    # Before dep score
    rect_before = mpatches.Rectangle((0.1, y_pos), cand["score_before"] * 2.5, bar_height,
                                      facecolor="#BDBDBD", edgecolor="none", alpha=0.6)
    ax_obj.add_patch(rect_before)
    ax_obj.text(cand["score_before"] * 2.5 + 0.02, y_pos + bar_height / 2,
                f"stage-1: {cand['score_before']:.3f}", fontsize=8, va="center", color="#555")

    y_pos -= 0.17
    # After dep score
    rect_after = mpatches.Rectangle((0.1, y_pos), cand["score_after"] * 2.5, bar_height,
                                     facecolor=color_main, edgecolor="none", alpha=0.85)
    ax_obj.add_patch(rect_after)
    ax_obj.text(cand["score_after"] * 2.5 + 0.02, y_pos + bar_height / 2,
                f"stage-2: {cand['score_after']:.3f}", fontsize=8, va="center", color="#333", fontweight="bold")

    y_pos -= 0.17
    # Dependency contribution
    dep_val = cand["dependency_total"]
    dep_color = "#E53935" if dep_val < 0 else "#4CAF50"
    dep_width = abs(dep_val) * 2.5
    dep_start = 0.1 + cand["score_after"] * 2.5 if dep_val > 0 else 0.1 + cand["score_before"] * 2.5
    rect_dep = mpatches.Rectangle((dep_start, y_pos), dep_width if dep_val > 0 else -dep_width, bar_height * 0.8,
                                   facecolor=dep_color, edgecolor="none", alpha=0.7)
    ax_obj.add_patch(rect_dep)
    ax_obj.text(dep_start + (dep_width / 2 if dep_val > 0 else -dep_width / 2), y_pos + bar_height * 0.4,
                f"dep: {dep_val:+.3f}", fontsize=7.5, va="center", ha="center", color="white", fontweight="bold")

    y_pos -= 0.20
    # Rule components by type
    ax_obj.text(0.1, y_pos, f"rule breakdown:", fontsize=8, fontweight="bold", color="#333")
    y_pos -= 0.06
    for rtype, rval in sorted(cand["rule_types"].items()):
        color_rt = "#2F6DA3" if rtype == "B" else "#9C4E8A"
        rect_rt = mpatches.Rectangle((0.15, y_pos), rval * 0.8, bar_height * 0.7,
                                      facecolor=color_rt, alpha=0.7)
        ax_obj.add_patch(rect_rt)
        ax_obj.text(0.15 + rval * 0.8 + 0.02, y_pos + bar_height * 0.35,
                    f"{rtype}: {rval:.3f}", fontsize=7, va="center")
        y_pos -= 0.08

    y_pos -= 0.06
    # Dependency components by type
    ax_obj.text(0.1, y_pos, f"dependency breakdown:", fontsize=8, fontweight="bold", color="#333")
    y_pos -= 0.06
    for dtype, dval in sorted(cand["dep_types"].items()):
        color_dt = "#4CAF50" if dval > 0 else "#E53935"
        rect_dt = mpatches.Rectangle((0.15, y_pos), abs(dval) * 2.5, bar_height * 0.7,
                                      facecolor=color_dt, alpha=0.7)
        ax_obj.add_patch(rect_dt)
        ax_obj.text(0.15 + abs(dval) * 2.5 + 0.02, y_pos + bar_height * 0.35,
                    f"{dtype}: {dval:+.3f}", fontsize=7, va="center")
        y_pos -= 0.08

    y_pos -= 0.08
    # Top dependency interactions
    ax_obj.text(0.1, y_pos, f"top dependencies:", fontsize=8, fontweight="bold", color="#333")
    y_pos -= 0.06

    top_deps = cand["top_deps"]
    if side_idx == 0:
        # English: show top negative
        shown = [d for d in top_deps if d["sign"] == "negative"][:4]
    else:
        # Persian: show top positive
        shown = [d for d in top_deps if d["sign"] == "positive"][:4]

    for i, dep in enumerate(shown):
        left_id = dep["left_rule_id"]
        right_id = dep["right_rule_id"]
        contrib = dep["estimated_candidate_contribution"]
        frac = dep["estimated_fraction_of_dependency_total"]
        dep_color = "#E53935" if contrib < 0 else "#4CAF50"

        dep_text = f"{dep['dep_id']} ({dep['dependency_type']}): {contrib:+.4f}  [{frac*100:.1f}%]"
        ax_obj.text(0.1, y_pos, dep_text, fontsize=6.5, color=dep_color, fontweight="bold" if abs(contrib) > 0.014 else "normal")
        y_pos -= 0.045

        # Show abbreviated rule pair
        left_text_short = dep.get("left_rule_text", "?")[:60] + "..."
        right_text_short = dep.get("right_rule_text", "?")[:60] + "..."
        ax_obj.text(0.15, y_pos, f"  {left_id} ↔ {right_id}", fontsize=5.5, color="#666")
        y_pos -= 0.045
        ax_obj.text(0.15, y_pos, f"  {left_text_short}", fontsize=5, color="#888")
        y_pos -= 0.04
        ax_obj.text(0.15, y_pos, f"  {right_text_short}", fontsize=5, color="#888")
        y_pos -= 0.07

    ax_obj.set_xlim(0, 1)
    ax_obj.set_ylim(0, 1)
    ax_obj.axis("off")

plt.suptitle(
    "Dependency-Aware Score Decomposition: English (penalized) vs Persian (boosted)",
    fontsize=12, fontweight="bold", y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(DATA, "score_decomposition.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"\nSaved score decomposition to {output_path}")
plt.close()

# === Part 2: Rule→Candidate connection diagram ===
fig2, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor("#FAFAFA")

# Layout: Iraq bottom center, English left-top, Persian right-top
# Rules in between
iraq_pos = (0.5, 0.08)
eng_pos = (0.25, 0.85)
per_pos = (0.75, 0.85)

# Draw Iraq
iraq_circle = plt.Circle(iraq_pos, 0.035, facecolor="#2E8B75", zorder=5, edgecolor="white", lw=2.5)
ax.add_patch(iraq_circle)
ax.text(iraq_pos[0], iraq_pos[1] - 0.06, "Iraq", fontsize=10, fontweight="bold", ha="center", va="top", color="#1B5E20")

# Draw candidates
for pos, label, color, rank_info in [
    (eng_pos, "English", "#FF9800", "1 → 2"),
    (per_pos, "Persian", "#E53935", "2 → 1"),
]:
    circle = plt.Circle(pos, 0.04, facecolor=color, alpha=0.9, zorder=5, edgecolor="white", lw=2.5)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1] + 0.06, label, fontsize=10, fontweight="bold", ha="center", va="bottom")
    ax.text(pos[0], pos[1] - 0.06, f"rank #{rank_info}", fontsize=8, fontweight="bold",
            color=color, ha="center", va="top")

# Draw dependency correction arrows between candidates
ax.annotate("", xy=(per_pos[0] - 0.08, per_pos[1]),
            xytext=(eng_pos[0] + 0.08, eng_pos[1]),
            arrowprops=dict(arrowstyle="<->", color="#9C27B0", lw=2.5, alpha=0.6,
                           connectionstyle="arc3,rad=0.15"))
ax.text(0.5, 0.9, "dependency correction\nreverses ranking", fontsize=8, ha="center",
        color="#9C27B0", fontweight="bold", style="italic")

# Score delta annotations
ax.text(eng_pos[0], eng_pos[1] - 0.13,
        f"Δ score = {english['dependency_total']:+.3f}", fontsize=9, ha="center",
        color="#E53935", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#E53935", alpha=0.15, edgecolor="none"))
ax.text(per_pos[0], per_pos[1] - 0.13,
        f"Δ score = {persian['dependency_total']:+.3f}", fontsize=9, ha="center",
        color="#4CAF50", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#4CAF50", alpha=0.15, edgecolor="none"))

# Now show top dependency pairs as "bridges" between the two sides
# English negative deps on left, Persian positive deps on right

dep_y_start = 0.35
dep_spacing = 0.11

# English negative deps
ax.text(0.25, dep_y_start + 0.08, "English: redundancy penalties", fontsize=8,
        fontweight="bold", ha="center", color="#E53935")
for i, dep in enumerate(english_neg[:4]):
    y = dep_y_start - i * dep_spacing
    contrib = dep["estimated_candidate_contribution"]
    left_id = dep["left_rule_id"]
    right_id = dep["right_rule_id"]
    left_text = dep.get("left_rule_text", "?")
    right_text = dep.get("right_rule_text", "?")

    # Shorten rule texts
    lt = left_text.replace("/m/0bfvd4", "The_Office").replace("award_category", "award").replace("award_honor", "honor")[:50]
    rt = right_text[:50]

    # Draw a "bridge" between left and right rule
    left_x = 0.12
    right_x = 0.38
    ax.plot([left_x, right_x], [y, y], color="#E53935", lw=1.5, alpha=0.5, zorder=2)
    # Cross mark in middle
    ax.text((left_x + right_x) / 2, y, "⊗", fontsize=8, ha="center", va="center", color="#E53935", zorder=3)
    # Contribution label
    ax.text((left_x + right_x) / 2, y + 0.025, f"{contrib:+.4f}", fontsize=6,
            ha="center", va="bottom", color="#E53935", fontweight="bold")
    # Rule labels
    ax.text(left_x, y - 0.02, f"{left_id}", fontsize=5.5, ha="center", va="top", color="#666")
    ax.text(right_x, y - 0.02, f"{right_id}", fontsize=5.5, ha="center", va="top", color="#666")

# Persian positive deps
ax.text(0.75, dep_y_start + 0.08, "Persian: synergy boosts", fontsize=8,
        fontweight="bold", ha="center", color="#4CAF50")
persian_pos_deps = [d for d in persian["top_deps"] if d["sign"] == "positive"][:6]
for i, dep in enumerate(persian_pos_deps[:4]):
    y = dep_y_start - i * dep_spacing
    contrib = dep["estimated_candidate_contribution"]
    left_id = dep["left_rule_id"]
    right_id = dep["right_rule_id"]
    left_text = dep.get("left_rule_text", "?")
    right_text = dep.get("right_rule_text", "?")

    lt = left_text[:50]
    rt = right_text[:50]

    left_x = 0.62
    right_x = 0.88
    ax.plot([left_x, right_x], [y, y], color="#4CAF50", lw=1.5, alpha=0.5, zorder=2)
    # Plus mark in middle
    ax.text((left_x + right_x) / 2, y, "⊕", fontsize=8, ha="center", va="center", color="#4CAF50", zorder=3)
    ax.text((left_x + right_x) / 2, y + 0.025, f"{contrib:+.4f}", fontsize=6,
            ha="center", va="bottom", color="#4CAF50", fontweight="bold")
    ax.text(left_x, y - 0.02, f"{left_id}", fontsize=5.5, ha="center", va="top", color="#666")
    ax.text(right_x, y - 0.02, f"{right_id}", fontsize=5.5, ha="center", va="top", color="#666")

# Central explanation
ax.text(0.5, 0.2,
        "English receives generic evidence from many rules (U-type dominant)\n"
        "but is heavily penalized by redundancy across co-fired rules\n"
        "Persian has fewer rules but benefits from positive dependency synergy",
        fontsize=8, ha="center", va="center", color="#555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ddd"))

ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1)
ax.axis("off")

output_path2 = os.path.join(DATA, "dependency_mechanism.png")
plt.savefig(output_path2, dpi=200, bbox_inches="tight")
print(f"Saved dependency mechanism to {output_path2}")
plt.close()
