import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

DATA = "/Users/a./Documents/RuleDep/reports/0407/case_study"
JSON_FILE = os.path.join(DATA, "FB15k-237_countries_spoken_in_m_0d05q4_0112_graph.json")

with open(JSON_FILE) as f:
    graph = json.load(f)

nodes_info = {n["id"]: n for n in graph["nodes"]}

# --- Collect prediction edges ---
pred_edges = []
for e in graph.get("prediction_edges", []):
    pred_edges.append(e)

# --- Collect KG edges ---
kg_edges = []
for e in graph.get("kg_edges", []):
    s = e["subject"]
    o = e["object"]
    if s in nodes_info and o in nodes_info:
        kg_edges.append((s, o, e.get("relation_label", "")))

# --- Determine which nodes to show ---
# Always show: Iraq (seed), prediction subjects/objects
seed_node = "/m/0d05q4"  # Iraq
pred_subjects = set()
pred_objects = set()
for e in pred_edges:
    pred_subjects.add(e["subject"])
    pred_objects.add(e["object"])

pred_nodes = pred_subjects | pred_objects

# Get KG neighbors for each prediction subject (limited to keep graph clean)
MAX_NEIGHBORS_PER_PRED = 8
selected_nodes = {seed_node} | pred_nodes
kg_edges_selected = []
kg_neighbors = {}  # pred_node -> list of neighbor node ids

for s, o, rel in kg_edges:
    if s in pred_nodes and o != seed_node:
        if s not in kg_neighbors:
            kg_neighbors[s] = []
        kg_neighbors[s].append((o, rel))
    if o in pred_nodes and s != seed_node:
        if o not in kg_neighbors:
            kg_neighbors[o] = []
        kg_neighbors[o].append((s, rel))

for pn in pred_nodes:
    if pn in kg_neighbors:
        neighbors = kg_neighbors[pn]
        # Sort by frequency or just take first N
        shown = neighbors[:MAX_NEIGHBORS_PER_PRED]
        for neighbor_id, rel in shown:
            selected_nodes.add(neighbor_id)
            kg_edges_selected.append((pn, neighbor_id, rel))
            # Also add reverse for completeness
            kg_edges_selected.append((neighbor_id, pn, rel))

# Build label map
def clean_label(node_id):
    info = nodes_info.get(node_id, {})
    return info.get("label", node_id.split("/")[-1]).replace("_", " ")

# --- Layout: manual hierarchical placement ---
# Iraq at bottom center
# Prediction candidates in a row above Iraq
# KG neighbors spread out on sides

pos = {}

# Seed node (Iraq) at bottom center
pos[seed_node] = (0.0, -0.3)

# Prediction subjects at top (above Iraq)
pred_list = sorted(pred_subjects)
x_spacing = 0.18
start_x = -(len(pred_list) - 1) * x_spacing / 2
for i, pid in enumerate(pred_list):
    pos[pid] = (start_x + i * x_spacing, 0.85)

# KG neighbors: arrange around their prediction parent
for pn in pred_nodes:
    if pn in kg_neighbors:
        neighbors = kg_neighbors[pn][:MAX_NEIGHBORS_PER_PRED]
        px, py = pos[pn]
        # Spread neighbors in a fan above and to sides
        for j, (nid, rel) in enumerate(neighbors):
            if nid in pos:
                continue
            angle = np.pi * 0.12 + j * (np.pi * 0.76 / max(len(neighbors) - 1, 1))
            radius = 0.42
            nx_ = px + radius * np.cos(angle)
            ny_ = py + radius * np.sin(angle) + 0.05
            pos[nid] = (nx_, ny_)

# --- Create figure ---
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor("#FAFAFA")

# --- Draw KG edges first (background, thin, gray) ---
drawn_kg = set()
for s, o, rel in kg_edges_selected:
    if s not in pos or o not in pos:
        continue
    edge_key = tuple(sorted([s, o]))
    if edge_key in drawn_kg:
        continue
    drawn_kg.add(edge_key)

    x1, y1 = pos[s]
    x2, y2 = pos[o]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color="#B0BEC5",
                               lw=0.6, alpha=0.45))

# --- Draw prediction edges (foreground, thick, colored) ---
pred_edge_colors = {}  # node_id -> color
for e in pred_edges:
    s = e["subject"]
    o = e["object"]
    if s not in pos or o not in pos:
        continue

    is_gold = e.get("is_gold", False)
    is_stage1_top1 = e.get("is_stage1_top1", False)
    rank_before = e["rank"]["before_dep"]
    rank_after = e["rank"]["after_dep"]
    score_before = e["rank_score"]["before_dep"]
    score_after = e["rank_score"]["after_dep"]

    # Color: gold = red, stage1_top1 = orange, others = muted
    if is_gold:
        edge_color = "#E53935"
        edge_width = 4.0
    elif is_stage1_top1:
        edge_color = "#FF9800"
        edge_width = 3.5
    else:
        edge_color = "#78909C"
        edge_width = 2.0

    pred_edge_colors[s] = edge_color

    # Draw with slight curve to stand out
    x1, y1 = pos[s]
    x2, y2 = pos[o]
    # Offset slightly for visual separation
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Draw curved arrow
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=edge_color,
                               lw=edge_width, alpha=0.85,
                               connectionstyle="arc3,rad=0.08"))

    # --- Rank annotation on the edge ---
    label = clean_label(s)
    rank_text = f"#{int(rank_before)} \u2192 #{int(rank_after)}"

    # Position the annotation slightly off-center
    ann_x = mid_x + 0.06
    ann_y = mid_y + 0.05

    bbox_props = dict(boxstyle="round,pad=0.18", facecolor=edge_color,
                      edgecolor="none", alpha=0.92)
    ax.text(ann_x, ann_y, rank_text, fontsize=8, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=bbox_props, zorder=10)

    # Node label below the rank
    ax.text(ann_x, ann_y - 0.065, label, fontsize=6.5,
            color=edge_color, ha="center", va="center",
            fontweight="bold", zorder=10)

# --- Draw nodes ---
node_colors_map = {
    seed_node: "#2E8B75",  # teal for Iraq
}
for pid in pred_subjects:
    for e in pred_edges:
        if e["subject"] == pid:
            if e.get("is_gold", False):
                node_colors_map[pid] = "#E53935"  # red for Persian (gold)
            elif e.get("is_stage1_top1", False):
                node_colors_map[pid] = "#FF9800"  # orange for English (was top1)
            else:
                node_colors_map[pid] = "#C77D28"  # amber for others

for nid in selected_nodes:
    if nid not in pos:
        continue
    color = node_colors_map.get(nid, "#90A4AE")
    is_pred = nid in pred_nodes
    is_seed = nid == seed_node

    node_size = 2200 if (is_pred or is_seed) else 900
    edge_width_node = 2.5 if (is_pred or is_seed) else 1.0
    alpha = 0.95 if (is_pred or is_seed) else 0.7

    circle = plt.Circle(pos[nid], 0.04 if (is_pred or is_seed) else 0.025,
                        facecolor=color, alpha=alpha, zorder=5,
                        edgecolor="white", lw=edge_width_node)
    ax.add_patch(circle)

    label = clean_label(nid)
    fs = 8 if (is_pred or is_seed) else 5.5
    font_color = "white" if is_seed else "#333333"

    if is_seed:
        # Draw label below the node
        ax.text(pos[nid][0], pos[nid][1] - 0.07, label,
                fontsize=fs, fontweight="bold", color="#1B5E20",
                ha="center", va="top", zorder=6)
    elif is_pred:
        # Draw label above the node
        ax.text(pos[nid][0], pos[nid][1] + 0.055, label,
                fontsize=fs, fontweight="bold", color="#333333",
                ha="center", va="bottom", zorder=6)

# --- Title ---
ax.set_title(
    "Case Study: countries\_spoken\_in(Iraq) — Dependency correction flips rank 1\u21922 / 2\u21921",
    fontsize=11, fontweight="bold", pad=15, color="#333333"
)

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor="#E53935", label="Gold (Persian)  #2\u2192#1"),
    mpatches.Patch(facecolor="#FF9800", label="Stage-1 top-1 (English)  #1\u2192#2"),
    mpatches.Patch(facecolor="#C77D28", label="Other candidate  #3\u2192#3"),
    mpatches.Patch(facecolor="#2E8B75", label="Seed entity (Iraq)"),
    plt.Line2D([0], [0], color="#B0BEC5", lw=1.5, alpha=0.6, label="KG edge"),
    plt.Line2D([0], [0], color="#E53935", lw=2.5, alpha=0.85, label="Prediction edge"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
          framealpha=0.9, edgecolor="#E0E0E0")

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 1.45)
ax.axis("off")

output_path = os.path.join(DATA, "FB15k-237_graph_viz_v3.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Saved to {output_path}")
plt.close()
