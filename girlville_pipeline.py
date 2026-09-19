"""Girlville spread map: 17 x 11 in @ 300 dpi — the frontispiece spread,
printed across the first fold after the title page.

Applies the current schelling-campaign-map skill from scratch:
  - seeds: girlville_data.py (sourceless path — Poisson-disc sampling with
    greedy nearest-anchor assignment to authorial anchors; the survey output
    is already committed and the geography approved, so seeds are kept)
  - finite Voronoi (cloud-center sign rule) clipped to the 17x11 canvas
  - typed multiplex graph G = (V, E_A, E_F, E_C) via the skill's
    bin/graph_model.py -> maps/girlville-graph.json
  - E_C is authored, not derived: Girlville has no portals or convoy routes
    yet, so the convoy layer stays empty until campaign events create one.

Ink-friendly (the reader prints on an inkjet): white background, thin
ice-blue borders, dark slate labels. Wild regions get a very light tint;
waypoints are dashed gray: neutral, unclaimed, never quest hosts.

Markers sit EXACTLY at their seeds, never at cell centroids.
Labels follow ONE uniform rule (no hand-tuned per-label offsets): centered
below the marker with a fixed vertical gap; a deterministic frame-clamp
shifts a label horizontally to stay inside the frame, or flips it above the
marker if it would cross the bottom edge. `get_label_bboxes()` exposes the
resulting boxes in canvas inches so the verify gate can enforce: no
label-label overlaps, no label-marker overlaps, no label outside the frame.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import Voronoi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from girlville_data import SEEDS  # noqa: E402

SKILL_BIN = os.path.join(os.path.dirname(HERE), "skills",
                         "schelling-campaign-map", "bin")
sys.path.insert(0, SKILL_BIN)
from graph_model import (  # noqa: E402
    SpaceType, CampaignRole, DiplomacyMultiplexGraph,
    Node, derive_layers, split_coasts, to_json,
)

OUT_DIR = os.path.join(HERE, "maps")
OUT_PNG = os.path.join(OUT_DIR, "girlville-frost-kingdom.png")
OUT_JSON = os.path.join(OUT_DIR, "girlville-summary.json")
OUT_GRAPH = os.path.join(OUT_DIR, "girlville-graph.json")

# Spread: 17 x 11 in @ 300 dpi (5100 x 3300 px). Data units are inches.
FIG_W, FIG_H, DPI = 17, 11, 300
CANVAS_W, CANVAS_H = 17.0, 11.0

ICE = "#9db8cc"      # region borders
SLATE = "#2b3a4a"    # labels
SUPPLY = "#4a7ba6"   # supply markers
WILD_C = "#5a6b7d"   # wild markers
WAY_C = "#8a8f98"    # waypoint markers/edges
WILD_TINT = "#eef4f9"

# ---- uniform label rule ----
# Type scales with the canvas HEIGHT ratio (8.8/5.1 ~= 1.73): labels sit
# below their markers, so vertical density is the binding constraint.
# (Scaling by the width ratio made labels collide; the gate caught it.)
LABEL_DY = 0.44           # fixed vertical gap below each marker, in inches
LABEL_FONTS = {"supply": 13.0, "wild": 12.0, "waypoint": 11.0}
FRAME_MARGIN_X = 0.034    # keep labels this far inside the frame
FRAME_MARGIN_Y = 0.022
# Two-line wraps for the long names (uniform rule still applies).
LABEL_TEXT = {
    "The Coven Below the Frost": "The Coven\nBelow the Frost",
    "Guild of Lacy Shadows": "Guild of\nLacy Shadows",
    "The Grand Frost Ballroom": "The Grand Frost\nBallroom",
    "Mines of Glimmōr": "Mines of\nGlimmōr",
}
# Marker half-extent in inches, for the collision check (from ms in points).
MARKER_HALF = {"supply": 0.085, "wild": 0.135, "waypoint": 0.07}
MARKER_MS = {"supply": 12, "wild": 19, "waypoint": 10}

# Sea cells for the typed graph (E_F layer). The Sewers counts as water.
SEA_NAMES = {"The Sewers"}


def voronoi_finite_polygons_2d(vor, radius=None):
    """Finite Voronoi regions. Outward normal uses the cloud-center sign rule:
    sign(dot(ridge midpoint - cloud center, normal)) x normal.
    (midpoint-minus-seed is perpendicular to the normal and silently wrong.)
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # finite ridge already in the region
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            n = np.sign(np.dot(midpoint - center, n)) * n  # cloud-center sign rule
            far_point = vor.vertices[v2] + n * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def clip_to_rect(poly, w, h):
    """Sutherland-Hodgman clip against [0,w]x[0,h]."""
    def inside(p, edge):
        x, y = p
        return (x >= 0 if edge == "left" else
                x <= w if edge == "right" else
                y >= 0 if edge == "bottom" else y <= h)
    def intersect(p1, p2, edge):
        x1, y1 = p1; x2, y2 = p2
        if edge == "left":   t = (0 - x1) / (x2 - x1); return (0.0, y1 + t * (y2 - y1))
        if edge == "right":  t = (w - x1) / (x2 - x1); return (float(w), y1 + t * (y2 - y1))
        if edge == "bottom": t = (0 - y1) / (y2 - y1); return (x1 + t * (x2 - x1), 0.0)
        t = (h - y1) / (y2 - y1); return (x1 + t * (x2 - x1), float(h))
    out = [tuple(p) for p in poly]
    for edge in ("left", "right", "bottom", "top"):
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for p in inp:
            if inside(p, edge):
                if not inside(s, edge):
                    out.append(intersect(s, p, edge))
                out.append(p)
            elif inside(s, edge):
                out.append(intersect(s, p, edge))
            s = p
    return np.asarray(out)


def seed_xy():
    """Seeds in canvas inches: normalized anchors mapped to the 17x11 rect.
    Relative geography is preserved exactly; the canvas is wider."""
    return np.array([[s["fx"] * CANVAS_W, s["fy"] * CANVAS_H] for s in SEEDS])


def build_figure():
    """Build the map figure. Returns (fig, ax, texts, xy)."""
    xy = seed_xy()
    vor = Voronoi(xy)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    # Map rect matches the 17:11 canvas aspect; title above, legend below.
    ax = fig.add_axes([0.10, 0.09, 0.80, 0.80])
    ax.set_facecolor("white")

    # Region cells
    for s, region in zip(SEEDS, regions):
        poly = clip_to_rect(vertices[region], CANVAS_W, CANVAS_H)
        if len(poly) < 3:
            continue
        if s["kind"] == "wild":
            ax.add_patch(Polygon(poly, closed=True, facecolor=WILD_TINT,
                                 edgecolor=ICE, linewidth=1.75, zorder=1))
        elif s["kind"] == "waypoint":
            ax.add_patch(Polygon(poly, closed=True, facecolor="white",
                                 edgecolor=WAY_C, linewidth=1.4,
                                 linestyle=(0, (7, 5)), zorder=1))
        else:
            ax.add_patch(Polygon(poly, closed=True, facecolor="white",
                                 edgecolor=ICE, linewidth=1.75, zorder=1))

    # Markers EXACTLY at the seeds + uniform labels (centered below marker)
    texts = []
    for s, (x, y) in zip(SEEDS, xy):
        if s["kind"] == "supply":
            ax.plot(x, y, marker="D", ms=MARKER_MS["supply"], mfc=SUPPLY,
                    mec="white", mew=1.75, zorder=5, clip_on=False)
            t = ax.text(x, y + LABEL_DY, LABEL_TEXT.get(s["name"], s["name"]),
                        fontsize=LABEL_FONTS["supply"], color=SLATE,
                        ha="center", va="top", zorder=6,
                        family="serif", weight="bold")
        elif s["kind"] == "wild":
            ax.plot(x, y, marker="*", ms=MARKER_MS["wild"], mfc=WILD_C,
                    mec="white", mew=1.4, zorder=5, clip_on=False)
            t = ax.text(x, y + LABEL_DY, LABEL_TEXT.get(s["name"], s["name"]),
                        fontsize=LABEL_FONTS["wild"], color=SLATE,
                        ha="center", va="top", zorder=6,
                        family="serif", style="italic")
        else:
            ax.plot(x, y, marker="o", ms=MARKER_MS["waypoint"], mfc="white",
                    mec=WAY_C, mew=2.0, zorder=5, clip_on=False)
            t = ax.text(x, y + LABEL_DY, LABEL_TEXT.get(s["name"], s["name"]),
                        fontsize=LABEL_FONTS["waypoint"], color=WAY_C,
                        ha="center", va="top", zorder=6,
                        family="serif", style="italic")
        texts.append(t)

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)  # origin top-left, matching (fx, fy)
    ax.set_aspect("equal")
    ax.axis("off")

    _frame_clamp_labels(fig, ax, texts, xy)

    # Title block (figure coordinates — clear of the map)
    fig.text(0.5, 0.945, "The Frost Kingdom of Girlville",
             fontsize=40, color=SLATE, ha="center", va="center",
             family="serif", weight="bold")
    fig.text(0.5, 0.04, "\u25c6 \u2014 supply  \u00b7  \u2605 \u2014 wild  \u00b7  \u25cb \u2014 neutral ground",
             fontsize=24, color=WAY_C, ha="center", va="center",
             family="DejaVu Sans")

    return fig, ax, texts, xy


def _frame_clamp_labels(fig, ax, texts, xy):
    """Deterministic frame clamp, part of the uniform label rule: shift a
    label horizontally to stay inside the frame; flip it above its marker if
    it would cross the bottom edge. Same geometric rule for every label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t, (sx, sy) in zip(texts, xy):
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        lo_x, hi_x, lo_y, hi_y = (min(x0, x1), max(x0, x1),
                                  min(y0, y1), max(y0, y1))
        nx, ny, va = sx, sy + LABEL_DY, "top"
        if lo_x < FRAME_MARGIN_X:
            nx += FRAME_MARGIN_X - lo_x
        elif hi_x > CANVAS_W - FRAME_MARGIN_X:
            nx -= hi_x - (CANVAS_W - FRAME_MARGIN_X)
        if hi_y > CANVAS_H - FRAME_MARGIN_Y:
            ny, va = sy - LABEL_DY, "bottom"
        t.set_position((nx, ny))
        t.set_va(va)


def get_label_bboxes():
    """Label and marker boxes in canvas inches, after frame clamping.
    Returns (label_boxes, marker_boxes): each a list of (x0, y0, x1, y1)."""
    fig, ax, texts, xy = build_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    label_boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        label_boxes.append((min(x0, x1), min(y0, y1),
                            max(x0, x1), max(y0, y1)))
    marker_boxes = []
    for s in SEEDS:
        mh = MARKER_HALF[s["kind"]]
        x, y = s["fx"] * CANVAS_W, s["fy"] * CANVAS_H
        marker_boxes.append((x - mh, y - mh, x + mh, y + mh))
    plt.close(fig)
    return label_boxes, marker_boxes


def build_graph():
    """Typed multiplex graph G = (V, E_A, E_F, E_C) from the Voronoi.

    Raw adjacency = shared Voronoi borders (ridge_points). E_A: borders
    between non-sea cells. E_F: borders touching a sea cell. E_C: authored
    only — Girlville has no portals/convoy routes yet, so it stays empty.
    """
    xy = seed_xy()
    vor = Voronoi(xy)
    names = [s["name"] for s in SEEDS]
    borders = [(names[i], names[j]) for i, j in vor.ridge_points]

    adj = {n: set() for n in names}
    for a, b in borders:
        adj[a].add(b)
        adj[b].add(a)

    nodes = {}
    for s in SEEDS:
        n = s["name"]
        if n in SEA_NAMES:
            st = SpaceType.SEA
        elif adj[n] & SEA_NAMES:
            st = SpaceType.COASTAL
        else:
            st = SpaceType.INLAND
        role = {"supply": CampaignRole.SUPPLY,
                "wild": CampaignRole.WILD,
                "waypoint": CampaignRole.WAYPOINT}[s["kind"]]
        nodes[n] = Node(id=n, name=n, space_type=st, campaign_role=role,
                        is_supply_center=(s["kind"] == "supply"))

    army, fleet = derive_layers(nodes, borders)
    graph = DiplomacyMultiplexGraph(nodes=nodes, army_edges=army,
                                    fleet_edges=fleet, convoy_edges=[])
    return split_coasts(graph)


def render():
    fig, ax, texts, xy = build_figure()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI)
    plt.close(fig)

    summary = {"canvas_inches": [FIG_W, FIG_H],
               "regions": [
                   {"name": s["name"], "kind": s["kind"],
                    "fx": s["fx"], "fy": s["fy"]}
                   for s in SEEDS]}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    graph = build_graph()
    with open(OUT_GRAPH, "w", encoding="utf-8") as f:
        f.write(to_json(graph, "The Frost Kingdom of Girlville"))

    n_coastal = sum(1 for n in graph.nodes.values()
                    if n.space_type == SpaceType.COASTAL)
    print(f"rendered {OUT_PNG} ({FIG_W*DPI:.0f}x{FIG_H*DPI:.0f})")
    print(f"wrote {OUT_JSON} ({len(SEEDS)} regions)")
    print(f"wrote {OUT_GRAPH}: {len(graph.army_edges)} E_A, "
          f"{len(graph.fleet_edges)} E_F, {len(graph.convoy_edges)} E_C, "
          f"{n_coastal} coastal nodes")


if __name__ == "__main__":
    render()
