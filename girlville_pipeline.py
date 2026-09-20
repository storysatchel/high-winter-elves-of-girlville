"""Girlville spread map: 17 x 13.14 in @ 300 dpi (5100x3941 px) — the
frontispiece spread, exactly the 11:8.5 two-up half-letter ratio.

Applies the current schelling-campaign-map skill from scratch:
  - seeds: girlville_data.py (sourceless path — Poisson-disc sampling with
    greedy nearest-anchor assignment to authorial anchors; the survey output
    is already committed and the geography approved, so seeds are kept)
  - fractal jittered Voronoi boundaries (skill step 3a, Boris the Brave):
    region ownership rasterized at render resolution; the typed graph is
    derived from that same raster, so picture and graph agree
  - typed multiplex graph G = (V, E_A, E_F, E_C) via the skill's
    bin/graph_model.py -> maps/girlville-graph.json
  - E_C is authored, not derived: Girlville has no portals or convoy routes
    yet, so the convoy layer stays empty until campaign events create one.

Ink-friendly (the reader prints on an inkjet): white background, thin
ice-blue borders, dark slate labels. Wild regions get a very light tint;
waypoints stay neutral white: unclaimed, never quest hosts.

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
from scipy.ndimage import binary_dilation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from girlville_data import SEEDS, GREAT_POWERS  # noqa: E402

REGION_POWER = {}
for _p in GREAT_POWERS:
    for _r in _p["regions"]:
        REGION_POWER[_r] = _p
del _p, _r

SKILL_BIN = os.path.join(os.path.dirname(HERE), "skills",
                         "schelling-campaign-map", "bin")
sys.path.insert(0, SKILL_BIN)
from graph_model import (  # noqa: E402
    SpaceType, CampaignRole, DiplomacyMultiplexGraph,
    Node, derive_layers, split_coasts, to_json,
)
from fractal_voronoi import (  # noqa: E402
    FractalPartition, adjacency_from_grid,
)

OUT_DIR = os.path.join(HERE, "maps")
OUT_PNG = os.path.join(OUT_DIR, "girlville-frost-kingdom.png")
OUT_JSON = os.path.join(OUT_DIR, "girlville-summary.json")
OUT_GRAPH = os.path.join(OUT_DIR, "girlville-graph.json")
OUT_POWERS = os.path.join(OUT_DIR, "girlville-powers.json")

# Spread: 17 x 13.14 in @ 300 dpi (5100 x 3941 px). Data units are inches.
# Spread-filling aspect: the map spans two half-letter pages (11 x 8.5),
# so the canvas keeps that exact ratio (17 wide -> 17*8.5/11 tall).
FIG_W, FIG_H, DPI = 17, 17 * 8.5 / 11, 300
CANVAS_W, CANVAS_H = 17.0, 17.0 * 8.5 / 11
# NOTE: int(), not round() — the saved figure is 5100x3940, so the raster
# must match the figure's real pixel grid 1:1.
PX_W, PX_H = int(FIG_W * DPI), int(FIG_H * DPI)

# ---- fractal boundaries (skill step 3a, opt-in) ----
# Organic, coastline-like borders instead of straight Voronoi edges.
# depth=7: visible jaggedness at every scale; rng_seed re-rolls the borders
# without moving a single seed (deterministic for fixed inputs).
FRACTAL_DEPTH = 7
FRACTAL_SEED = 7
BORDER_PX = 9  # border width in raster pixels (~2.2pt at 300dpi)

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


def _rgb8(hexcolor):
    """matplotlib color -> (r, g, b) uint8 triple."""
    return tuple(int(round(v * 255))
                 for v in matplotlib.colors.to_rgb(hexcolor))


_partition_cache = {}


def get_partition():
    """FractalPartition ownership raster, computed once per process.

    The render AND the typed graph both derive from this same raster,
    so the picture and the graph agree by construction (skill rule).
    Row 0 is the top (y=0); values are seed indices.
    """
    if "grid" not in _partition_cache:
        fp = FractalPartition(seed_xy(), (CANVAS_W, CANVAS_H),
                              depth=FRACTAL_DEPTH, rng_seed=FRACTAL_SEED)
        bad = fp.check_seeds_self()
        if bad:
            raise RuntimeError(
                "fractal seeds do not own their location: %r" % (bad,))
        _partition_cache["grid"] = fp.rasterize(PX_W, PX_H).astype(np.int32)
    return _partition_cache["grid"]


def seed_xy():
    """Seeds in canvas inches: normalized anchors mapped to the 17x11 rect.
    Relative geography is preserved exactly; the canvas is wider."""
    return np.array([[s["fx"] * CANVAS_W, s["fy"] * CANVAS_H] for s in SEEDS])


def build_figure():
    """Build the map figure. Returns (fig, ax, texts, xy)."""
    xy = seed_xy()

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")
    # True full bleed: the fractal partition fills the entire figure,
    # edge to edge. Title, legend, and power key ride ON the map,
    # haloed for legibility.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    # Region cells, painted straight from the ownership raster
    # (row 0 = top, y=0): supply centers wear their great power's color;
    # wilds keep the wild tint; waypoints stay neutral white. Borders are
    # the dilated boundary mask in ice-blue — the same raster the typed
    # graph derives from, so picture and graph agree.
    grid = get_partition()
    fills = np.empty((len(SEEDS), 3), dtype=np.uint8)
    for i, s in enumerate(SEEDS):
        if s["kind"] == "wild":
            fills[i] = _rgb8(WILD_TINT)
        elif s["kind"] == "waypoint":
            fills[i] = (255, 255, 255)
        else:
            fills[i] = _rgb8(REGION_POWER[s["name"]]["color"])
    rgb = fills[grid]
    bmask = binary_dilation(FractalPartition.boundary_mask(grid),
                            iterations=BORDER_PX // 2)
    rgb[bmask] = _rgb8(ICE)
    ax.imshow(rgb, extent=[0, CANVAS_W, CANVAS_H, 0], origin="upper",
              interpolation="nearest", zorder=0)

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

    # Title, legend, and power key ride ON the map art, haloed for legibility.
    def _halo(fs):
        return [patheffects.withStroke(linewidth=max(2.5, fs / 6), foreground="white")]

    fig.text(0.5, 0.945, "The Frost Kingdom of Girlville",
             fontsize=26, color=SLATE, ha="center", va="center",
             family="serif", weight="bold", path_effects=_halo(26))
    fig.text(0.5, 0.058, "\u25c6 \u2014 supply  \u00b7  \u2605 \u2014 wild  \u00b7  \u25cb \u2014 neutral ground",
             fontsize=24, color=SLATE, ha="center", va="center",
             family="DejaVu Sans", path_effects=_halo(24))
    # Great-power key: one swatch + name per power, slotted across the foot.
    for i, p in enumerate(GREAT_POWERS):
        _x = 0.035 + i * (0.93 / 7)
        fig.text(_x, 0.02, "\u25a0", fontsize=16, color=p["color"],
                 ha="left", va="center", family="DejaVu Sans",
                 path_effects=_halo(16))
        fig.text(_x + 0.016, 0.02, p["short"], fontsize=16, color=SLATE,
                 ha="left", va="center", family="DejaVu Sans",
                 path_effects=_halo(16))
    del i, p, _x

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
    """Typed multiplex graph G = (V, E_A, E_F, E_C) from the fractal raster.

    Raw adjacency = shared fractal borders, read off the SAME raster the
    render paints (skill rule: the graph and the picture must agree).
    E_A: borders between non-sea cells. E_F: borders touching a sea cell.
    E_C: authored only — Girlville has no portals/convoy routes yet, so it
    stays empty.
    """
    names = [s["name"] for s in SEEDS]
    borders = [(names[i], names[j])
               for i, j in sorted(adjacency_from_grid(get_partition()))]

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
                    "fx": s["fx"], "fy": s["fy"],
                    "power": (REGION_POWER[s["name"]]["name"]
                              if s["kind"] == "supply" else None)}
                   for s in SEEDS]}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    powers = [{"name": p["name"], "short": p["short"],
               "princes": p["princes"], "color": p["color"],
               "regions": p["regions"]} for p in GREAT_POWERS]
    with open(OUT_POWERS, "w", encoding="utf-8") as f:
        json.dump(powers, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_POWERS} ({len(powers)} great powers)")

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
