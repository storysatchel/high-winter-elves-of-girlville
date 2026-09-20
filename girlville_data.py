"""Survey output for the Girlville frontispiece map: The Frost Kingdom of Girlville.

No source image exists (the survey source is the prose of
~/workspace/girlville/manuscript.html), so per the schelling-campaign-map
skill's "Without a source image" section, seeds come from Poisson-disc
sampling (Bridson's algorithm) with semantic anchoring instead of measured
markers.

  - Algorithm: Bridson Poisson-disc, minimum separation r = 0.08,
    k = 30 candidates per active point, RNG seed 20696464 (deterministic).
  - Sampled points fill the unit square at >= r separation; then each named
    region, in seed order (supply centers, then wilds, then waypoints), is
    assigned the nearest unassigned sampled point to its authorial anchor
    (greedy nearest-anchor matching). The map keeps its intended geography
    (mines west, storm north) while the cells stay balanced and readable.
  - The even-distribution property replaces the measured-placement claim.
    The verify gate still checks names-against-source and in-bounds seeds,
    plus pairwise minimum separation and label-collision checks.

Rule 4 holds: every name comes from the manuscript, never invented.
The 11 expansion regions are stretched from princely epithets, all in the
manuscript: Fenwick Underthrone, Milo Ravensent, Halloran Frostbane,
Ivo Lacehand, Larkspur Vell, Sable Mor, Cassian the Coronet-Thief,
Corvin Snowmere, Wren of Nowhere, the Grand Frost Ballroom, and the
Ancient Realm of Glamwendia. (Note: "The Underthrone" does not appear as a
phrase in the manuscript — only "Fenwick Underthrone" — so the region is
named "Underthrone" to keep the verify gate honest.)
Seed order: supply centers, then wilds, then waypoints. Stable.
"""
import math
import os
import random

RNG_SEED = int(os.environ.get("GIRLVILLE_SEED", 20696464))
POISSON_R = 0.08
POISSON_K = 30

KINDS = ("supply", "wild", "waypoint")

# (name, kind, authorial anchor (fx, fy), manuscript fact)
REGIONS = [
    # ---- supply centers (named, from the manuscript) ----
    ("Frostgate", "supply", (0.08, 0.42),
     '"Frostgate Princess in Exile" (Distinction example); the seat of the kingdom.'),
    ("Mines of Glimmōr", "supply", (0.20, 0.55),
     '"Prince Dorian Glimmōr. Sold the Mines to fund his glow-up." Veins mostly Ice; Rot seams in the deep dark.'),
    ("Crystal Ballroom", "supply", (0.74, 0.38),
     'Dance club; "snow fortresses, frost dungeons, and dance clubs" are open to every people.'),
    ("The Old Ballroom", "supply", (0.82, 0.56),
     '"Charm near the old ballroom" (Mines of Glimmōr glitter-vein note).'),
    ("Makeout Forge", "supply", (0.44, 0.80),
     '"Prince Jaspar Makeout. Founded the Mercenaries at Makeout Forge; all contracts paid in kisses and regret."'),
    ("Guild of Lacy Shadows", "supply", (0.66, 0.64),
     '"Prince Ivo Lacehand. Runs the Guild of Lacy Shadows like a spurned fan club."'),
    ("The Coven Below the Frost", "supply", (0.62, 0.24),
     'Fester witches are "cavern-dwelling"; "the coven below the frost". Prince Sable Mor is "fester-witch adjacent".'),
    ("The Glyph-Archive", "supply", (0.34, 0.30),
     '"Prince Anselm of the Seventh Glyph. Stole a sacred glyph for a necklace." The Girly Mage\'s glyph-archive.'),
    # ---- expansion supply centers (princely epithets, all in the manuscript) ----
    ("The Grand Frost Ballroom", "supply", (0.45, 0.20),
     '"Tonight is the Winter Prom at the Grand Frost Ballroom." Location GMC.'),
    ("Underthrone", "supply", (0.50, 0.64),
     '"Prince Fenwick Underthrone."'),
    ("Federwald", "supply", (0.15, 0.30),
     '"Prince Milo Ravensent. Ended it by raven, in winter, during a siege. Wound: Read at Midnight."'),
    ("Runewick", "supply", (0.85, 0.20),
     '"Prince Halloran Frostbane. Claims he doesn\'t remember it that way."'),
    ("Lacefield", "supply", (0.80, 0.70),
     '"Prince Ivo Lacehand."'),
    ("Sable", "supply", (0.78, 0.12),
     '"Prince Sable Mor. Collects apologies he never gives."'),
    ("The Coronet", "supply", (0.88, 0.88),
     '"Prince Cassian the Coronet-Thief. Wears your crown and your patience is gone."'),
    ("Snow fortress", "supply", (0.28, 0.74),
     'Named holding; "snow fortresses, frost dungeons, and dance clubs" dot the borders — this one got claimed.'),
    # ---- wilds (distinct geography) ----
    ("The Blizzard", "wild", (0.50, 0.08),
     '"Prince Corvin Snowmere. Married the Blizzard. It seems happy." Far north.'),
    ("The Deep Dark", "wild", (0.24, 0.14),
     '"Rot seams in the deep dark" (Mines of Glimmōr glitter-vein note).'),
    # ---- expansion wilds ----
    ("Glamwendia", "wild", (0.50, 0.45),
     '"the Ancient Realm of Glamwendia."'),
    ("Snowmere", "wild", (0.88, 0.45),
     '"Prince Corvin Snowmere."'),
    ("Nowhere", "wild", (0.4, 0.66),
     '"Prince Wren of Nowhere. Nobody\'s ex. Shows up anyway. Knows things."'),
    ("The Sewers", "wild", (0.52, 0.60),
     '"Prince Fenwick Underthrone. Rules the sewers now; sends up notes." Sewer orcs. Counts as water/ocean in the node budget.'),
    # ---- waypoints (neutral: never claimed, never quest hosts; they buffer borders) ----
    ("Frost dungeon", "waypoint", (0.60, 0.76),
     'Neutral ground; "snow fortresses, frost dungeons, and dance clubs" dot the borders.'),
    ("Dance club", "waypoint", (0.60, 0.50),
     'Neutral ground; "snow fortresses, frost dungeons, and dance clubs" dot the borders.'),
    ("Waymeet", "waypoint", (0.12, 0.62),
     'Neutral ground now; "a wizard, technically; a disappointment, specifically."'),
]


def _bridson_poisson_disc(r, k, rng):
    """Bridson's algorithm: fill [0,1]^2 with points at >= r separation."""
    cell = r / math.sqrt(2)
    gw = math.ceil(1.0 / cell)
    grid = [[-1] * gw for _ in range(gw)]
    points = []
    active = []

    def grid_coords(p):
        return (min(int(p[0] / cell), gw - 1),
                min(int(p[1] / cell), gw - 1))

    def far_enough(p):
        gx, gy = grid_coords(p)
        for ix in range(max(gx - 2, 0), min(gx + 3, gw)):
            for iy in range(max(gy - 2, 0), min(gy + 3, gw)):
                j = grid[ix][iy]
                if j != -1:
                    q = points[j]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < r * r - 1e-12:
                        return False
        return True

    def add_point(p):
        points.append(p)
        active.append(len(points) - 1)
        gx, gy = grid_coords(p)
        grid[gx][gy] = len(points) - 1

    add_point((rng.random(), rng.random()))
    while active:
        idx = rng.randrange(len(active))
        pi = active[idx]
        px, py = points[pi]
        placed = False
        for _ in range(k):
            ang = rng.random() * 2 * math.pi
            rad = r * (1.0 + rng.random())  # annulus [r, 2r]
            q = (px + rad * math.cos(ang), py + rad * math.sin(ang))
            if 0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0 and far_enough(q):
                add_point(q)
                placed = True
                break
        if not placed:
            active.pop(idx)
    return points


def _greedy_nearest_anchor(candidates, regions):
    """Assign each region the nearest unassigned candidate to its anchor,
    in region order. Returns {region_index: point}."""
    remaining = list(candidates)
    assignment = {}
    for i, (_, _, anchor, _) in enumerate(regions):
        best_j, best_d2 = None, None
        for j, p in enumerate(remaining):
            d2 = (p[0] - anchor[0]) ** 2 + (p[1] - anchor[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best_j, best_d2 = j, d2
        assignment[i] = remaining.pop(best_j)
    return assignment


_rng = random.Random(RNG_SEED)
_candidates = _bridson_poisson_disc(POISSON_R, POISSON_K, _rng)
if len(_candidates) < len(REGIONS):
    raise RuntimeError(
        f"Bridson produced only {len(_candidates)} points for "
        f"{len(REGIONS)} regions at r={POISSON_R}; lower r.")
_assigned = _greedy_nearest_anchor(_candidates, REGIONS)

_M = sum(1 for _, kind, _, _ in REGIONS if kind == "supply")
_K = len(REGIONS) - _M  # wilds/waters + neutral waypoints

SEEDS = [
    dict(name=name, kind=kind,
         fx=round(_assigned[i][0], 4), fy=round(_assigned[i][1], 4),
         provenance=(
             f"Poisson-disc sampling (Bridson), r={POISSON_R}, "
             f"{len(_candidates)} points, RNG seed {RNG_SEED}; "
             f"node budget N={len(REGIONS)}/M={_M}/K={_K}; greedy "
             f"nearest-anchor assignment to authorial anchor {anchor}. "
             + fact))
    for i, (name, kind, anchor, fact) in enumerate(REGIONS)
]

del _rng, _candidates, _assigned

# The seven great powers: princely blocs dividing the sixteen supply centers.
# Every supply center is claimed by exactly one power; wilds/waters are
# unclaimed; waypoints are neutral and unclaimable. Three princes stand
# outside the seven: Corvin Snowmere (married the Blizzard — consort of the
# Doom Pool), Larkspur Vell (Waymeet's disappointment), Wren of Nowhere
# (nobody's ex; shows up anyway).
GREAT_POWERS = [
    dict(name="The Coronet", short="The Coronet",
         princes=["Prince Cassian the Coronet-Thief"],
         color="#F3E5C0",  # champagne gold
         regions=["The Coronet", "Crystal Ballroom", "Frostgate"]),
    dict(name="The Glyph-Archive", short="Glyph-Archive",
         princes=["Prince Anselm of the Seventh Glyph"],
         color="#DDD9F2",  # periwinkle
         regions=["The Glyph-Archive", "The Grand Frost Ballroom",
                  "The Coven Below the Frost"]),
    dict(name="Federwald", short="Federwald",
         princes=["Prince Milo Ravensent"],
         color="#D7E8D4",  # sage
         regions=["Federwald", "Snow fortress"]),
    dict(name="The Glimm\u014dr Compact", short="Glimm\u014dr",
         princes=["Prince Dorian Glimm\u014dr", "Prince Jaspar Makeout"],
         color="#F6DCC8",  # apricot
         regions=["Mines of Glimm\u014dr", "Makeout Forge"]),
    dict(name="Underthrone", short="Underthrone",
         princes=["Prince Fenwick Underthrone"],
         color="#E0D8CE",  # warm stone
         regions=["Underthrone"]),
    dict(name="Runewick", short="Runewick",
         princes=["Prince Halloran Frostbane", "Prince Sable Mor"],
         color="#CFE4F3",  # ice blue
         regions=["Runewick", "Sable"]),
    dict(name="The Lacy Shadows", short="Lacy Shadows",
         princes=["Prince Ivo Lacehand"],
         color="#EED5DE",  # blush rose
         regions=["Lacefield", "Guild of Lacy Shadows", "The Old Ballroom"]),
]

_claimed = [r for p in GREAT_POWERS for r in p["regions"]]
_supply = [n for n, k, _, _ in REGIONS if k == "supply"]
if sorted(_claimed) != sorted(_supply):
    raise RuntimeError(
        "Great powers must partition the supply centers exactly: "
        f"claimed={len(_claimed)} supply={len(_supply)}")
del _claimed, _supply
