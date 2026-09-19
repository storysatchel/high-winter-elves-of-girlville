"""Verify gate for the Girlville map build. Exits non-zero on ANY failure.

Checks (text-source analog of the skill's headless verify):
  1. every seed is within [0,1]
  2. every seed name appears in manuscript.html (entities normalized, case-insensitive)
  3. every kind is a valid kind
  4. the summary JSON (if it exists) lists all seeds
  5. pairwise minimum separation >= POISSON_R (the even-distribution property
     that replaces the hand-placement claim)
  6. label layout (matplotlib text bboxes): no label overlaps another label
     or any marker, and no label runs outside the map frame

Run BEFORE every render. Never render on a failed verify.
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from girlville_data import SEEDS, KINDS, POISSON_R  # noqa: E402

MANUSCRIPT = os.path.join(HERE, "manuscript.html")
SUMMARY = os.path.join(HERE, "maps", "girlville-summary.json")

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


# 1. seeds within [0,1]
for s in SEEDS:
    for axis in ("fx", "fy"):
        v = s[axis]
        if not (0.0 <= v <= 1.0):
            fail(f"{s['name']}: {axis}={v} outside [0,1]")

# 2. every name appears in the manuscript
html = open(MANUSCRIPT, encoding="utf-8").read()
html = html.replace("&#x14d;", "ō").replace("&#x14D;", "Ō")
text = re.sub(r"<[^>]+>", " ", html).lower()
for s in SEEDS:
    if s["name"].lower() not in text:
        fail(f"{s['name']!r} not found in manuscript.html")

# 3. kinds valid
for s in SEEDS:
    if s["kind"] not in KINDS:
        fail(f"{s['name']}: invalid kind {s['kind']!r}")

# 4. summary JSON lists all seeds (if it exists yet)
if os.path.exists(SUMMARY):
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    names = {r["name"] for r in summary.get("regions", [])}
    for s in SEEDS:
        if s["name"] not in names:
            fail(f"{s['name']!r} missing from summary JSON")
else:
    print("(summary JSON not present yet — skipping completeness check)")

kinds = {}
for s in SEEDS:
    kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
print(f"checked {len(SEEDS)} seeds: {kinds}")

# 5. Poisson-disc minimum separation
min_d2 = None
min_pair = None
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        d2 = ((SEEDS[i]["fx"] - SEEDS[j]["fx"]) ** 2 +
              (SEEDS[i]["fy"] - SEEDS[j]["fy"]) ** 2)
        if min_d2 is None or d2 < min_d2:
            min_d2, min_pair = d2, (SEEDS[i]["name"], SEEDS[j]["name"])
min_d = math.sqrt(min_d2)
print(f"min pairwise separation: {min_d:.4f} "
      f"(r={POISSON_R}) between {min_pair[0]!r} and {min_pair[1]!r}")
if min_d < POISSON_R - 1e-9:
    fail(f"separation {min_d:.4f} < r={POISSON_R}: "
         f"{min_pair[0]!r} vs {min_pair[1]!r}")

# 6. label placement: uniform rule, no overlaps, inside the frame
# (boxes are in canvas inches now; the 17x11 spread frame)
from girlville_pipeline import get_label_bboxes, CANVAS_W, CANVAS_H  # noqa: E402

LABEL_PAD = 0.05  # small padding around each box, in inches


def _overlaps(a, b):
    return (a[0] < b[2] + LABEL_PAD and b[0] < a[2] + LABEL_PAD and
            a[1] < b[3] + LABEL_PAD and b[1] < a[3] + LABEL_PAD)


label_boxes, marker_boxes = get_label_bboxes()
for i in range(len(label_boxes)):
    for j in range(i + 1, len(label_boxes)):
        if _overlaps(label_boxes[i], label_boxes[j]):
            fail(f"label overlap: {SEEDS[i]['name']!r} vs {SEEDS[j]['name']!r}")
    for j in range(len(marker_boxes)):
        if _overlaps(label_boxes[i], marker_boxes[j]):
            fail(f"label {SEEDS[i]['name']!r} overlaps "
                 f"marker of {SEEDS[j]['name']!r}")
    x0, y0, x1, y1 = label_boxes[i]
    if not (0.0 <= x0 and x1 <= CANVAS_W and 0.0 <= y0 and y1 <= CANVAS_H):
        fail(f"label {SEEDS[i]['name']!r} outside frame: "
             f"({x0:.4f}, {y0:.4f}, {x1:.4f}, {y1:.4f})")
print(f"checked {len(label_boxes)} label boxes: no overlaps, all inside frame")

if failures:
    print(f"{len(failures)} FAILURES — do not render")
    sys.exit(1)
print("verify OK")
