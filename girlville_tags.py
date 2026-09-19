"""Elemental tag profile for each gazetteer location in Girlville.

Tags (Ice, Rot, Charm) hang thick or thin on locations. Where a tag hangs
thick, dice carrying it step up once; where it hangs thin, they step down
once. Locations not listed here hang even -- no effect.

This module is the single source of truth for the tag lines rendered into
the gazetteer by apply_tags.py. check_balance() is the gate: it fails loudly
if the tags stop being balanced across the three elements.
"""

VALID_TAGS = ("Ice", "Rot", "Charm")

# name -> {"thick": [...], "thin": [...]}
LOCATION_TAGS = {
    "Frostgate":                    {"thick": ["Charm"], "thin": []},
    "Mines of Glimm\u014dr":        {"thick": ["Ice"],   "thin": ["Charm"]},
    "The Coven Below the Frost":    {"thick": ["Rot"],   "thin": []},
    "The Grand Frost Ballroom":     {"thick": ["Charm"], "thin": []},
    "Underthrone":                  {"thick": ["Rot"],   "thin": []},
    "Federwald":                     {"thick": ["Ice"],   "thin": []},
    "Lacefield":                    {"thick": ["Charm"], "thin": []},
    "The Coronet":                  {"thick": ["Charm"], "thin": []},
    "Snow fortress":                {"thick": ["Ice"],   "thin": ["Rot"]},
    "The Blizzard":                 {"thick": ["Ice"],   "thin": []},
    "The Deep Dark":                {"thick": ["Rot"],   "thin": ["Charm"]},
    "Snowmere":                     {"thick": ["Ice"],   "thin": ["Rot"]},
    "The Sewers":                   {"thick": ["Rot"],   "thin": ["Ice"]},
}


def _norm(name):
    # the manuscript encodes \u014d as an HTML entity
    return name.replace("\u014d", "&#x14d;")


# gazetteer entry titles that differ from the canonical region name
ENTRY_TITLES = {
    "Mines of Glimm\u014dr": "The Mines of Glimm\u014dr",
    "Underthrone": "The Underthrone",
}


def entry_title(region_name):
    return _norm(ENTRY_TITLES.get(region_name, region_name))


def normalized():
    return {_norm(k): v for k, v in LOCATION_TAGS.items()}


def format_tagline(name):
    """Render the gazetteer tag line for a location (canonical region name),
    e.g. '<b>Ice</b> hangs thick; <b>Rot</b> hangs thin.'"""
    entry = LOCATION_TAGS[name]
    parts = []
    for key, verb_sg, verb_pl in (("thick", "hangs thick", "hang thick"),
                                  ("thin", "hangs thin", "hang thin")):
        tags = entry[key]
        if not tags:
            continue
        bolded = " and ".join(f"<b>{t}</b>" for t in tags)
        verb = verb_sg if len(tags) == 1 else verb_pl
        parts.append(f"{bolded} {verb}")
    return "; ".join(parts) + "."


def check_balance():
    """Fail loudly if the tag economy is unbalanced. Returns a report dict."""
    errors = []
    tags = normalized()

    for name, entry in tags.items():
        for key in ("thick", "thin"):
            for t in entry[key]:
                if t not in VALID_TAGS:
                    errors.append(f"{name}: invalid tag {t!r}")
        overlap = set(entry["thick"]) & set(entry["thin"])
        if overlap:
            errors.append(f"{name}: tag(s) both thick and thin: {sorted(overlap)}")

    # every tagged name must be a real map region
    try:
        from girlville_data import REGIONS
        region_names = {_norm(n) for n, _, _, _ in REGIONS}
        for name in tags:
            if name not in region_names:
                errors.append(f"{name}: not a region in girlville_data.REGIONS")
    except ImportError:
        pass  # cross-check skipped when run outside the project dir

    thick_counts = {t: 0 for t in VALID_TAGS}
    thin_counts = {t: 0 for t in VALID_TAGS}
    for entry in tags.values():
        for t in entry["thick"]:
            thick_counts[t] += 1
        for t in entry["thin"]:
            thin_counts[t] += 1

    for t in VALID_TAGS:
        if thick_counts[t] < 3:
            errors.append(
                f"{t} hangs thick only {thick_counts[t]}x (need >= 3)")
    if max(thick_counts.values()) - min(thick_counts.values()) > 2:
        errors.append(
            f"thick counts out of parity: {thick_counts}")
    for t in VALID_TAGS:
        if thin_counts[t] < 1:
            errors.append(
                f"{t} never hangs thin (every tag needs a hostile ground)")

    report = {"thick": thick_counts, "thin": thin_counts,
              "locations": len(tags)}
    if errors:
        raise ValueError("tag balance FAILED:\n  " + "\n  ".join(errors))
    return report


if __name__ == "__main__":
    report = check_balance()
    print(f"tag balance OK: {report['locations']} locations, "
          f"thick {report['thick']}, thin {report['thin']}")
