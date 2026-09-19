#!/usr/bin/env python3
"""Render the gazetteer tag lines in manuscript.html from girlville_tags.py.

The tag data module is the single source of truth. This script:
  1. runs the balance gate (refuses to render on failure),
  2. strips any existing <span class="ptags"> lines from gazetteer entries,
  3. re-inserts them procedurally from the data.

Idempotent: run it any time the tag data changes, then push manuscript.html.
"""
import re
import sys

import girlville_tags as T

MANUSCRIPT = "manuscript.html"


def main():
    report = T.check_balance()  # gate: raises on imbalance

    t = open(MANUSCRIPT).read()
    g_start = t.index("Appendix: Gazetteer")
    g_end = t.index("BACK COVER")
    head, gaz, tail = t[:g_start], t[g_start:g_end], t[g_end:]

    # 1. strip existing tag lines (procedural render owns them now)
    gaz = re.sub(r' <span class="ptags">.*?</span>', "", gaz)

    # 2. re-insert from data, matched on gazetteer entry title
    want = {T.entry_title(raw): raw for raw in T.LOCATION_TAGS}
    done = []

    def repl(m):
        title, body = m.group(1), m.group(2)
        if title in want:
            region = want[title]
            line = T.format_tagline(region)
            done.append(region)
            return (f'<p><b>{title}.</b>{body} '
                    f'<span class="ptags">{line}</span></p>')
        return m.group(0)

    gaz = re.sub(r"<p><b>([^<]+)\.</b>(.*?)</p>", repl, gaz, flags=re.S)

    missing = set(T.LOCATION_TAGS) - set(done)
    if missing:
        sys.exit(f"apply_tags FAILED: no gazetteer entry for {sorted(missing)}")

    open(MANUSCRIPT, "w").write(head + gaz + tail)
    print(f"rendered {len(done)} tag lines into {MANUSCRIPT} "
          f"(thick {report['thick']}, thin {report['thin']})")


if __name__ == "__main__":
    main()
