"""
check_bdd_lanes.py — verify a downloaded BDD100K really contains what we need
BEFORE any training code is written against it.

WHY THIS EXISTS
    The lane model needs one specific thing: a per-line SOLID vs DASHED label.
    Most public lane datasets do not have it (TuSimple, CULane label WHERE the
    lanes are, not WHAT KIND of line), and many BDD100K mirrors ship only the
    images, or only the detection / drivable-area labels.

    Discovering that after a multi-GB download and a day of loader code is a bad
    way to spend a day. This reads the labels and tells you in about a minute.

USAGE
    python check_bdd_lanes.py <path-to-bdd100k-root>

WHAT GOOD LOOKS LIKE
    A non-zero count for BOTH 'solid' and 'dashed', and a healthy number of
    images with no lane at all — those become our 'absent' class, and they are
    the images that teach the model to stay quiet rather than guess.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def find_label_files(root: Path) -> list[Path]:
    """Any JSON that plausibly holds labels. Mirrors disagree on layout, so we
    search rather than assume a path."""
    out = []
    for p in root.rglob("*.json"):
        name = p.name.lower()
        if p.stat().st_size < 10_000:
            continue  # config/readme sized, not an annotation set
        if "lane" in name or "label" in name or "polygon" in name or "det" in name:
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_size, reverse=True)


def walk_for_lanes(node, styles: Counter, types: Counter, seen: list):
    """BDD100K has shipped several nestings. Rather than hardcode one, walk the
    tree and pick up anything carrying a laneStyle attribute."""
    if isinstance(node, dict):
        attrs = node.get("attributes")
        cat = str(node.get("category", "")).lower()
        if isinstance(attrs, dict) and ("laneStyle" in attrs or "laneDirection" in attrs):
            styles[str(attrs.get("laneStyle", "?")).lower()] += 1
            types[str(attrs.get("laneType", "?")).lower()] += 1
            seen.append(True)
        elif "lane" in cat and isinstance(attrs, dict):
            styles["<lane, no laneStyle>"] += 1
        for v in node.values():
            walk_for_lanes(v, styles, types, seen)
    elif isinstance(node, list):
        for v in node:
            walk_for_lanes(v, styles, types, seen)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"path does not exist: {root}")
        return 2

    imgs = sum(1 for _ in root.rglob("*.jpg"))
    print(f"root          : {root}")
    print(f"jpg images    : {imgs}")

    mask_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name == "masks"]
    if mask_dirs:
        print(f"mask dirs     : {[str(p.relative_to(root)) for p in mask_dirs][:3]}")

    labels = find_label_files(root)
    if not labels:
        print("\nNO LABEL JSON FOUND.")
        print("This mirror is images-only. You need the lane annotation set too.")
        return 1

    print(f"label files   : {len(labels)} (checking the largest few)\n")

    styles: Counter = Counter()
    types: Counter = Counter()
    for lf in labels[:3]:
        print(f"  reading {lf.relative_to(root)} ({lf.stat().st_size/1e6:.0f} MB) ...")
        try:
            with open(lf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"    could not parse ({type(e).__name__}) - skipping")
            continue
        walk_for_lanes(data, styles, types, [])

    print("\n--- laneStyle counts ---")
    if not styles:
        print("NONE FOUND.")
        print("\nVERDICT: unusable for solid-vs-dashed. The lanes here (if any)")
        print("say WHERE, not WHAT KIND. Find a mirror with the lane attribute")
        print("labels, or switch to ApolloScape lane segmentation, whose class")
        print("names encode it directly (s_w_s = solid white, s_w_d = dashed).")
        return 1

    for k, v in styles.most_common():
        print(f"  {k:24s} {v}")

    if types:
        print("\n--- laneType counts (top 8) ---")
        for k, v in types.most_common(8):
            print(f"  {k:24s} {v}")

    solid = styles.get("solid", 0)
    dashed = styles.get("dashed", 0)
    print("\n--- verdict ---")
    if solid and dashed:
        print(f"USABLE. solid={solid}, dashed={dashed}")
        print("Map straight onto the existing scheme: dashed->1 (broken), solid->2.")
        print("Keep the lane-free images; they are the 'absent' class (0).")
        return 0

    print(f"INCOMPLETE. solid={solid}, dashed={dashed}")
    print("Both must be non-zero to train the solid/broken head.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
