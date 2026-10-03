"""
bdd_to_ufld.py — turns BDD100K's lane polylines into the row-anchor targets
UFLD trains on, plus the solid/dashed label UFLD does not natively carry.

WHAT UFLD WANTS
    The image is sliced into a fixed set of horizontal ROW ANCHORS. Each
    anchor is divided into GRIDING_NUM cells across. For every (lane slot,
    anchor) the network answers a single multiple-choice question:
        "which cell is this lane in, or is it absent here?"
    So a target is an integer per (slot, anchor): 0..GRIDING_NUM-1 for a
    cell, or GRIDING_NUM meaning ABSENT. That absent class is not padding -
    it is the mechanism that lets the model decline to answer, which is the
    whole point of the silence requirement.

WHY THIS FILE IS NOT A TEN-LINE FORMAT SHUFFLE
    BDD does not come in a shape UFLD understands, and three specific
    mismatches will each silently corrupt training if not handled. All three
    numbers below were measured off the val split before this was written,
    not guessed.

    1. HALF THE "lane" LABELS ARE NOT LANE LINES.
       In val: single white 35452, double yellow 5426, single yellow 2817,
       double white 831 - but also road curb 15821 and crosswalk 15342.
       Curbs and crosswalks are ~41% of everything tagged "lane". Trained on
       as-is, the app announces "crossed solid line" at every zebra crossing.
       Dropped by laneType, and again by laneDirection != parallel.

    2. EACH PAINTED STRIPE IS ANNOTATED TWICE.
       BDD traces both EDGES of a marking, so one lane line arrives as two
       near-parallel polylines. Measured gap between adjacent lines at
       y=600: p05=15px, p10=18px, p25=23px, then median 36px and p75=306px.
       That bimodal split is the tell - under DOUBLE_MERGE_PX is one stripe
       seen twice, above it is a genuinely different lane. Without merging,
       "nearest line on my left" is actually the far edge of the same stripe
       and every slot assignment is off by one.

    3. THE LABELS DO NOT REACH THE BOTTOM OF THE IMAGE.
       Lowest point of a lane line: p25=457, median=545, p75=616, p90=677
       (image is 720 tall). Fraction of lane-bearing images with a line
       crossing a given row peaks at 75% around y=500 and falls to 14% by
       y=700. UFLD's stock TuSimple anchors run y=160..710, which would aim
       a third of the anchors at a band that is nearly always empty - and
       that band is precisely where the car is. So the anchors here cover
       ROW_TOP..ROW_BOTTOM, where the labels actually live, and the
       departure decision extrapolates the fitted lane downward to the
       bumper at inference instead of pretending it was labelled.

LANE SLOTS
    Four slots, ordered left to right around the camera, CULane style:
        0 = second lane line to the left    2 = nearest line to the right
        1 = nearest line to the left        3 = second line to the right
    Slots 1 and 2 bound the ego lane and are the only two the departure
    warning consults. Assignment assumes the dashcam sits near the vehicle
    centreline, so image centre is the car - true enough for BDD.

STYLE HEAD
    Per slot: 0 absent, 1 dashed, 2 solid. Deliberately the same vocabulary
    scnn_lane/dataset.py already uses (0 absent, 1 broken, 2 solid) so the
    downstream alert logic does not need a second mapping.

USAGE
    python bdd_to_ufld.py --data d:/adas/data --split val   --out d:/adas/data/ufld
    python bdd_to_ufld.py --data d:/adas/data --split train --out d:/adas/data/ufld
    python bdd_to_ufld.py --data d:/adas/data --split val --limit 40 --debug-vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

# --- geometry of the source data -------------------------------------------
IMG_W, IMG_H = 1280, 720

# Row anchors. See mismatch (3) above for why these are not UFLD's defaults.
ROW_TOP, ROW_BOTTOM, ROW_STEP = 380, 620, 5
ROW_ANCHORS = list(range(ROW_TOP, ROW_BOTTOM + 1, ROW_STEP))   # 49 anchors

GRIDING_NUM = 100          # cells across; index GRIDING_NUM means ABSENT
ABSENT = GRIDING_NUM

NUM_LANES = 4
CAR_X = IMG_W / 2.0        # dashcam assumed near the centreline

# --- when are two polylines the same physical line? See mismatch (2). -------
#
# The obvious test - horizontal gap at a shared row anchor - is WRONG, and
# wrong in a way that looks right on the frames you happen to measure. A
# stripe's two traced edges sit a fixed PERPENDICULAR distance apart. Convert
# that to a horizontal gap and it explodes as the line flattens:
#
#     line 80 deg from horizontal (near, steep) : 12px perp -> 12px across
#     line 20 deg from horizontal (side of road): 12px perp -> 35px across
#
# The p05=15px / p25=23px figures measured at y=600 sampled mostly steep
# near lines, so a flat 25px threshold looked generous. It silently fails on
# every shallow line - which is most of them once the road bends or the
# marking is off to one side. So: measure perpendicular distance.
#
# The allowance also has to shrink with depth, because a stripe subtends
# fewer pixels the further away it is. Linear in row from the vanishing
# region works well enough and needs no camera model.
# Chosen from the measured distribution, not by eye. Depth-normalised
# perpendicular gap between two polylines in the same frame:
#     same laneType+laneStyle (one marking traced twice) p10 = 12..21 px,
#         tail decaying to ~50
#     different laneType/laneStyle (real neighbouring lanes) p05 = 91.5 px
# so anything under ~50 is one marking and anything over ~90 is two lanes.
# 48 sits in the empty valley between them. Note this must exceed a single
# stripe's width: "double yellow" and "double white" are genuinely TWO
# painted stripes forming ONE lane boundary, and both must fold together.
MERGE_PERP_AT_BOTTOM = 48.0   # px allowance at ROW_BOTTOM
VANISH_Y = 340.0              # rows converge towards here; just above ROW_TOP
MIN_SHARED_ANCHORS = 2

# Two polylines that never share a row can still be one line cut into
# segments (very common - BDD annotators break a line at an occlusion). They
# are joined by extrapolating each fit across the gap, but only across a
# short gap, because extrapolation is the least trustworthy thing here.
MAX_JOIN_GAP_ANCHORS = 12

# Row used only for left/right ordering. Lanes are fitted and evaluated here,
# extrapolating when a line stops short, so lines labelled at different
# heights are still comparable.
REF_ROW = float(ROW_BOTTOM)

# laneType values that are not lane boundaries. See mismatch (1).
DROP_TYPES = {"crosswalk", "road curb"}

STYLE_ABSENT, STYLE_DASHED, STYLE_SOLID = 0, 1, 2


# --- polyline handling ------------------------------------------------------

def _bezier(p0, p1, p2, p3, steps=12):
    """BDD marks bezier control points with 'C' in poly2d['types']. Sampling
    them properly matters more than it looks: a curve flattened to its four
    raw vertices can sit tens of pixels off through a bend, which is several
    grid cells of error exactly where the road curves."""
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def densify(poly: dict) -> list[tuple[float, float]]:
    """poly2d -> a dense point list, expanding any cubic bezier segments."""
    verts = [(float(a), float(b)) for a, b in poly["vertices"]]
    types = poly.get("types") or ("L" * len(verts))
    if len(verts) < 2:
        return verts
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(verts) - 1:
        if (types[i] == "L" and i + 3 < len(verts)
                and types[i + 1] == "C" and types[i + 2] == "C" and types[i + 3] == "L"):
            out += _bezier(verts[i], verts[i + 1], verts[i + 2], verts[i + 3])
            i += 3
        else:
            out += [verts[i], verts[i + 1]]
            i += 1
    return out


def xs_at_anchors(pts: list[tuple[float, float]]) -> np.ndarray:
    """x of this polyline at each row anchor, NaN where the line does not
    span that row. Never extrapolates - an unlabelled row must stay absent,
    because inventing a position there is exactly the confident-but-wrong
    frame the alert engine has no debounce against."""
    out = np.full(len(ROW_ANCHORS), np.nan, dtype=np.float64)
    for a, b in zip(pts, pts[1:]):
        y0, y1 = a[1], b[1]
        if y0 == y1:
            continue
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        for k, r in enumerate(ROW_ANCHORS):
            if lo <= r <= hi:
                out[k] = a[0] + (b[0] - a[0]) * (r - y0) / (y1 - y0)
    return out


_ANCHOR_Y = np.asarray(ROW_ANCHORS, dtype=np.float64)


def _fit(xs: np.ndarray):
    """Least-squares x = f(y) through one polyline's anchor samples."""
    ok = ~np.isnan(xs)
    n = int(ok.sum())
    if n == 0:
        return None
    if n == 1:
        return np.array([float(xs[ok][0])])
    deg = 2 if n >= 6 else 1
    try:
        return np.polyfit(_ANCHOR_Y[ok], xs[ok], deg)
    except Exception:
        return np.array([float(xs[ok][-1])])


def _slope(c, y: float) -> float:
    """dx/dy at y. Near 0 for a steep line, large for a shallow one - which
    is exactly the factor the horizontal-gap test was missing."""
    if c is None or len(c) < 2:
        return 0.0
    return float(np.polyval(np.polyder(c), y))


def _perp_allow(y: float) -> float:
    """Permitted perpendicular separation for two traces of one stripe at
    row y. Shrinks towards the vanishing region because a stripe subtends
    fewer pixels the further off it is."""
    return MERGE_PERP_AT_BOTTOM * max(y - VANISH_Y, 1.0) / (ROW_BOTTOM - VANISH_Y)


def _same_line(xa, ca, xb, cb) -> bool:
    both = ~np.isnan(xa) & ~np.isnan(xb)
    if int(both.sum()) >= MIN_SHARED_ANCHORS:
        ys = _ANCHOR_Y[both]
        s = 0.5 * (np.array([_slope(ca, y) for y in ys])
                   + np.array([_slope(cb, y) for y in ys]))
        perp = np.abs(xa[both] - xb[both]) / np.sqrt(1.0 + s * s)
        ratio = perp / np.array([_perp_allow(y) for y in ys])
        # mean stops one noisy anchor splitting a genuine pair; max stops a
        # pair that coincides low down but fans apart higher up - two real
        # lanes converging towards the vanishing point - from being fused.
        return bool(ratio.mean() < 1.0 and ratio.max() < 2.0)

    # No shared rows: possibly one line the annotator cut into segments.
    ia = np.where(~np.isnan(xa))[0]
    ib = np.where(~np.isnan(xb))[0]
    if ia.size == 0 or ib.size == 0 or ca is None or cb is None:
        return False
    if ia[-1] < ib[0]:
        gap, edges = int(ib[0] - ia[-1]), (ia[-1], ib[0])
    elif ib[-1] < ia[0]:
        gap, edges = int(ia[0] - ib[-1]), (ib[-1], ia[0])
    else:
        return False
    if gap > MAX_JOIN_GAP_ANCHORS:
        return False
    for k in edges:
        y = float(_ANCHOR_Y[k])
        s = 0.5 * (_slope(ca, y) + _slope(cb, y))
        perp = abs(float(np.polyval(ca, y)) - float(np.polyval(cb, y))) / np.sqrt(1.0 + s * s)
        if perp > _perp_allow(y):
            return False
    return True


def merge_doubled(lines: list[tuple[np.ndarray, int]]) -> list[tuple[np.ndarray, int]]:
    """Collapse the two traced edges of one painted stripe - and the several
    segments of one interrupted line - into a single line. Averaging a pair
    lands on the stripe centre, which is what a human means by 'the line'."""
    groups: list[list] = []          # [xs, style, coeffs]
    for xs, style in lines:
        c = _fit(xs)
        placed = False
        for g in groups:
            if _same_line(xs, c, g[0], g[2]):
                merged = np.where(np.isnan(g[0]), xs,
                                  np.where(np.isnan(xs), g[0], (g[0] + xs) / 2.0))
                # Both edges of a stripe carry the same laneStyle; where a
                # pair disagrees, solid wins.
                g[0], g[1], g[2] = merged, max(g[1], style), _fit(merged)
                placed = True
                break
        if not placed:
            groups.append([xs.copy(), style, c])
    return [(g[0], g[1]) for g in groups]


def ref_x(xs: np.ndarray) -> float | None:
    """Fitted x at REF_ROW, used only to order lines left-to-right. A line
    labelled high up is extrapolated down so it can be compared with one
    labelled low; this never becomes a training target."""
    c = _fit(xs)
    return None if c is None else float(np.polyval(c, REF_ROW))


def assign_slots(groups: list[tuple[np.ndarray, int]]):
    """Order merged lines into the four ego-relative slots."""
    left, right = [], []
    for xs, style in groups:
        rx = ref_x(xs)
        if rx is None:
            continue
        (left if rx < CAR_X else right).append((rx, xs, style))
    left.sort(key=lambda t: -t[0])    # nearest-to-car first
    right.sort(key=lambda t: t[0])

    grid = np.full((NUM_LANES, len(ROW_ANCHORS)), ABSENT, dtype=np.int16)
    styles = np.full(NUM_LANES, STYLE_ABSENT, dtype=np.int8)

    for slot, src in ((1, left), (0, left), (2, right), (3, right)):
        idx = 0 if slot in (1, 2) else 1
        if len(src) <= idx:
            continue
        _, xs, style = src[idx]
        ok = ~np.isnan(xs)
        cells = np.clip((xs[ok] / IMG_W * GRIDING_NUM).astype(np.int32),
                        0, GRIDING_NUM - 1)
        grid[slot, np.where(ok)[0]] = cells.astype(np.int16)
        styles[slot] = style
    return grid, styles


# --- driver -----------------------------------------------------------------

def index_images(img_dir: str) -> dict[str, str]:
    """Map bare filename -> path relative to img_dir.

    WHY NOT just join(img_dir, name)
        The labels reference images by bare filename, and the official BDD
        layout is one flat directory per split, so a join is the obvious
        thing to write. This Kaggle mirror is not the official layout: of
        the 70000 train images only 1156 sit loose in train/, the rest are
        buried in train/trainA (37216), train/trainB (24750), train/testA
        (4130) and train/testB (2748). A flat join finds the 1156, reports
        a cheerful success, and silently trains on 1.7% of the dataset.
        Walking once and indexing costs a second and cannot be fooled by
        whatever nesting the next mirror invents."""
    index: dict[str, str] = {}
    dupes = 0
    for root, _dirs, files in os.walk(img_dir):
        for f in files:
            if not f.lower().endswith(".jpg"):
                continue
            if f in index:
                dupes += 1
                continue
            index[f] = os.path.relpath(os.path.join(root, f), img_dir)
    if dupes:
        print(f"  WARNING: {dupes} duplicate basenames, kept first of each")
    return index


def convert(label_path: str, img_dir: str, index: dict[str, str], limit: int | None):
    with open(label_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if limit:
        data = data[:limit]

    names, grids, styles_all = [], [], []
    stats = Counter()
    t0 = time.time()

    for i, im in enumerate(data, 1):
        name = im.get("name")
        if not name:
            continue
        rel = index.get(name)
        if rel is None:
            stats["image missing on disk"] += 1
            continue

        lines = []
        for obj in im.get("labels") or []:
            if obj.get("category") != "lane":
                continue
            attrs = obj.get("attributes") or {}
            if attrs.get("laneDirection") != "parallel":
                stats["dropped: not parallel"] += 1
                continue
            if str(attrs.get("laneType", "")).lower() in DROP_TYPES:
                stats["dropped: curb/crosswalk"] += 1
                continue
            style = (STYLE_SOLID if str(attrs.get("laneStyle", "")).lower() == "solid"
                     else STYLE_DASHED)
            for p in obj.get("poly2d") or []:
                pts = densify(p)
                if len(pts) < 2:
                    continue
                xs = xs_at_anchors(pts)
                if np.isnan(xs).all():
                    stats["dropped: outside anchor band"] += 1
                    continue
                lines.append((xs, style))

        merged = merge_doubled(lines)
        stats["lines before merge"] += len(lines)
        stats["lines after merge"] += len(merged)

        grid, st = assign_slots(merged)
        # Store the path relative to the split root, not the bare name, so
        # the loader never has to rediscover this mirror's nesting.
        names.append(rel.replace("\\", "/"))
        grids.append(grid)
        styles_all.append(st)

        n_present = int((st != STYLE_ABSENT).sum())
        stats[f"images with {n_present} lane slot(s)"] += 1
        if st[1] != STYLE_ABSENT and st[2] != STYLE_ABSENT:
            stats["images with BOTH ego lines"] += 1

        if i % 5000 == 0:
            print(f"  {i}/{len(data)}  ({time.time()-t0:.0f}s)", flush=True)

    return names, np.stack(grids), np.stack(styles_all), stats


def debug_vis(names, grids, styles, img_dir, out_dir, n=12):
    """Draw the decoded targets back onto the images.

    A converter can be wrong in a way that every summary statistic still
    looks healthy - slots swapped, anchors off by a row, merge too greedy.
    Looking at twelve pictures catches all three in a minute."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    colours = {0: (255, 160, 0), 1: (0, 255, 0), 2: (0, 200, 255), 3: (255, 0, 255)}
    written = 0
    for k in range(len(names)):
        if written >= n:
            break
        if (styles[k] != STYLE_ABSENT).sum() < 2:
            continue
        img = cv2.imread(os.path.join(img_dir, names[k]))
        if img is None:
            continue
        for slot in range(NUM_LANES):
            for ai, r in enumerate(ROW_ANCHORS):
                c = grids[k, slot, ai]
                if c == ABSENT:
                    continue
                x = int((c + 0.5) / GRIDING_NUM * IMG_W)
                cv2.circle(img, (x, r), 3, colours[slot], -1)
            lab = {0: "absent", 1: "dashed", 2: "solid"}[int(styles[k, slot])]
            cv2.putText(img, f"slot{slot}:{lab}", (10, 30 + 26 * slot),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colours[slot], 2)
        cv2.line(img, (int(CAR_X), IMG_H), (int(CAR_X), ROW_TOP), (60, 60, 60), 1)
        cv2.imwrite(os.path.join(out_dir, f"check_{written:02d}.jpg"), img)
        written += 1
    print(f"wrote {written} check images -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root holding bdd100k/ and bdd100k_labels_release/")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug-vis", action="store_true")
    args = ap.parse_args()

    label_path = os.path.join(args.data, "bdd100k_labels_release", "bdd100k",
                              "labels", f"bdd100k_labels_images_{args.split}.json")
    img_dir = os.path.join(args.data, "bdd100k", "bdd100k", "images", "100k", args.split)

    for p in (label_path, img_dir):
        if not os.path.exists(p):
            print(f"missing: {p}\nRun extract_bdd.py first.")
            return 2

    print(f"anchors : {len(ROW_ANCHORS)} rows, y={ROW_TOP}..{ROW_BOTTOM} step {ROW_STEP}")
    print(f"grid    : {GRIDING_NUM} cells (+1 absent), {NUM_LANES} lane slots")
    print(f"reading : {label_path}")

    print("indexing images ...")
    index = index_images(img_dir)
    print(f"  found {len(index)} jpgs under {img_dir}")

    names, grids, styles, stats = convert(label_path, img_dir, index, args.limit)

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"{args.split}.npz")
    np.savez_compressed(
        dest,
        names=np.array(names),
        grids=grids,
        styles=styles,
        row_anchors=np.array(ROW_ANCHORS),
        griding_num=GRIDING_NUM,
        num_lanes=NUM_LANES,
        img_wh=np.array([IMG_W, IMG_H]),
    )

    print(f"\nwrote {dest}  ({len(names)} images)")
    print("\n--- conversion stats ---")
    for k in sorted(stats):
        print(f"  {k:34s} {stats[k]}")

    filled = int((grids != ABSENT).sum())
    total = grids.size
    print(f"\n  anchor cells filled              {filled}/{total} ({100*filled/total:.1f}%)")
    print(f"  (the rest are the ABSENT class - that is expected, not a bug)")

    if args.debug_vis:
        debug_vis(names, grids, styles, img_dir, os.path.join(args.out, "check"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
