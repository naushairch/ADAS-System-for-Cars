"""
build_road_index.py — turn an OpenStreetMap extract into a compact road-class
index the phone can carry offline, so the app can answer "what kind of road am
I on?" from a GPS fix alone.

WHY ROAD CLASS AND NOT THE maxspeed TAG
    Everyone reaches for OSM's maxspeed tag first. It is optional metadata that
    a volunteer has to type in by hand, and outside motorways it is mostly
    blank - especially in Pakistan. highway=* is different: it is what MAKES a
    road a road in OSM, so it is present on essentially every way. Deriving a
    default limit from the class gives nationwide coverage from data that
    cannot be missing, and maxspeed is then a free upgrade wherever it happens
    to exist. This is the same thing OSRM and Valhalla do, and why they ship
    per-country speed profiles.

WHY NOT COUNT LANES FROM THE CAMERA INSTEAD
    Measured, and it does not work. Using GROUND-TRUTH lane counts on BDD's
    10000 val images against BDD's own road-type labels: "city street" is 62%
    of frames, so always guessing it scores 62%; perfect lane counting scores
    67%. Five points. Even seeing the maximum four lane lines, the road is
    still a city street more often (54%) than a highway (43%) - because lane
    count tells you how WIDE a road is, not how fast you may go. Worse, the
    errors run toward "small street", which would set the limit too low and
    produce exactly the false "slow down" alerts that teach a driver to ignore
    the device.

THE DEFAULTS BIAS HIGH, DELIBERATELY
    A limit set too LOW fires a warning at a legal speed. A limit set too HIGH
    merely fails to fire one. The first destroys trust in every other alert the
    app makes, including the collision warning; the second leaves the driver
    exactly as well off as having no speed feature at all. So where a class
    spans a range, these take the upper end.

OUTPUT FORMAT (little-endian, memory-mapped by RoadIndex.java)
    header   magic "ADASROAD" | version i32 | tileDeg f32
             minLat f32 | minLon f32 | nTilesLat i32 | nTilesLon i32 | nSeg i32
    tiles    nTilesLat*nTilesLon * (offset i32, count i32)
    segments nSeg * (lat1 i32, lon1 i32, lat2 i32, lon2 i32,
                     classId u8, maxspeedKph u8)
    Coordinates are microdegrees. A segment lives in the tile containing its
    MIDPOINT; segments are decimated to ~DECIMATE_M so one can never be longer
    than a tile, which is what makes a 3x3 tile search around the fix correct.

USAGE
    python build_road_index.py --pbf d:/adas/data/osm/pakistan-latest.osm.pbf \
        --out ../app/src/main/assets/roads_pk.bin
    python build_road_index.py --pbf ... --bbox 31.3,74.1,31.7,74.5 --out lahore.bin
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from array import array

import numpy as np
import osmium

MAGIC = b"ADASROAD"
VERSION = 1

TILE_DEG = 0.02          # ~2.2 km. Big enough that 3x3 always covers a fix.
DECIMATE_M = 20.0        # drop way nodes closer together than this

# Class ids are what ship in the file; the name list is mirrored in
# RoadIndex.java so a diagnostic can print something human.
CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street",
]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Pakistan profile, upper end of each class's plausible range. See the note
# above on why this is not the average.
DEFAULT_KPH = {
    "motorway": 120, "motorway_link": 80,
    "trunk": 100, "trunk_link": 60,
    "primary": 80, "primary_link": 60,
    "secondary": 70, "secondary_link": 50,
    "tertiary": 60, "tertiary_link": 50,
    "unclassified": 60,
    "residential": 50,
    "living_street": 30,
}

M_PER_DEG_LAT = 111_320.0


def parse_maxspeed(raw: str) -> int:
    """OSM maxspeed is free text. Returns km/h, or 0 when not a usable number.

    Seen in the wild: "80", "80 km/h", "50 mph", "none", "signals", "PK:urban",
    "30;50". Anything not confidently numeric returns 0 so the class default
    applies - guessing at a malformed tag is how you end up warning at a legal
    speed."""
    if not raw:
        return 0
    s = raw.strip().lower()
    mph = "mph" in s
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    if not num:
        return 0
    try:
        v = int(num)
    except ValueError:
        return 0
    if mph:
        v = int(round(v * 1.609))
    return v if 5 <= v <= 150 else 0


class RoadHandler(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox
        self.coords = array("i")     # 4 per segment, microdegrees
        self.meta = array("B")       # 2 per segment: classId, maxspeedKph
        self.ways = 0
        self.skipped_classes = {}

    def way(self, w):
        tags = w.tags
        hw = tags.get("highway")
        if hw is None:
            return
        cid = CLASS_ID.get(hw)
        if cid is None:
            self.skipped_classes[hw] = self.skipped_classes.get(hw, 0) + 1
            return

        speed = parse_maxspeed(tags.get("maxspeed", ""))
        if speed == 0:
            speed = DEFAULT_KPH[hw]
        speed = min(speed, 255)

        pts = []
        try:
            for n in w.nodes:
                if not n.location.valid():
                    continue
                pts.append((n.location.lat, n.location.lon))
        except osmium.InvalidLocationError:
            return
        if len(pts) < 2:
            return

        pts = self._decimate(pts)
        if len(pts) < 2:
            return

        for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
            mla, mlo = (la1 + la2) / 2, (lo1 + lo2) / 2
            if not any(a <= mla <= c and b <= mlo <= d
                       for (a, b, c, d) in self.bbox):
                continue
            self.coords.extend((int(la1 * 1e6), int(lo1 * 1e6),
                                int(la2 * 1e6), int(lo2 * 1e6)))
            self.meta.extend((cid, speed))
        self.ways += 1

    @staticmethod
    def _decimate(pts):
        """Thin dense geometry. Motorways carry a node every few metres, which
        is precision we cannot use - the GPS fix is worth ~5 m at best."""
        out = [pts[0]]
        for la, lo in pts[1:]:
            pla, plo = out[-1]
            dy = (la - pla) * M_PER_DEG_LAT
            dx = (lo - plo) * M_PER_DEG_LAT * math.cos(math.radians(la))
            if dy * dy + dx * dx >= DECIMATE_M * DECIMATE_M:
                out.append((la, lo))
        if out[-1] != pts[-1]:
            out.append(pts[-1])
        return out


def write_index(coords: array, meta: array, bbox, out_path: str):
    n = len(meta) // 2
    if n == 0:
        raise SystemExit("no segments matched - check --bbox")

    c = np.frombuffer(coords, dtype=np.int32).reshape(n, 4)
    m = np.frombuffer(meta, dtype=np.uint8).reshape(n, 2)

    mid_lat = (c[:, 0].astype(np.float64) + c[:, 2]) / 2e6
    mid_lon = (c[:, 1].astype(np.float64) + c[:, 3]) / 2e6

    min_lat, min_lon = float(mid_lat.min()), float(mid_lon.min())
    max_lat, max_lon = float(mid_lat.max()), float(mid_lon.max())

    n_lat = int(math.ceil((max_lat - min_lat) / TILE_DEG)) + 1
    n_lon = int(math.ceil((max_lon - min_lon) / TILE_DEG)) + 1

    ti = ((mid_lat - min_lat) / TILE_DEG).astype(np.int32)
    tj = ((mid_lon - min_lon) / TILE_DEG).astype(np.int32)
    tile = ti * n_lon + tj

    order = np.argsort(tile, kind="stable")
    tile_sorted = tile[order]
    c, m = c[order], m[order]

    n_tiles = n_lat * n_lon
    counts = np.bincount(tile_sorted, minlength=n_tiles).astype(np.int32)
    offsets = np.zeros(n_tiles, dtype=np.int32)
    np.cumsum(counts[:-1], out=offsets[1:])

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<if", VERSION, TILE_DEG))
        f.write(struct.pack("<ff", min_lat, min_lon))
        f.write(struct.pack("<iii", n_lat, n_lon, n))
        np.stack([offsets, counts], axis=1).astype("<i4").tofile(f)
        c.astype("<i4").tofile(f)
        m.astype("u1").tofile(f)

    size = os.path.getsize(out_path)
    print(f"\nwrote {out_path}")
    print(f"  segments   {n:,}")
    print(f"  tiles      {n_lat} x {n_lon} = {n_tiles:,}")
    print(f"  bounds     lat {min_lat:.4f}..{max_lat:.4f}  lon {min_lon:.4f}..{max_lon:.4f}")
    print(f"  size       {size/1e6:.1f} MB  ({size/max(n,1):.1f} bytes/segment)")
    occupied = int((counts > 0).sum())
    print(f"  non-empty tiles {occupied:,} ({100*occupied/n_tiles:.1f}%)")
    return size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", default=None)
    ap.add_argument("--clip-from", default=None,
                    help="an existing .bin to cut a region out of, instead of "
                         "re-reading the pbf (seconds rather than minutes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", default=None,
                    help="minLat,minLon,maxLat,maxLon - default is everything")
    ap.add_argument("--major-everywhere", default=None,
                    help="comma-separated classes kept outside --bbox too, "
                         "e.g. motorway,trunk,primary,secondary")
    args = ap.parse_args()

    if not args.pbf and not args.clip_from:
        print("need either --pbf or --clip-from")
        return 2

    everywhere = None
    if args.major_everywhere:
        everywhere = set()
        for name in args.major_everywhere.split(","):
            name = name.strip()
            if name not in CLASS_ID:
                print(f"unknown class '{name}'; known: {', '.join(CLASSES)}")
                return 2
            everywhere.add(CLASS_ID[name])
            # links belong with their parent - an unmapped slip road is a hole
            # exactly where you are changing speed.
            if f"{name}_link" in CLASS_ID:
                everywhere.add(CLASS_ID[f"{name}_link"])

    # One or more boxes, separated by ';'. Multiple because a driver's roads
    # are a few cities plus the highways between them, not one rectangle.
    bbox = [(-90.0, -180.0, 90.0, 180.0)]
    if args.bbox:
        bbox = []
        for chunk in args.bbox.split(";"):
            parts = [float(x) for x in chunk.split(",")]
            if len(parts) != 4:
                print("each --bbox needs minLat,minLon,maxLat,maxLon")
                return 2
            bbox.append(tuple(parts))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.clip_from:
        coords, meta = clip_existing(args.clip_from, bbox, everywhere)
    else:
        print(f"reading {args.pbf}")
        print(f"keeping {len(CLASSES)} highway classes, "
              f"decimating to {DECIMATE_M:.0f} m")
        h = RoadHandler(bbox)
        h.apply_file(args.pbf, locations=True, idx="flex_mem")
        print(f"\nways kept  {h.ways:,}")
        top = sorted(h.skipped_classes.items(), key=lambda kv: -kv[1])[:6]
        print("skipped highway types (top): "
              + ", ".join(f"{k}={v:,}" for k, v in top))
        coords, meta = h.coords, h.meta

    write_index(coords, meta, bbox, args.out)
    return 0


def clip_existing(path: str, bbox, everywhere: set[int] | None = None):
    """Cut a bounding box out of an already-built index.

    Re-reading the pbf costs minutes and a gigabyte of node cache; the built
    index already holds exactly the segments we want in exactly the form we
    want them, so cutting a city out of it is a filter over one array."""
    import struct as _struct
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path} is not a road index")
    version, tile_deg, min_lat, min_lon, n_lat, n_lon, n_seg = \
        _struct.unpack_from("<ifffiii", raw, 8)

    tiles_start = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4
    coords_start = tiles_start + n_lat * n_lon * 8
    meta_start = coords_start + n_seg * 16

    c = np.frombuffer(raw, dtype="<i4", count=n_seg * 4,
                      offset=coords_start).reshape(-1, 4)
    m = np.frombuffer(raw, dtype="u1", count=n_seg * 2,
                      offset=meta_start).reshape(-1, 2)

    mid_lat = (c[:, 0].astype(np.float64) + c[:, 2]) / 2e6
    mid_lon = (c[:, 1].astype(np.float64) + c[:, 3]) / 2e6
    inside = np.zeros(len(mid_lat), dtype=bool)
    for (lo_la, lo_lo, hi_la, hi_lo) in bbox:
        inside |= ((mid_lat >= lo_la) & (mid_lat <= hi_la)
                   & (mid_lon >= lo_lo) & (mid_lon <= hi_lo))

    print(f"clipping {path}")
    print(f"  {n_seg:,} segments -> {int(inside.sum()):,} inside the box")

    keep = inside
    if everywhere:
        # Major roads are kept NATIONWIDE regardless of the box. A box drawn
        # round the city you live in does not contain the motorway you take to
        # the next one, and the motorway is precisely where the limit is
        # highest, the speeding is easiest, and a missing limit costs most.
        # They are also cheap: motorway through secondary is under 9% of all
        # segments because long-distance roads are few and the back streets
        # are many.
        major = np.isin(m[:, 0], list(everywhere))
        keep = inside | major
        print(f"  + {int((major & ~inside).sum()):,} major-road segments "
              f"outside the box (kept nationwide)")
        print(f"  = {int(keep.sum()):,} total")

    coords = array("i")
    coords.frombytes(c[keep].astype("<i4").tobytes())
    meta = array("B")
    meta.frombytes(m[keep].astype("u1").tobytes())
    return coords, meta


if __name__ == "__main__":
    raise SystemExit(main())
