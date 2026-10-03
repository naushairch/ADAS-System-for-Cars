"""
check_road_index.py — read a road index EXACTLY the way RoadIndex.java reads
it, and prove the tile search returns the true nearest road.

WHY THIS EXISTS
    The binary layout is written here in Python and parsed again in Java, with
    the byte offsets derived independently in both places. That is two
    statements of the same fact, and when they disagree nothing crashes: the
    reader lands mid-record and returns plausible-looking coordinates for the
    wrong road. The app would then quietly set 120 on a residential street.

    So this re-implements the Java reader's arithmetic - same header, same
    offsets, same 3x3 tile walk - and checks two things:

      1. every field round-trips (bounds, counts, class ids in range),
      2. the 3x3 tile search finds the same segment a BRUTE-FORCE scan over
         every segment in the file finds.

    Point 2 is the one that matters. Tiling is an optimisation, and an
    optimisation that silently returns the second-nearest road is worse than
    no optimisation at all.

USAGE
    python check_road_index.py --index ../app/src/main/assets/roads_pk.bin
    python check_road_index.py --index roads_pk.bin --at 31.5204,74.3587
"""

from __future__ import annotations

import argparse
import math
import struct

import numpy as np

MAGIC = b"ADASROAD"
HEADER_BYTES = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4
MAX_SNAP_M = 25.0                 # must match RoadIndex.MAX_SNAP_M
M_PER_DEG_LAT = 111320.0

CLASS_NAMES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street",
]


class Index:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.raw = f.read()
        if self.raw[:8] != MAGIC:
            raise SystemExit("bad magic - not a road index")
        (self.version, self.tile_deg, self.min_lat, self.min_lon,
         self.n_lat, self.n_lon, self.n_seg) = struct.unpack_from(
            "<ifffiii", self.raw, 8)

        self.tiles_start = HEADER_BYTES
        self.coords_start = self.tiles_start + self.n_lat * self.n_lon * 8
        self.meta_start = self.coords_start + self.n_seg * 16
        need = self.meta_start + self.n_seg * 2
        if len(self.raw) != need:
            raise SystemExit(
                f"size mismatch: file {len(self.raw)}, layout implies {need}")

        self.tiles = np.frombuffer(self.raw, dtype="<i4",
                                   count=self.n_lat * self.n_lon * 2,
                                   offset=self.tiles_start).reshape(-1, 2)
        self.coords = np.frombuffer(self.raw, dtype="<i4",
                                    count=self.n_seg * 4,
                                    offset=self.coords_start).reshape(-1, 4)
        self.meta = np.frombuffer(self.raw, dtype="u1", count=self.n_seg * 2,
                                  offset=self.meta_start).reshape(-1, 2)

    # --- the same maths RoadIndex.java does -------------------------------
    @staticmethod
    def _seg_d2(plat, plon, la1, lo1, la2, lo2, cos_lat):
        px = (plon - lo1) * M_PER_DEG_LAT * cos_lat
        py = (plat - la1) * M_PER_DEG_LAT
        vx = (lo2 - lo1) * M_PER_DEG_LAT * cos_lat
        vy = (la2 - la1) * M_PER_DEG_LAT
        len2 = vx * vx + vy * vy
        t = np.zeros_like(len2) if np.ndim(len2) else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(len2 <= 1e-9, 0.0, (px * vx + py * vy) / np.maximum(len2, 1e-9))
        t = np.clip(t, 0.0, 1.0)
        dx = px - t * vx
        dy = py - t * vy
        return dx * dx + dy * dy

    def lookup_tiled(self, lat, lon):
        ti = int(math.floor((lat - self.min_lat) / self.tile_deg))
        tj = int(math.floor((lon - self.min_lon) / self.tile_deg))
        cos_lat = math.cos(math.radians(lat))
        best, best_idx = MAX_SNAP_M ** 2, -1
        for di in (-1, 0, 1):
            i = ti + di
            if not (0 <= i < self.n_lat):
                continue
            for dj in (-1, 0, 1):
                j = tj + dj
                if not (0 <= j < self.n_lon):
                    continue
                off, cnt = self.tiles[i * self.n_lon + j]
                if cnt == 0:
                    continue
                c = self.coords[off:off + cnt].astype(np.float64) / 1e6
                d2 = self._seg_d2(lat, lon, c[:, 0], c[:, 1], c[:, 2], c[:, 3], cos_lat)
                k = int(np.argmin(d2))
                if d2[k] < best:
                    best, best_idx = float(d2[k]), int(off) + k
        return (best_idx, math.sqrt(best)) if best_idx >= 0 else (None, None)

    def lookup_bruteforce(self, lat, lon):
        cos_lat = math.cos(math.radians(lat))
        c = self.coords.astype(np.float64) / 1e6
        d2 = self._seg_d2(lat, lon, c[:, 0], c[:, 1], c[:, 2], c[:, 3], cos_lat)
        k = int(np.argmin(d2))
        return (k, math.sqrt(float(d2[k]))) if d2[k] <= MAX_SNAP_M ** 2 else (None, None)

    def describe(self, idx):
        cid, kph = self.meta[idx]
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"?{cid}"
        return f"{name} {kph} km/h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--at", default=None, help="lat,lon to query")
    ap.add_argument("--trials", type=int, default=300)
    args = ap.parse_args()

    ix = Index(args.index)
    mid_lat = (ix.coords[:, 0].astype(np.float64) + ix.coords[:, 2]) / 2e6
    mid_lon = (ix.coords[:, 1].astype(np.float64) + ix.coords[:, 3]) / 2e6

    print(f"index      {args.index}")
    print(f"  version  {ix.version}   tile {ix.tile_deg} deg")
    print(f"  segments {ix.n_seg:,}   tiles {ix.n_lat} x {ix.n_lon}")
    print(f"  bounds   lat {mid_lat.min():.4f}..{mid_lat.max():.4f}  "
          f"lon {mid_lon.min():.4f}..{mid_lon.max():.4f}")

    bad_cls = int((ix.meta[:, 0] >= len(CLASS_NAMES)).sum())
    bad_kph = int(((ix.meta[:, 1] < 5) | (ix.meta[:, 1] > 150)).sum())
    print(f"  class ids out of range {bad_cls}   speeds out of range {bad_kph}")
    if bad_cls or bad_kph:
        print("  FAIL: the reader and writer disagree about record layout")
        return 1

    counts = np.bincount(ix.meta[:, 0], minlength=len(CLASS_NAMES))
    print("\nsegments by road class")
    for i, n in enumerate(counts):
        if n:
            print(f"  {CLASS_NAMES[i]:16s} {n:9,}  ({100*n/ix.n_seg:4.1f}%)")

    if args.at:
        la, lo = (float(x) for x in args.at.split(","))
        idx, d = ix.lookup_tiled(la, lo)
        print(f"\nquery {la},{lo}")
        print("  no road within 25 m" if idx is None
              else f"  {ix.describe(idx)}  at {d:.1f} m")

    # The real test: tiled search must agree with an exhaustive scan.
    rng = np.random.default_rng(0)
    pick = rng.integers(0, ix.n_seg, size=args.trials)
    agree = checked = tiled_found = 0
    for s in pick:
        # jitter a point off a real segment's midpoint, so some queries land
        # near a road and some land nowhere near one
        la = float(mid_lat[s]) + float(rng.normal(0, 0.0004))
        lo = float(mid_lon[s]) + float(rng.normal(0, 0.0004))
        t_idx, t_d = ix.lookup_tiled(la, lo)
        b_idx, b_d = ix.lookup_bruteforce(la, lo)
        checked += 1
        if t_idx is not None:
            tiled_found += 1
        if (t_idx is None) == (b_idx is None):
            if t_idx is None or abs(t_d - b_d) < 0.01:
                agree += 1
    print(f"\ntiled vs brute force over {checked} random queries")
    print(f"  matched a road   {tiled_found}/{checked}")
    print(f"  identical result {agree}/{checked}  "
          f"{'PASS' if agree == checked else 'FAIL'}")
    return 0 if agree == checked else 1


if __name__ == "__main__":
    raise SystemExit(main())
