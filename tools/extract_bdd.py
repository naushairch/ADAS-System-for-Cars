"""
extract_bdd.py — pull ONLY the parts of the Kaggle BDD100K archive we need,
onto the drive that has room for them.

WHY THIS EXISTS RATHER THAN "just unzip it"
    Two hard constraints on this machine, both of which bite silently:

    1. C: has ~1.8 GB free and the archive itself is 8.2 GB. Unzipping in
       place cannot work, and Explorer's unzip fails *most of the way
       through*, after twenty minutes, leaving a half-written tree that
       looks plausible. D: has ~128 GB.

    2. The archive holds four things and we need one of them:
           bdd100k/bdd100k/images/100k/{train,val}   <- THIS (80k imgs)
           bdd100k/bdd100k/images/100k/test          <- no labels, skip
           bdd100k/bdd100k/images/10k                <- seg subset, skip
           bdd100k_seg/                              <- seg masks, skip
           bdd100k_labels_release/                   <- THIS (the labels)
       Skipping the rest saves roughly 2 GB and a lot of minutes.

USAGE
    python extract_bdd.py --zip "C:/Users/naush/Downloads/archive (1).zip" \
                          --out d:/adas/data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

# Prefixes worth extracting. Anything not matching one of these is skipped.
KEEP_PREFIXES = (
    "bdd100k/bdd100k/images/100k/train/",
    "bdd100k/bdd100k/images/100k/val/",
    "bdd100k_labels_release/bdd100k/labels/",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to the Kaggle archive")
    ap.add_argument("--out", required=True, help="destination root (use D:)")
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        print(f"archive not found: {args.zip}")
        return 2

    os.makedirs(args.out, exist_ok=True)

    with zipfile.ZipFile(args.zip) as z:
        members = [i for i in z.infolist()
                   if not i.is_dir() and i.filename.startswith(KEEP_PREFIXES)]
        total = sum(i.file_size for i in members)
        print(f"selected {len(members)} files, {total/1e9:.2f} GB uncompressed")

        free = _free_bytes(args.out)
        print(f"free on destination: {free/1e9:.1f} GB")
        if free < total * 1.05:
            print("NOT ENOUGH ROOM. Pick a drive with more space.")
            return 1

        done = 0
        t0 = time.time()
        for n, info in enumerate(members, 1):
            z.extract(info, args.out)
            done += info.file_size
            if n % 2000 == 0 or n == len(members):
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f"  {n:6d}/{len(members)}  {done/1e9:5.2f} GB"
                      f"  {rate/1e6:5.1f} MB/s  eta {eta/60:4.1f} min", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}")
    return 0


def _free_bytes(path: str) -> int:
    import shutil
    return shutil.disk_usage(path).free


if __name__ == "__main__":
    raise SystemExit(main())
