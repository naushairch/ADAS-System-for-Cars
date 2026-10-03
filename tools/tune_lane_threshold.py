"""
tune_lane_threshold.py — pick LaneDetector.MIN_CELL_CONFIDENCE with evidence
instead of a guess.

WHY THIS IS A SEPARATE STEP FROM TRAINING
    Training reports one operating point: plain argmax, no confidence floor.
    The app does not run that way. LaneDetector rejects a winning cell below
    MIN_CELL_CONFIDENCE, and LaneDeparture then requires MIN_ANCHORS_FOR_LINE
    located anchors before it will call a slot a line at all. Those two gates
    move the false-alarm rate a long way, and the right setting is a choice
    about consequences, not a number that falls out of the loss.

    Raising the threshold buys silence and pays in missed lines. On a road
    with clear paint that is a bad trade. On a road where the paint is gone -
    which is the case this feature exists to survive - it is the right one.

TWO LEVELS ARE REPORTED, AND THE SECOND IS THE ONE THAT MATTERS
    anchor level  per (slot, row) cell. What training prints. Fine-grained,
                  but a handful of stray anchors is not a warning.
    LINE level    per slot: did the model claim a lane boundary exists at all
                  (>= MIN_ANCHORS located)? This is what LaneDeparture gates
                  on, so a false line here is what actually becomes an
                  audible alert on empty road. Tune against this column.

USAGE
    python tune_lane_threshold.py --ufld d:/adas/data/ufld \
        --images d:/adas/data/bdd100k/bdd100k/images/100k --limit 2000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ufld_lane.dataset import BDDLaneDataset
from ufld_lane.model import UFLDNet, GRIDING_NUM, ABSENT, NUM_LANES

# Must match LaneDeparture.java / LaneDetector.java.
MIN_ANCHORS_FOR_LINE = 8

THRESHOLDS = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufld", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(__file__), "ufld_lane", "weights", "ufld_lane.pt"))
    ap.add_argument("--limit", type=int, default=2000,
                    help="val images to use; 0 for all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    ds = BDDLaneDataset(os.path.join(args.ufld, "val.npz"),
                        os.path.join(args.images, "val"), train=False)
    if args.limit and args.limit < len(ds):
        ds = Subset(ds, range(args.limit))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    # Same default-to-2048 rule as export_lane_tflite.py: checkpoints predating
    # --hidden have no such key. Without this the h512 checkpoint dies here with
    # a load_state_dict shape error rather than being tuned.
    net = UFLDNet(num_anchors=int(ck["num_anchors"]),
                  backbone=ck.get("backbone", "resnet18"), pretrained=False,
                  hidden=int(ck.get("hidden", 2048)))
    net.load_state_dict(ck["model"])
    net.to(device).eval()

    m = ck.get("metrics", {})
    print(f"checkpoint : {args.ckpt}")
    print(f"hidden     : {int(ck.get('hidden', 2048))}")
    if m:
        print(f"trained to : pos {100*m['pos_acc']:.1f}%  "
              f"false-lane {100*m['false_lane_rate']:.2f}%  "
              f"miss {100*m['miss_rate']:.1f}%")
    print(f"images     : {len(ds)}\n")

    nT = len(THRESHOLDS)
    a_false = np.zeros(nT, dtype=np.int64)     # anchor: claimed, truly absent
    a_absent = 0
    a_miss = np.zeros(nT, dtype=np.int64)      # anchor: abstained, truly present
    a_present = 0
    a_close = np.zeros(nT, dtype=np.int64)     # anchor: within 1 cell, kept

    l_false = np.zeros(nT, dtype=np.int64)     # LINE: claimed, slot truly absent
    l_absent = 0
    l_miss = np.zeros(nT, dtype=np.int64)      # LINE: not claimed, slot present
    l_present = 0

    with torch.no_grad():
        for img, grid, style in dl:
            img = img.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                cls, _ = net(img)
            cls = cls.float()

            prob = cls.softmax(dim=1)                      # (B,G+1,A,L)
            cells = prob[:, :GRIDING_NUM, :, :]
            p_absent = prob[:, GRIDING_NUM, :, :]
            mass = cells.sum(dim=1)
            peak, arg = cells.max(dim=1)                   # (B,A,L)

            tgt = grid.to(device).permute(0, 2, 1)         # (B,A,L)
            present = tgt != ABSENT
            absent = ~present
            slot_present = (style.to(device) != 0)         # (B,L)

            a_absent += int(absent.sum())
            a_present += int(present.sum())
            l_absent += int((~slot_present).sum())
            l_present += int(slot_present.sum())

            beats_absent = mass > p_absent
            for ti, t in enumerate(THRESHOLDS):
                kept = beats_absent & (peak >= t)          # (B,A,L)
                a_false[ti] += int((kept & absent).sum())
                a_miss[ti] += int((~kept & present).sum())
                a_close[ti] += int((kept & present & ((arg - tgt).abs() <= 1)).sum())

                n_anchors = kept.sum(dim=1)                # (B,L)
                claims = n_anchors >= MIN_ANCHORS_FOR_LINE
                l_false[ti] += int((claims & ~slot_present).sum())
                l_miss[ti] += int((~claims & slot_present).sum())

    print("                 anchor level              LINE level (what the app gates on)")
    print("  thresh   false-lane   miss   pos-acc     false-LINE   miss-LINE")
    print("  " + "-" * 68)
    for ti, t in enumerate(THRESHOLDS):
        print(f"  {t:5.2f}   {100*a_false[ti]/max(a_absent,1):8.2f}%  "
              f"{100*a_miss[ti]/max(a_present,1):5.1f}%  "
              f"{100*a_close[ti]/max(a_present,1):6.1f}%     "
              f"{100*l_false[ti]/max(l_absent,1):8.2f}%   "
              f"{100*l_miss[ti]/max(l_present,1):7.1f}%")

    print("\nHOW TO READ THIS")
    print("  false-LINE is the rate at which the app would claim a lane boundary")
    print("  that is not there. That is the number that becomes a wrong warning.")
    print("  miss-LINE is what you pay for it. Pick the largest threshold whose")
    print("  miss-LINE you can still live with, NOT the best-looking f1.")
    print("\n  Then re-check on your OWN footage. BDD is well-marked American road;")
    print("  it cannot tell you how this behaves on faded Pakistani markings, and")
    print("  that is the case the whole silence design exists for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
