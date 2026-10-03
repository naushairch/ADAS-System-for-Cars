"""
train_lane_ufld.py — trains ufld_lane/model.py on the targets bdd_to_ufld.py
produced, and writes ufld_lane/weights/ufld_lane.pt.

TWO LOSSES PLUS A REGULARISER
    position   cross-entropy over GRIDING_NUM+1 choices per (slot, anchor).
               The +1 is ABSENT, and it is a real class, not padding.
    style      cross-entropy over {absent, dashed, solid} per slot.
    similarity lane lines are continuous, so the distribution at one row
               anchor should resemble the one at the anchor above it. UFLD
               adds this because nothing else in the loss knows that the
               anchors are vertically ordered - without it the model is free
               to produce a zig-zag that is locally plausible and globally
               nonsense.

THE METRIC THAT ACTUALLY MATTERS HERE
    Not position accuracy. FALSE LANE RATE: of all (slot, anchor) targets
    that are genuinely ABSENT, how often did the model claim a line anyway?

    That is the number the silence requirement lives or dies on, and it is
    invisible in an accuracy figure. Roughly 80% of targets are ABSENT, so a
    model that answers "no lane" everywhere scores ~80% accuracy while being
    completely useless, and a model that guesses aggressively can post good
    position accuracy while inventing lanes on every unmarked road. Both are
    reported separately below, every epoch, for that reason.

ON --absent-weight
    ABSENT dominates the targets, so the usual move is to down-weight it and
    let the model concentrate on the rare positive cells. Resist doing that
    hard here. Down-weighting ABSENT directly trades away the ability to
    stay quiet, which is the entire feature being built. The default of 0.4
    is a deliberate compromise; push it toward 1.0 for a more silent, more
    conservative model, and toward 0.1 only if you are willing to accept
    lane warnings on blank tarmac.

USAGE
    python train_lane_ufld.py --ufld d:/adas/data/ufld \
        --images d:/adas/data/bdd100k/bdd100k/images/100k --epochs 15
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ufld_lane.dataset import BDDLaneDataset
from ufld_lane.model import UFLDNet, GRIDING_NUM, ABSENT, NUM_LANES

DEFAULT_OUT = os.path.join(os.path.dirname(__file__),
                           "ufld_lane", "weights", "ufld_lane.pt")


def similarity_loss(cls_logits: torch.Tensor) -> torch.Tensor:
    """Encourage adjacent row anchors to agree. cls: (B, G+1, A, L).
    Compared over the real cells only - ABSENT is a categorical state, not a
    position, so smoothing it against a neighbour is meaningless."""
    p = cls_logits[:, :GRIDING_NUM, :, :].softmax(dim=1)
    return F.l1_loss(p[:, :, 1:, :], p[:, :, :-1, :])


def evaluate(model, loader, device, absent_w):
    model.eval()
    tot = dict(loss=0.0, n=0,
               pos_correct=0, pos_total=0,
               false_lane=0, true_absent=0,
               missed=0, true_present=0,
               style_correct=0, style_total=0)

    w = torch.ones(GRIDING_NUM + 1, device=device)
    w[ABSENT] = absent_w

    with torch.no_grad():
        for img, grid, style in loader:
            img, grid, style = img.to(device), grid.to(device), style.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                cls, st = model(img)
                # cls (B,G+1,A,L); target wants (B,A,L)
                tgt = grid.permute(0, 2, 1)
                loss = F.cross_entropy(cls, tgt, weight=w)
            tot["loss"] += float(loss) * img.size(0)
            tot["n"] += img.size(0)

            pred = cls.argmax(dim=1)                  # (B,A,L)
            present = tgt != ABSENT
            absent = ~present

            # Position accuracy: within one cell, on genuinely present lanes.
            close = (pred - tgt).abs() <= 1
            tot["pos_correct"] += int((close & present).sum())
            tot["pos_total"] += int(present.sum())

            # The two failure modes, counted separately.
            tot["false_lane"] += int(((pred != ABSENT) & absent).sum())
            tot["true_absent"] += int(absent.sum())
            tot["missed"] += int(((pred == ABSENT) & present).sum())
            tot["true_present"] += int(present.sum())

            tot["style_correct"] += int((st.argmax(-1) == style).sum())
            tot["style_total"] += int(style.numel())

    d = lambda a, b: (a / b) if b else float("nan")
    return {
        "loss": d(tot["loss"], tot["n"]),
        "pos_acc": d(tot["pos_correct"], tot["pos_total"]),
        "false_lane_rate": d(tot["false_lane"], tot["true_absent"]),
        "miss_rate": d(tot["missed"], tot["true_present"]),
        "style_acc": d(tot["style_correct"], tot["style_total"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufld", required=True, help="dir holding train.npz / val.npz")
    ap.add_argument("--images", required=True, help=".../images/100k")
    ap.add_argument("--epochs", type=int, default=15)
    # Defaults sized for THIS machine: 15.6 GB RAM (~2.4 GB actually free)
    # and an 8 GB 4070 that already has ~2.7 GB taken by the desktop.
    #
    # --workers is the dangerous one on Windows. There is no fork(), so every
    # worker is a fresh process that re-imports torch and opencv - on the
    # order of a GB resident each. Asking for 8 crashed on the first forward
    # pass with a bare "RuntimeError: bad allocation", which reads like a GPU
    # problem and is not one: it is the host running out of RAM. 2 workers is
    # about the ceiling here; raise it only after watching Task Manager.
    #
    # --batch 16 keeps activations near 3 GB, inside the ~5.3 GB of VRAM left
    # after the desktop. 32 fits only on an idle GPU.
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--absent-weight", type=float, default=0.4)
    ap.add_argument("--sim-weight", type=float, default=0.2)
    ap.add_argument("--style-weight", type=float, default=1.0)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    # Width of the cls_head bottleneck. 2048 is the original and reproduces the
    # baseline checkpoint; 512 cuts the model from 55.9M to 22.7M parameters.
    # See the note in ufld_lane/model.py.
    #
    # NOTE --epochs is part of the architecture comparison, not a separate dial:
    # OneCycleLR anneals over exactly args.epochs, so a run stopped early is NOT
    # the same model as a shorter run trained to completion. Comparing a 5-epoch
    # h512 against the 15-epoch baseline measures both changes at once.
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA. This will take days rather than an hour.")

    tr = BDDLaneDataset(os.path.join(args.ufld, "train.npz"),
                        os.path.join(args.images, "train"), train=True)
    va = BDDLaneDataset(os.path.join(args.ufld, "val.npz"),
                        os.path.join(args.images, "val"), train=False)

    print(f"train {len(tr)} images | val {len(va)} images")
    print(f"anchors {tr.num_anchors} | ABSENT share of targets "
          f"{100*tr.class_balance():.1f}%")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0)

    model = UFLDNet(num_anchors=tr.num_anchors, backbone=args.backbone,
                    hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.1f}M  (hidden={args.hidden})  "
          f"~{2*n_par/1e6:.0f} MB at float16", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    w = torch.ones(GRIDING_NUM + 1, device=device)
    w[ABSENT] = args.absent_weight

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for it, (img, grid, style) in enumerate(tl, 1):
            img = img.to(device, non_blocking=True)
            grid = grid.to(device, non_blocking=True)
            style = style.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                cls, st = model(img)
                tgt = grid.permute(0, 2, 1)               # (B,A,L)
                loss_pos = F.cross_entropy(cls, tgt, weight=w)
                loss_sty = F.cross_entropy(st.reshape(-1, 3), style.reshape(-1))
                loss_sim = similarity_loss(cls)
                loss = (loss_pos
                        + args.style_weight * loss_sty
                        + args.sim_weight * loss_sim)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            run += float(loss)
            if it % 200 == 0:
                print(f"  ep{ep} {it}/{len(tl)} loss {run/it:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        m = evaluate(model, vl, device, args.absent_weight)
        print(f"\nepoch {ep}/{args.epochs}  {time.time()-t0:.0f}s")
        print(f"  val loss          {m['loss']:.4f}")
        print(f"  position acc      {100*m['pos_acc']:.1f}%   (within 1 cell, where a lane exists)")
        print(f"  FALSE LANE RATE   {100*m['false_lane_rate']:.2f}%  <- invented a lane on empty road")
        print(f"  miss rate         {100*m['miss_rate']:.1f}%   (stayed silent where a lane existed)")
        print(f"  style acc         {100*m['style_acc']:.1f}%\n", flush=True)

        if m["loss"] < best:
            best = m["loss"]
            torch.save({
                "model": model.state_dict(),
                "num_anchors": tr.num_anchors,
                "row_anchors": tr.row_anchors,
                "griding_num": GRIDING_NUM,
                "num_lanes": NUM_LANES,
                "backbone": args.backbone,
                # Recorded so export_lane_tflite.py can rebuild the exact
                # architecture. Absent in checkpoints written before this knob
                # existed, which is why the reader defaults it to 2048.
                "hidden": args.hidden,
                "epochs": args.epochs,
                "metrics": m,
            }, args.out)
            print(f"  saved -> {args.out}\n", flush=True)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
