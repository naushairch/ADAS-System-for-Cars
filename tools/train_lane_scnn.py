"""
train_lane_scnn.py — trains scnn_lane/model.py on data from
carla_collect_lanes.py and writes scnn_lane/weights/scnn_lane.pt, which
lane_detect.py loads automatically.

Two losses, added together, BOTH weighted per-sample by how close the
ground-truth line truly is to the car (see scnn_lane/dataset.py):
    segmentation  - per-pixel {background, left, right}, class-weighted
                    because background dominates every frame
    existence     - per-lane (left, right) {absent, broken, solid}

WHY PER-SAMPLE OFFSET WEIGHTING, NOT ROW WEIGHTING
    An earlier version of this file weighted the loss by ROW (rows nearer
    the car counted for more). That measurably improved position accuracy
    on the COMMON case - the car centred, ~2m from each line - because
    nearly every frame has a line somewhere in its bottom rows, so that's
    what the extra row-weight mostly trained on. It did almost nothing for
    the RARE case that the touch warning actually depends on: the car
    genuinely close to a line (~9% of frames - lane changes are brief).
    Measured directly against ground truth, the model still only agreed
    with the true "car is on the line" verdict on ~20% of true near-line
    frames after that fix, with the same ~1.3m directional bias as before -
    row weighting was reweighting the wrong axis.

    Weighting each SAMPLE (not row) by its own true offset-from-line
    directly counteracts the actual imbalance: a genuinely close-to-the-
    line frame gets up to NEAR_LINE_MAX_WEIGHT times the say in the loss
    that a typical centred frame gets (see scnn_lane/dataset.py), instead
    of being drowned out by the 91% of frames that don't need that precision.

USAGE
    python train_lane_scnn.py --data lane_data --epochs 30
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scnn_lane.dataset import LaneBEVDataset, CAR_X_PX, METRES_PER_BEV_PX, NEAR_BAND_PX
from scnn_lane.model import SCNNLaneNet

DEFAULT_OUT = os.path.join(os.path.dirname(__file__),
                           "scnn_lane", "weights", "scnn_lane.pt")

# What "near-line" means for the RECALL diagnostic below - matches
# LaneDetector's default touch_threshold_m ballpark (half_track + line
# width + tolerance, DEFAULT_HALF_TRACK_M=0.78 in lane_detect.py).
NEAR_METRIC_CUTOFF_M = 0.967


def _true_near_side(mask_band_np):
    """mask_band_np: (NEAR_BAND_PX, W) ground-truth slice for one sample.
    Returns (side_class, offset_m) for whichever line is nearer the car,
    or (None, None) if neither line appears in this band."""
    best = None
    for cls in (1, 2):
        cols = np.where(mask_band_np == cls)[1]
        if cols.size:
            off = abs(cols.mean() - CAR_X_PX) * METRES_PER_BEV_PX
            if best is None or off < best[1]:
                best = (cls, off)
    return best if best else (None, None)


def _pred_offset_for_class(pred_band_np, cls):
    cols = np.where(pred_band_np == cls)[1]
    if cols.size:
        return abs(cols.mean() - CAR_X_PX) * METRES_PER_BEV_PX
    return None


def evaluate(model, loader, device, seg_weights, exist_criterion):
    model.eval()
    n_batches = 0
    seg_loss_sum = exist_loss_sum = 0.0
    seg_correct = seg_total = 0
    exist_correct = exist_total = 0
    near_total = near_hits = 0
    near_pos_errs = []

    with torch.no_grad():
        for img, mask, exist, weight in loader:
            img, mask, exist = img.to(device), mask.to(device), exist.to(device)
            seg_logits, exist_logits = model(img)

            seg_loss_sum += F.cross_entropy(seg_logits, mask, weight=seg_weights).item()
            exist_loss_sum += exist_criterion(
                exist_logits.reshape(-1, exist_logits.shape[-1]),
                exist.reshape(-1)).item()
            n_batches += 1

            seg_pred = seg_logits.argmax(1)
            seg_correct += (seg_pred == mask).sum().item()
            seg_total += mask.numel()

            exist_pred = exist_logits.argmax(-1)
            exist_correct += (exist_pred == exist).sum().item()
            exist_total += exist.numel()

            # The metric that actually matters: of the frames where the car
            # is TRULY within touch range of a line, how often does the
            # model's own predicted position also fall within touch range?
            h = mask.shape[1]
            mask_band = mask[:, h - NEAR_BAND_PX:, :].cpu().numpy()
            pred_band = seg_pred[:, h - NEAR_BAND_PX:, :].cpu().numpy()
            for b in range(mask.shape[0]):
                cls, true_off = _true_near_side(mask_band[b])
                if cls is None or true_off > NEAR_METRIC_CUTOFF_M:
                    continue
                near_total += 1
                pred_off = _pred_offset_for_class(pred_band[b], cls)
                if pred_off is not None:
                    near_pos_errs.append(pred_off - true_off)
                    if pred_off <= NEAR_METRIC_CUTOFF_M:
                        near_hits += 1

    return dict(
        seg_loss=seg_loss_sum / max(1, n_batches),
        exist_loss=exist_loss_sum / max(1, n_batches),
        seg_acc=seg_correct / max(1, seg_total),
        exist_acc=exist_correct / max(1, exist_total),
        near_line_recall=near_hits / near_total if near_total else float("nan"),
        near_line_bias_m=sum(near_pos_errs) / len(near_pos_errs) if near_pos_errs else float("nan"),
        near_total=near_total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="output dir of carla_collect_lanes.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seg-weight", type=float, default=1.0)
    ap.add_argument("--exist-weight", type=float, default=1.0)
    ap.add_argument("--bg-class-weight", type=float, default=1.0)
    ap.add_argument("--line-class-weight", type=float, default=6.0,
                    help="up-weights left/right pixels; background dominates otherwise")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds = LaneBEVDataset(args.data, split="train")
    val_ds = LaneBEVDataset(args.data, split="val")
    print(f"train frames: {len(train_ds)}   val frames: {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit(f"no training frames found under {args.data} - "
                         "run carla_collect_lanes.py first")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    model = SCNNLaneNet().to(device)

    seg_weights = torch.tensor(
        [args.bg_class_weight, args.line_class_weight, args.line_class_weight],
        device=device)
    exist_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # near_line_recall - not existence accuracy, not seg accuracy - is the
    # thing that was actually broken (see module docstring), so it drives
    # checkpoint selection. EXIST_ACC_FLOOR guards against a checkpoint
    # "winning" on recall only because existence classification collapsed
    # alongside it.
    EXIST_ACC_FLOOR = 0.90
    best_recall = -1.0
    best_exist_acc = -1.0
    last_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for img, mask, exist, weight in train_loader:
            img, mask, exist, weight = (img.to(device), mask.to(device),
                                        exist.to(device), weight.to(device))

            seg_logits, exist_logits = model(img)

            pixel_loss = F.cross_entropy(seg_logits, mask, weight=seg_weights,
                                         reduction="none")           # [B,H,W]
            per_sample_seg = pixel_loss.mean(dim=(1, 2))              # [B]
            seg_loss = (per_sample_seg * weight).mean()

            exist_loss_per = F.cross_entropy(
                exist_logits.reshape(-1, exist_logits.shape[-1]),
                exist.reshape(-1), reduction="none")                  # [B*2]
            exist_w = weight.repeat_interleave(2)                     # [B*2]
            exist_loss = (exist_loss_per * exist_w).mean()

            loss = args.seg_weight * seg_loss + args.exist_weight * exist_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        train_loss = running / max(1, len(train_loader))

        msg = f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f}"
        last_state = model.state_dict()
        if len(val_ds):
            m = evaluate(model, val_loader, device, seg_weights, exist_criterion)
            msg += (f"  val_seg_acc {m['seg_acc']:.3f}  val_exist_acc {m['exist_acc']:.3f}"
                    f"  val_near_recall {m['near_line_recall']:.3f}"
                    f" (n={m['near_total']})"
                    f"  val_near_bias_m {m['near_line_bias_m']:+.3f}"
                    f"  val_loss {m['seg_loss'] + m['exist_loss']:.4f}")
            if (m["exist_acc"] >= EXIST_ACC_FLOOR
                    and m["near_line_recall"] > best_recall):
                best_recall = m["near_line_recall"]
                best_exist_acc = m["exist_acc"]
                torch.save(model.state_dict(), args.out)
                msg += "  [saved]"
        else:
            torch.save(model.state_dict(), args.out)
            msg += "  [saved]"

        print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)

    if best_recall < 0 and last_state is not None:
        print("\nWARNING: no epoch reached the existence-accuracy floor "
             f"({EXIST_ACC_FLOOR}) with a valid near-line measurement; "
             "saving the last epoch's weights instead.")
        torch.save(last_state, args.out)
    else:
        print(f"\nbest: val_exist_acc {best_exist_acc:.3f}  "
             f"val_near_recall {best_recall:.3f}")
    print(f"weights -> {args.out}")


if __name__ == "__main__":
    main()
