"""
eval_real_dataset.py — measure the detector against HUMAN-LABELLED real road images.

WHY
    Everything else in this project is validated against CARLA or synthetic
    geometry. Neither can answer the question that decides whether this works on a
    real road: does a COCO-trained YOLOv8n actually see the traffic that is out
    there, and does it label it well enough for the distance maths to be right?

    Ground truth answers that. The alternative is finding out from the driver's
    seat.

TWO THINGS ARE MEASURED, AND THEY MATTER FOR DIFFERENT REASONS

    1. RECALL - was the object found at all, under any label?
       This is what forward-collision warning depends on. An object you never
       detect can never be warned about, whatever you call it.

    2. LABEL CORRECTNESS - was it given a class whose assumed real-world width
       is right?
       DistanceEstimator computes  distance = realWidth * focal / boxWidthPx,
       and realWidth is looked up from the CLASS. So a mislabel is not cosmetic:
       call a 0.7 m electric bicycle a 1.75 m car and it is reported two and a
       half times further away than it is. The warning then arrives late, or
       never. A confident wrong distance is more dangerous than no distance.

DATASET FORMAT (maadaa.ai "Sunny Day City Road Dash Cam" and similar)
    <image>.jpg  alongside  <image>.json:
        { "imageWidth": .., "imageHeight": .., "objects": [
            { "label": "Car", "coordinate": [[x1,y1],[x2,y2]], ... }, ... ] }

    Note: in the sample set those two keys are SWAPPED with respect to the actual
    files, so image dimensions are read from the image itself, never the JSON.

USAGE
    python eval_real_dataset.py --data "C:\\path\\to\\extracted\\dataset"
    python eval_real_dataset.py --data ... --annotate --out real_eval
"""

import argparse
import collections
import glob
import json
import os

import cv2
import numpy as np

from adas_core import DistanceEstimator

# COCO ids the pipeline keeps.
CLS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
             5: "bus", 7: "truck"}
KEEP = set(CLS_NAMES)

# Dataset label -> COCO classes that would be an ACCEPTABLE label for it.
# "Acceptable" means the assumed width is close enough not to wreck the distance.
ACCEPTABLE = {
    "Car":             {2},
    "Van":             {2, 7},
    "Bus":             {5},
    "Truck":           {7, 5},
    "Person":          {0},
    "Bicycle":         {1, 3},
    "Electricbicycle": {3, 1},
    "Tricycle":        {3, 1},        # no real COCO equivalent
    "Motorcycle":      {3},
}

# True real-world width, metres — what the distance maths SHOULD be using.
TRUE_WIDTH_M = {
    "Car": 1.75, "Van": 2.00, "Bus": 2.50, "Truck": 2.45,
    "Person": 0.50, "Bicycle": 0.60, "Electricbicycle": 0.70,
    "Tricycle": 1.10, "Motorcycle": 0.75,
}

# Categories with no honest COCO equivalent — flagged separately in the report.
NO_COCO_EQUIVALENT = {"Electricbicycle", "Tricycle", "Van"}

IOU_MATCH = 0.45


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if uni <= 0 else inter / uni


def load_gt(json_path, scale):
    d = json.load(open(json_path, encoding="utf-8"))
    out = []
    for o in d.get("objects", []):
        c = o.get("coordinate")
        if not c or len(c) != 2:
            continue
        (x1, y1), (x2, y2) = c
        box = (min(x1, x2) * scale, min(y1, y2) * scale,
               max(x1, x2) * scale, max(y1, y2) * scale)
        out.append({"label": o.get("label", "?"), "box": box,
                    "covered": o.get("covered", "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="folder containing <name>.jpg + <name>.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="must match CONF in carla_live.py / Detector.java")
    ap.add_argument("--width", type=int, default=1280,
                    help="resize to this width, mimicking the app's analysis "
                         "resolution. Detection at native 2704px would flatter "
                         "the model relative to what the phone actually runs.")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="real_eval")
    args = ap.parse_args()

    images = [p for p in sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"),
                                          recursive=True))
              if "_effect" not in os.path.basename(p)
              and os.path.exists(p[:-4] + ".json")]
    if not images:
        raise SystemExit(f"no annotated images under {args.data}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    if args.annotate:
        os.makedirs(args.out, exist_ok=True)

    gt_total = collections.Counter()
    found_any = collections.Counter()
    found_right = collections.Counter()
    assigned = collections.defaultdict(collections.Counter)
    n_pred = 0
    n_fp = 0

    # Recall stratified by apparent size. A flat recall number is dominated by
    # tiny distant objects that no collision system could act on anyway —
    # DistanceEstimator.estimate() already returns NaN below 12px because the
    # distance would be meaningless. What matters is recall on things big enough
    # to be a hazard.
    SIZE_BINS = [(0, 12, "<12px (ignored by design)"),
                 (12, 30, "12-30px (far)"),
                 (30, 60, "30-60px (mid)"),
                 (60, 120, "60-120px (near)"),
                 (120, 10 ** 9, ">120px (close)")]
    bin_gt = collections.Counter()
    bin_seen = collections.Counter()

    # YOLO detecting the ego vehicle's own bonnet as a car.
    bonnet_fp = 0

    print(f"{len(images)} annotated images, analysed at {args.width}px wide, "
          f"conf={args.conf}\n")

    for p in images:
        img = cv2.imread(p)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        scale = args.width / float(w0)
        img = cv2.resize(img, (args.width, int(round(h0 * scale))),
                         interpolation=cv2.INTER_AREA)

        gts = load_gt(p[:-4] + ".json", scale)

        res = model.predict(img, conf=args.conf, verbose=False)[0]
        preds = []
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in KEEP:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append({"cls": cid, "box": (x1, y1, x2, y2),
                          "conf": float(b.conf[0]), "used": False})
        n_pred += len(preds)

        for g in gts:
            gt_total[g["label"]] += 1
            gw = g["box"][2] - g["box"][0]
            bin_name = next(n for lo, hi, n in SIZE_BINS if lo <= gw < hi)
            bin_gt[bin_name] += 1

            best, best_iou = None, IOU_MATCH
            for pr in preds:
                if pr["used"]:
                    continue
                v = iou(g["box"], pr["box"])
                if v > best_iou:
                    best_iou, best = v, pr
            if best is None:
                continue
            best["used"] = True
            bin_seen[bin_name] += 1
            found_any[g["label"]] += 1
            assigned[g["label"]][CLS_NAMES[best["cls"]]] += 1
            if best["cls"] in ACCEPTABLE.get(g["label"], set()):
                found_right[g["label"]] += 1

        ih, iw = img.shape[:2]
        for pr in preds:
            if pr["used"]:
                continue
            n_fp += 1
            # Bottom of frame, very wide, centred: that is our own bonnet.
            x1, y1, x2, y2 = pr["box"]
            if (y2 > ih * 0.88 and (x2 - x1) > iw * 0.35
                    and abs((x1 + x2) / 2 - iw / 2) < iw * 0.30):
                bonnet_fp += 1

        if args.annotate:
            vis = img.copy()
            for g in gts:
                x1, y1, x2, y2 = [int(v) for v in g["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, g["label"], (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            for pr in preds:
                x1, y1, x2, y2 = [int(v) for v in pr["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, CLS_NAMES[pr["cls"]], (x1, min(vis.shape[0] - 4,
                            y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255), 1)
            cv2.putText(vis, "green = truth   red = YOLO", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(args.out, os.path.basename(p)), vis)

    # ------------------------------------------------------------ report
    print("=" * 74)
    print("RECALL - was the object seen at all? (this is what FCW depends on)")
    print("=" * 74)
    print(f"{'category':18s} {'truth':>6s} {'seen':>6s} {'recall':>8s} "
          f"{'right class':>12s}")
    total_gt = total_seen = total_right = 0
    for label, n in gt_total.most_common():
        seen = found_any[label]
        right = found_right[label]
        total_gt += n
        total_seen += seen
        total_right += right
        flag = "  <-- no COCO class" if label in NO_COCO_EQUIVALENT else ""
        print(f"{label:18s} {n:6d} {seen:6d} {seen / n * 100:7.0f}% "
              f"{right / n * 100:11.0f}%{flag}")
    print("-" * 74)
    print(f"{'TOTAL':18s} {total_gt:6d} {total_seen:6d} "
          f"{total_seen / max(1, total_gt) * 100:7.0f}% "
          f"{total_right / max(1, total_gt) * 100:11.0f}%")

    print()
    print("=" * 74)
    print("WHAT IT CALLED THINGS, and what that does to the distance")
    print("=" * 74)
    for label in gt_total:
        if not assigned[label]:
            continue
        true_w = TRUE_WIDTH_M.get(label)
        parts = []
        for name, c in assigned[label].most_common():
            cid = [k for k, v in CLS_NAMES.items() if v == name][0]
            assumed = DistanceEstimator.real_width_for(cid)
            if true_w:
                factor = assumed / true_w
                parts.append(f"{name} x{c} (reads {factor:.2f}x distance)")
            else:
                parts.append(f"{name} x{c}")
        print(f"  {label:18s} -> " + ", ".join(parts))

    print()
    print("=" * 74)
    print("RECALL BY APPARENT SIZE - the number that actually matters")
    print("=" * 74)
    for _lo, _hi, name in SIZE_BINS:
        n = bin_gt[name]
        if not n:
            continue
        print(f"  {name:28s} {bin_seen[name]:4d}/{n:<4d} "
              f"{bin_seen[name] / n * 100:5.0f}%")

    print()
    print("=" * 74)
    print(f"unmatched detections (possible false positives): {n_fp} "
          f"of {n_pred} total ({n_fp / max(1, n_pred) * 100:.0f}%)")
    if bonnet_fp:
        print(f"  of which EGO BONNET detected as a vehicle: {bonnet_fp} "
              f"(in {len(images)} images)")
        print("  -> a huge box at the bottom of frame, dead centre, that never "
              "moves.")
        print("     Its width makes it read as ~1m away and it sits in the ego")
        print("     corridor permanently. depth_obstacle.py already masks this")
        print("     band (BONNET_FRACTION); the YOLO path does not.")
    print("  NOTE: some of these are real objects the dataset chose not to label")
    print("  (parked cars far away, cross traffic). Treat as an upper bound.")
    if args.annotate:
        print(f"\nannotated comparisons -> {args.out}")


if __name__ == "__main__":
    main()
