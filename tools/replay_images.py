"""
replay_images.py — run the full Python pipeline over a folder of real images.

WHY THIS EXISTS
    Everything validated so far has been either synthetic geometry (the parity
    scenarios) or CARLA renders. Neither tells you how YOLOv8n-on-COCO behaves on
    an actual road: whether it finds motorbikes at range, what it does with a
    rickshaw it has no class for, how often a roadside parked car ends up looking
    like it is in your lane.

    This answers those questions from a folder of JPEGs, with no phone, no car and
    no risk. It is the same chain carla_live.py runs — detector, tracker, distance,
    alert engine — pointed at still frames instead of a simulator.

TWO USES
    1. Right now: any real dashcam footage. Run it, look at the annotated frames,
       find out what breaks before you ever mount the phone.
    2. Later: the recording SessionRecorder pulls off the phone. Same folder
       layout, and with ego.csv it replays the drive with true per-frame speed.
       That is the tuning loop — drive once, replay a hundred times.

    ReplayRunner on the phone does the same thing in Java. Run both on the same
    folder and any divergence is a twin-drift bug — see run_parity.py.

FOCAL LENGTH MATTERS
    Every distance comes from  distance = realWidth * focal / boxWidthPx.  For
    your own recording, pass the focal the app measured (shown on its HUD). For
    footage from an unknown camera there is no correct value, so distances are
    indicative only — detection quality is still perfectly meaningful.

USAGE
    python replay_images.py --images path\\to\\frames
    python replay_images.py --images path\\to\\frames --annotate --limit 300
    python replay_images.py --images sessions\\replay\\20260810_1530\\frames \\
                            --ego sessions\\replay\\20260810_1530\\ego.csv \\
                            --focal 812 --annotate
"""

import argparse
import csv
import glob
import math
import os
import time

import cv2
import numpy as np

from adas_core import (AlertEngine, AlertType, DistanceEstimator, IouTracker)

KEEP_CLASSES = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, motorcycle, bus, truck
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
               5: "bus", 7: "truck"}
CONF = 0.35

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")


def load_frames(folder, limit):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths.sort()
    if limit:
        paths = paths[:limit]
    return paths


def load_ego(path, n):
    """ego.csv from SessionRecorder: frame,t_ms,ego_ms,lat,lon,limit_ms."""
    if not path or not os.path.exists(path):
        return None
    ego = [float("nan")] * n
    limit = [float("nan")] * n
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                i = int(row["frame"])
                if 0 <= i < n:
                    ego[i] = float(row["ego_ms"])
                    limit[i] = float(row["limit_ms"])
            except (KeyError, ValueError):
                continue
    return ego, limit


def draw(frame, tracks, ego_v, alert_name, hud):
    for t in tracks:
        if not t.is_confirmed:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.box]
        in_path = (not math.isnan(t.distance_m)
                   and DistanceEstimator.in_ego_corridor(
                       t.box, frame.shape[1], t.distance_m))
        oncoming = DistanceEstimator.is_oncoming(t.closing_speed_ms, ego_v)

        if in_path and oncoming:
            col = (0, 165, 255)
        elif in_path and t.ttc_s < 3.0:
            col = (0, 0, 255)
        elif in_path:
            col = (0, 255, 255)
        else:
            col = (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        label = CLASS_NAMES.get(t.class_id, "?")
        if not math.isnan(t.distance_m):
            label += f" {t.distance_m:.0f}m"
        cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for i, line in enumerate(hud):
        cv2.putText(frame, line, (12, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if alert_name:
        cv2.putText(frame, alert_name, (12, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of frames")
    ap.add_argument("--ego", default=None, help="ego.csv from SessionRecorder")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--focal", type=float, default=None,
                    help="FOCAL_PX in these images' pixels. Read it off the "
                         "app's HUD after calibrating.")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="capture rate of the source frames")
    ap.add_argument("--ego-speed", type=float, default=None,
                    help="constant ego speed m/s when there is no ego.csv")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames")
    ap.add_argument("--out", default="replay_out")
    ap.add_argument("--annotate", action="store_true",
                    help="also write annotated frames so you can eyeball them")
    args = ap.parse_args()

    paths = load_frames(args.images, args.limit)
    if not paths:
        raise SystemExit(f"no images found in {args.images}")

    first = cv2.imread(paths[0])
    if first is None:
        raise SystemExit(f"could not read {paths[0]}")
    h, w = first.shape[:2]

    if args.focal is not None:
        DistanceEstimator.FOCAL_PX = args.focal
        focal_note = f"{args.focal:.1f} (supplied)"
    else:
        focal_note = f"{DistanceEstimator.FOCAL_PX:.1f} (PLACEHOLDER - distances indicative only)"

    ego_data = load_ego(args.ego, len(paths))
    if ego_data is None and args.ego_speed is None:
        print("NOTE: no ego.csv and no --ego-speed. Assuming 15 m/s (54 km/h) so "
              "the speed gate does not mute everything.\n")

    from ultralytics import YOLO
    model = YOLO(args.model)

    os.makedirs(args.out, exist_ok=True)
    ann_dir = os.path.join(args.out, "annotated")
    if args.annotate:
        os.makedirs(ann_dir, exist_ok=True)

    tracker = IouTracker()
    engine = AlertEngine()

    dt_ms = int(1000.0 / args.fps)
    sim_ms = 100_000

    log_path = os.path.join(args.out, "replay_log.csv")
    log = open(log_path, "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["frame", "t_s", "ego_ms", "n_dets", "n_tracks", "near_dist_m",
                 "near_ttc_s", "alert", "reason", "infer_ms"])

    print(f"frames    : {len(paths)}")
    print(f"size      : {w}x{h}")
    print(f"focal_px  : {focal_note}")
    print(f"source fps: {args.fps}\n")

    class_counts = {}
    alerts = []
    det_total = 0
    infer_total = 0.0
    t_start = time.time()

    for i, p in enumerate(paths):
        frame = cv2.imread(p)
        if frame is None:
            continue

        t0 = time.time()
        res = model.predict(frame, conf=CONF, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000.0
        infer_total += infer_ms

        dets = []
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid not in KEEP_CLASSES:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append(((x1, y1, x2, y2), cid, float(b.conf[0])))
            class_counts[cid] = class_counts.get(cid, 0) + 1
        det_total += len(dets)

        tracks = tracker.update(dets)
        for t in tracks:
            if t.missed == 0:
                t.update_distance(
                    DistanceEstimator.estimate(t.box, t.class_id), sim_ms)

        if ego_data is not None and not math.isnan(ego_data[0][i]):
            ego_v = ego_data[0][i]
            limit_ms = ego_data[1][i]
        else:
            ego_v = args.ego_speed if args.ego_speed is not None else 15.0
            limit_ms = 13.9

        alert, _subject, reason = engine.evaluate(
            tracks, w, ego_v, limit_ms, None, sim_ms,
            obstacle_ttc=None, obstacle_dist=None, lane_cross=None)

        near = None
        for t in tracks:
            if not t.is_confirmed or math.isnan(t.distance_m):
                continue
            if not DistanceEstimator.in_ego_corridor(t.box, w, t.distance_m):
                continue
            if near is None or t.distance_m < near.distance_m:
                near = t

        if alert is not None:
            alerts.append((i, alert.name, reason))
            print(f"  frame {i:5d}  {alert.name:17s} {reason}")

        wr.writerow([i, round(i / args.fps, 2), round(ego_v, 2), len(dets),
                     len(tracks),
                     "" if near is None else round(near.distance_m, 2),
                     "" if near is None or math.isinf(near.ttc_s)
                        else round(near.ttc_s, 2),
                     alert.name if alert else "", reason, round(infer_ms)])

        if args.annotate:
            hud = [f"frame {i}  dets {len(dets)}  tracks {len(tracks)}",
                   f"ego {ego_v * 3.6:.0f} km/h  focal {DistanceEstimator.FOCAL_PX:.0f}"]
            vis = draw(frame.copy(), tracks, ego_v,
                       alert.name if alert else "", hud)
            cv2.imwrite(os.path.join(ann_dir, f"{i:06d}.jpg"), vis)

        sim_ms += dt_ms
        if i and i % 100 == 0:
            print(f"  ... {i}/{len(paths)}")

    log.close()

    n = len(paths)
    minutes = n / args.fps / 60.0
    print("\n================ SUMMARY ================")
    print(f"frames processed   : {n}  ({minutes:.1f} min of footage at {args.fps} fps)")
    print(f"mean inference     : {infer_total / max(1, n):.0f} ms/frame (desktop)")
    print(f"detections/frame   : {det_total / max(1, n):.2f}")
    print("by class           :")
    for cid, c in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {CLASS_NAMES.get(cid, cid):11s} {c:6d}"
              f"  ({c / max(1, n):.2f}/frame)")
    print(f"\nalerts             : {len(alerts)}")
    for name in sorted({a[1] for a in alerts}):
        print(f"    {name:17s} {sum(1 for a in alerts if a[1] == name)}")
    if minutes > 0:
        print(f"alert rate         : {len(alerts) / minutes:.2f} per minute")
    print(f"\nlog   -> {log_path}")
    if args.annotate:
        print(f"frames-> {ann_dir}")
    print(f"elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
