"""
carla_live.py — drive in CARLA and hear the ADAS speak, live.

Runs the full pipeline in real time:

    YOLO vehicles  -> tracker -> distance -> time-to-contact
    depth model    -> obstacles standing in the corridor -> stopping distance
    optical flow   -> looming objects (comparison only)
    lane lines     -> solid-line touch warning
    sign YOLO      -> speed-limit sign -> auto speed limit -> reduce-speed warning
    GPS speed      -> speed limit warning (auto from signs, or manual - press L)

Sign detection needs a trained model at --sign-model (default sign_yolov8n.pt).
See carla_collect_signs.py and train_sign_yolo.py. Without one, the limit
stays manual, same as before.

--------------------------------------------------------------------------------
CONTROLS
    W          go forward
    S          BRAKE while moving; REVERSE once the car has stopped
    A / D      steer
    SPACE      handbrake
    P          autopilot on/off
    R          teleport to a fresh spot
    ESC        quit

--------------------------------------------------------------------------------
RUNNING

    Everything on:
        python carla_live.py --traffic 15 --lanes --depth-obstacles

    Add the optical-flow detector for comparison:
        python carla_live.py --traffic 15 --lanes --depth-obstacles --obstacles

    See why the obstacle path did or did not warn:
        ... --debug-obstacle
--------------------------------------------------------------------------------
"""

import argparse
import csv
import math
import os
import queue
import random
import time

import numpy as np
import pygame
import cv2
import carla
from ultralytics import YOLO

from adas_core import (AlertEngine, AlertType, DistanceEstimator, IouTracker,
                       ground_truth_focal)
from lane_detect import LaneDetector
from obstacle_flow import LoomingDetector, set_sensitivity as set_looming_sensitivity
from depth_obstacle import DepthObstacleDetector
from trip_summary import TripStats, build_report, print_report, save_report, HISTORY_PATH

IMG_W, IMG_H = 1280, 720
FOV_DEG = 78.0
CAM_X, CAM_Y, CAM_Z = 0.8, 0.0, 1.25
CAM_PITCH = -5.0

FIXED_DT = 0.05          # 20 Hz
KEEP_CLASSES = {0, 1, 2, 3, 5, 7}   # person, bicycle, car, motorcycle, bus, truck
CONF = 0.35

STOPPED_MS = 0.6         # below this speed we consider the car stopped

SIGN_CONF = 0.5
# A single frame's detection is cheap to get by chance (a round blue sign, a
# wheel, whatever). Require the SAME value to win N processed frames in a row
# before it overrides the current limit - the same "don't act on a blip"
# reasoning as AlertEngine.SPEED_CONFIRM_FRAMES, just for what the limit IS
# rather than whether the car is over it.
SIGN_CONFIRM_FRAMES = 3


class Sounds:
    """Plays the same WAVs the phone uses. Falls back to console text."""

    def __init__(self, raw_dir):
        self.ok = False
        self.clips = {}
        self.chime = None
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            chime = os.path.join(raw_dir, "chime.wav")
            if os.path.exists(chime):
                self.chime = pygame.mixer.Sound(chime)
            for t in AlertType:
                p = os.path.join(raw_dir, f"en_{t.audio_key}.wav")
                if os.path.exists(p):
                    self.clips[t] = pygame.mixer.Sound(p)
            self.ok = bool(self.clips)
        except Exception as e:
            print(f"audio disabled ({e}); alerts will print instead")
        if not self.ok:
            print("NOTE: run generate_alerts.py for real voice alerts.")

    def play(self, alert_type):
        if not self.ok:
            print(f"  >>> {alert_type.name}")
            return
        pygame.mixer.stop()
        if self.chime:
            self.chime.play()
        clip = self.clips.get(alert_type)
        if clip:
            clip.play()


def _ls(v):
    return "solid" if v is True else ("broken" if v is False else "?")


def _short(marking_type):
    t = str(marking_type)
    if "Solid" in t:
        return "solid"
    if "Broken" in t:
        return "broken"
    return t.lower()[:6]


def speed_ms(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def gap_metres(ego, other):
    c = ego.get_transform().location.distance(other.get_transform().location)
    return max(0.0, c - (ego.bounding_box.extent.x + other.bounding_box.extent.x))


def true_nearest_ahead(world, ego, max_lateral=2.0, max_range=90.0):
    """Ground truth: closest vehicle actually in front and roughly in our lane."""
    et = ego.get_transform()
    fwd = et.get_forward_vector()
    right = carla.Location(x=-fwd.y, y=fwd.x, z=0)
    best, best_d = None, max_range
    for a in world.get_actors().filter("vehicle.*"):
        if a.id == ego.id:
            continue
        d = a.get_transform().location - et.location
        along = d.x * fwd.x + d.y * fwd.y
        lateral = d.x * right.x + d.y * right.y
        if along <= 0 or along > max_range or abs(lateral) > max_lateral:
            continue
        g = gap_metres(ego, a)
        if g < best_d:
            best_d, best = g, a
    return best, (best_d if best else -1.0)


def draw(frame, tracks, hud_lines, banner, ego_v=0.0):
    for t in tracks:
        if not t.is_confirmed:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.box]
        in_path = (not math.isnan(t.distance_m)
                   and DistanceEstimator.in_ego_corridor(
                       t.box, frame.shape[1], t.distance_m))
        oncoming = DistanceEstimator.is_oncoming(t.closing_speed_ms, ego_v)

        if in_path and oncoming:
            col = (0, 165, 255)          # orange: in our cone but coming AT us
        elif in_path and t.ttc_s < 3.0:
            col = (0, 0, 255)            # red: real hazard
        elif in_path:
            col = (0, 255, 255)          # yellow: in our path, no danger yet
        else:
            col = (0, 200, 0)            # green: not our problem
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if col == (0, 0, 255) else 2)

        lab = (f"#{t.id}" if math.isnan(t.distance_m)
               else f"#{t.id} {t.distance_m:.0f}m "
                    f"{'-' if math.isinf(t.ttc_s) else f'{t.ttc_s:.1f}s'}"
                    f"{' ONCOMING' if oncoming and in_path else ''}")
        cv2.putText(frame, lab, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    half = (1.6 * DistanceEstimator.FOCAL_PX) / 20.0
    cx = frame.shape[1] // 2
    cv2.line(frame, (int(cx - half), frame.shape[0]),
             (int(cx - half * 0.25), int(frame.shape[0] * 0.55)), (255, 200, 0), 2)
    cv2.line(frame, (int(cx + half), frame.shape[0]),
             (int(cx + half * 0.25), int(frame.shape[0] * 0.55)), (255, 200, 0), 2)

    for i, line in enumerate(hud_lines):
        if not line:
            continue
        cv2.putText(frame, line, (16, 32 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if banner:
        cv2.putText(frame, banner, (16, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--traffic", type=int, default=25)
    ap.add_argument("--autopilot", action="store_true")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--raw", default=os.path.join(os.path.dirname(__file__), "raw"))
    ap.add_argument("--log", default="carla_live_log.csv")
    ap.add_argument("--speed-limit", type=float, default=50.0, help="km/h")

    ap.add_argument("--sign-model",
                    default=os.path.join(os.path.dirname(__file__), "sign_yolov8n.pt"),
                    help="speed-limit sign detector - see carla_collect_signs.py "
                    "and train_sign_yolo.py")
    ap.add_argument("--no-sign-detect", action="store_true",
                    help="ignore --sign-model even if it exists; limit stays manual")
    ap.add_argument("--sign-every", type=int, default=2,
                    help="run the sign model every Nth frame")

    ap.add_argument("--lanes", action="store_true",
                    help="solid-line detection and touch warning")
    ap.add_argument("--depth-obstacles", action="store_true",
                    help="obstacle detection from monocular depth")
    ap.add_argument("--depth-every", type=int, default=1,
                    help="run the depth model every Nth frame")
    ap.add_argument("--obstacles", action="store_true",
                    help="optical-flow looming detector (comparison only)")
    ap.add_argument("--obstacle-sensitivity", type=int, default=2,
                    choices=[1, 2, 3, 4],
                    help="1=very strict .. 4=sensitive")
    ap.add_argument("--debug-obstacle", action="store_true",
                    help="print why the obstacle path did or did not warn")
    args = ap.parse_args()

    DistanceEstimator.FOCAL_PX = ground_truth_focal(IMG_W, FOV_DEG)
    print(f"FOCAL_PX from simulator geometry: {DistanceEstimator.FOCAL_PX:.1f}")

    print("loading YOLO...")
    model = YOLO(args.model)

    sign_model = None
    if args.no_sign_detect:
        print("sign detection disabled (--no-sign-detect); limit stays manual")
    elif os.path.exists(args.sign_model):
        print("loading sign model...")
        sign_model = YOLO(args.sign_model)
    else:
        print(f"no sign model at {args.sign_model} - limit stays manual (press L). "
              "run carla_collect_signs.py then train_sign_yolo.py to enable "
              "auto speed-limit detection.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    actors = []
    ego = world.spawn_actor(bl.filter("vehicle.tesla.model3")[0], spawns[0])
    actors.append(ego)

    for sp in spawns[1:1 + args.traffic]:
        try:
            npc = world.spawn_actor(random.choice(bl.filter("vehicle.*")), sp)
            npc.set_autopilot(True, tm.get_port())
            actors.append(npc)
        except Exception:
            pass
    print(f"spawned {len(actors) - 1} traffic vehicles")

    cam_bp = bl.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(FOV_DEG))
    # DO NOT set sensor_tick here. In synchronous mode it makes the camera skip
    # ticks, the frame queue stays empty, and q.get() blocks forever.
    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
                        carla.Rotation(pitch=CAM_PITCH)),
        attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    actors.append(cam)

    lanes = None
    if args.lanes:
        lanes = LaneDetector(IMG_W, IMG_H)
        lanes.set_vehicle_half_width(ego.bounding_box.extent.y)
        print(f"lane touch threshold: {lanes.touch_threshold_m:.2f} m from "
              f"centreline (vehicle half-width {ego.bounding_box.extent.y:.2f} m)")

    looming = None
    if args.obstacles:
        set_looming_sensitivity(args.obstacle_sensitivity)
        looming = LoomingDetector(IMG_W, IMG_H, FIXED_DT)
        print(f"optical-flow detector ON at sensitivity "
              f"{args.obstacle_sensitivity}/4")

    depth_obs = None
    if args.depth_obstacles:
        depth_obs = DepthObstacleDetector(
            IMG_W, IMG_H, DistanceEstimator.FOCAL_PX,
            cam_height_m=CAM_Z, cam_pitch_deg=CAM_PITCH, dt=FIXED_DT)
        if not depth_obs.ok:
            depth_obs = None
            print("depth obstacles unavailable - continuing without")

    print(f"\nfeatures: collision=ON  "
          f"lanes={'ON' if args.lanes else 'off'}  "
          f"depth-obstacles={'ON' if depth_obs else 'off'}  "
          f"flow-obstacles={'ON' if looming else 'off'}")

    pygame.init()
    pygame.display.set_mode((360, 130))
    pygame.display.set_caption("ADAS control - click here for keyboard")
    sounds = Sounds(args.raw)

    tracker = IouTracker()
    engine = AlertEngine()
    autopilot = args.autopilot
    ego.set_autopilot(autopilot, tm.get_port())

    limit_kmh = args.speed_limit
    limit_source = "manual"
    limits = [30, 50, 60, 80, 100, 120]
    limit_idx = limits.index(50) if 50 in limits else 1
    sign_pending_val, sign_pending_count = None, 0
    sign_box = None   # last detected sign's (x1,y1,x2,y2,val) for the HUD/overlay

    sim_ms = 100_000
    banner, banner_until = "", 0
    frame_i = 0
    n_alerts, n_false = 0, 0
    trip = TripStats()
    reverse_held = False
    last_depth = (None, None, 0)

    log = open(args.log, "w", newline="")
    w = csv.writer(log)
    w.writerow(["frame", "sim_t_s", "ego_kmh", "true_dist_m", "est_dist_m", "err_m",
                "est_ttc_s", "n_tracks", "limit_kmh", "limit_source",
                "lane_left", "lane_left_true", "lane_right", "lane_right_true",
                "depth_dist_m", "flow_ttc_s", "alert", "reason", "infer_ms"])

    cv2.namedWindow("ADAS - CARLA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS - CARLA", 1100, 620)

    print("\nW forward | S brake, then reverse when stopped | A/D steer")
    print("P autopilot | R reset | L cycle speed limit | ESC quit")
    print("Click the small pygame window for keyboard control.\n")

    try:
        clock = pygame.time.Clock()
        while True:
            quit_now = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    quit_now = True
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        quit_now = True
                    elif e.key == pygame.K_p:
                        autopilot = not autopilot
                        ego.set_autopilot(autopilot, tm.get_port())
                        print(f"autopilot {'ON' if autopilot else 'OFF'}")
                    elif e.key == pygame.K_r:
                        ego.set_transform(random.choice(spawns))
                        tracker.reset()
                    elif e.key == pygame.K_l:
                        limit_idx = (limit_idx + 1) % len(limits)
                        limit_kmh = limits[limit_idx]
                        limit_source = "manual"
                        sign_pending_val, sign_pending_count = None, 0
                        print(f"speed limit set to {limit_kmh} km/h (manual)")
            if quit_now:
                break

            ego_v = speed_ms(ego)

            if not autopilot:
                k = pygame.key.get_pressed()
                c = carla.VehicleControl()
                stopped = ego_v < STOPPED_MS

                # S does two jobs. With speed still on the car it brakes; once
                # stopped, holding S drives backwards. reverse_held latches so
                # the gear does not flip the instant the car rolls backwards.
                if k[pygame.K_s]:
                    if stopped or reverse_held:
                        reverse_held = True
                        c.reverse = True
                        c.throttle = 0.45
                        c.brake = 0.0
                    else:
                        c.reverse = False
                        c.throttle = 0.0
                        c.brake = 0.85
                else:
                    reverse_held = False
                    c.reverse = False
                    c.throttle = 0.75 if k[pygame.K_w] else 0.0
                    c.brake = 0.0

                if k[pygame.K_w] and reverse_held:
                    reverse_held = False
                    c.reverse = False
                    c.throttle = 0.75
                    c.brake = 0.0

                c.steer = (-0.4 if k[pygame.K_a] else 0.0) + (0.4 if k[pygame.K_d] else 0.0)
                c.hand_brake = k[pygame.K_SPACE]
                ego.apply_control(c)

            world.tick()
            try:
                image = q.get(timeout=2.0)
            except queue.Empty:
                print("  (no camera frame this tick - skipping)")
                continue

            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()

            # ---- vehicles ----
            t0 = time.time()
            res = model.predict(frame, conf=CONF, verbose=False)[0]
            infer_ms = int((time.time() - t0) * 1000)

            dets = []
            for b in res.boxes:
                cid = int(b.cls[0])
                if cid not in KEEP_CLASSES:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append(((x1, y1, x2, y2), cid, float(b.conf[0])))

            tracks = tracker.update(dets)
            for t in tracks:
                if t.missed == 0:
                    t.update_distance(DistanceEstimator.estimate(t.box, t.class_id),
                                      sim_ms)

            veh_boxes = [t.box for t in tracks if t.is_confirmed]

            # ---- speed-limit sign ----
            if sign_model is not None and frame_i % args.sign_every == 0:
                sres = sign_model.predict(frame, conf=SIGN_CONF, verbose=False)[0]
                best_val, best_conf, best_box = None, 0.0, None
                for b in sres.boxes:
                    conf = float(b.conf[0])
                    if conf <= best_conf:
                        continue
                    try:
                        val = int(sign_model.names[int(b.cls[0])])
                    except (KeyError, ValueError):
                        continue
                    best_val, best_conf, best_box = val, conf, [float(v) for v in b.xyxy[0]]

                if best_val is not None:
                    sign_box = (*best_box, best_val)
                    if best_val == sign_pending_val:
                        sign_pending_count += 1
                    else:
                        sign_pending_val, sign_pending_count = best_val, 1
                    if (sign_pending_count >= SIGN_CONFIRM_FRAMES
                            and (limit_kmh != best_val or limit_source != "sign")):
                        limit_kmh = best_val
                        limit_source = "sign"
                        print(f"t={frame_i * FIXED_DT:6.2f}s  speed limit sign: "
                              f"{best_val} km/h", flush=True)
                else:
                    sign_pending_val, sign_pending_count = None, 0
                    sign_box = None

            # ---- lanes ----
            lane_info = (lanes.process(frame) if lanes is not None
                         else dict(left_solid=None, right_solid=None,
                                   left_offset_m=None, right_offset_m=None,
                                   left_fill=0.0, right_fill=0.0,
                                   crossed_solid=None, confidence=0.0,
                                   touching=False))

            # ---- obstacles ----
            depth_dist = depth_ttc = None
            depth_px = 0
            if depth_obs is not None and (frame_i % args.depth_every == 0):
                depth_dist, depth_ttc, depth_px = depth_obs.process(
                    frame, ego_v, veh_boxes)
                last_depth = (depth_dist, depth_ttc, depth_px)
            elif depth_obs is not None:
                depth_dist, depth_ttc, depth_px = last_depth

            look_ttc, look_n = (looming.process(frame, veh_boxes, ego_v)
                                if looming is not None else (None, 0))

            # ---- alerts ----
            alert, subject, reason = engine.evaluate(
                tracks, IMG_W, ego_v, limit_kmh / 3.6, None, sim_ms,
                obstacle_ttc=None, obstacle_dist=depth_dist,
                lane_cross=lane_info["crossed_solid"])

            gt_actor, gt_dist = true_nearest_ahead(world, ego)
            wp = world.get_map().get_waypoint(ego.get_transform().location)
            true_left = str(wp.left_lane_marking.type) if wp else "?"
            true_right = str(wp.right_lane_marking.type) if wp else "?"

            est = None
            for t in tracks:
                if not t.is_confirmed or math.isnan(t.distance_m):
                    continue
                if not DistanceEstimator.in_ego_corridor(t.box, IMG_W, t.distance_m):
                    continue
                if DistanceEstimator.is_oncoming(t.closing_speed_ms, ego_v):
                    continue
                if est is None or t.distance_m < est.distance_m:
                    est = t

            est_d = est.distance_m if est else -1.0
            err = (est_d - gt_dist) if (est and gt_dist > 0) else float("nan")

            if alert:
                sounds.play(alert)
                banner, banner_until = alert.name, sim_ms + 1500
                n_alerts += 1
                trip.record(alert)
                print(f"t={frame_i * FIXED_DT:6.2f}s  {alert.name:17s} {reason}",
                      flush=True)

            w.writerow([frame_i, round(frame_i * FIXED_DT, 2), round(ego_v * 3.6, 1),
                        round(gt_dist, 2), round(est_d, 2),
                        "" if math.isnan(err) else round(err, 2),
                        round(est.ttc_s, 2) if est and not math.isinf(est.ttc_s) else -1,
                        len(tracks), limit_kmh, limit_source,
                        _ls(lane_info["left_solid"]), _short(true_left),
                        _ls(lane_info["right_solid"]), _short(true_right),
                        "" if depth_dist is None else round(depth_dist, 2),
                        "" if look_ttc is None else round(look_ttc, 2),
                        alert.name if alert else "", reason, infer_ms])

            gear = "R" if reverse_held else ("A" if autopilot else "D")
            hud = [
                f"{ego_v * 3.6:5.1f} km/h  [{gear}]   tracks {len(tracks):2d}   "
                f"inf {infer_ms:3d}ms",
                f"vehicle: est {est_d:6.1f} m   true {gt_dist:6.1f} m   "
                f"err {'--' if math.isnan(err) else f'{err:+.1f}'} m",
                (f"lane L {_ls(lane_info['left_solid'])}/{_short(true_left)}  "
                 f"R {_ls(lane_info['right_solid'])}/{_short(true_right)}  "
                 f"{'ON LINE' if lane_info.get('touching') else ''}")
                if lanes is not None else "",
                (f"obstacle {'--' if depth_dist is None else f'{depth_dist:.0f}m'}"
                 f"   need {AlertEngine.warn_distance(ego_v):.0f}m to stop"
                 + (f"   flow {look_ttc:.1f}s" if look_ttc is not None else ""))
                if (depth_obs or looming) else "",
                f"limit {limit_kmh:.0f} km/h ({limit_source}, press L)   "
                f"alerts {n_alerts}",
            ]

            vis = draw(frame, tracks, hud,
                       banner if sim_ms < banner_until else "", ego_v)
            if depth_obs is not None:
                vis = depth_obs.overlay(vis, depth_dist, depth_ttc, depth_px)
            if looming is not None:
                vis = looming.overlay(vis, look_ttc, look_n)
            if sign_box is not None:
                x1, y1, x2, y2, val = sign_box
                confirmed = sign_pending_count >= SIGN_CONFIRM_FRAMES
                col = (0, 255, 0) if confirmed else (0, 200, 255)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(vis, f"{val} km/h", (int(x1), max(18, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.imshow("ADAS - CARLA", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if args.debug_obstacle and frame_i % 10 == 0:
                dbg = engine.obs_debug
                d = dbg.get("dist")
                print(f"  [obstacle] "
                      f"dist={'--' if d is None else f'{d:5.1f}m'}  "
                      f"speed={dbg.get('speed', 0) * 3.6:5.1f}km/h  "
                      f"warn_at={dbg.get('warn_d', 0):5.1f}m  "
                      f"brake_at={dbg.get('urgent_d', 0):5.1f}m  "
                      f"stage={dbg.get('stage', 0)}  "
                      f"-> {dbg.get('reason') or 'ALERT CONDITIONS MET'}",
                      flush=True)

            if frame_i % 20 == 0:
                print(f"  frame {frame_i:5d}  {ego_v * 3.6:5.1f} km/h  "
                      f"tracks {len(tracks):2d}  inf {infer_ms:4d}ms", flush=True)

            sim_ms += int(FIXED_DT * 1000)
            frame_i += 1
            clock.tick(60)

    finally:
        # Trip report first, before any CARLA-native cleanup calls below —
        # those can hard-crash the process (native crash, no Python
        # traceback possible) if the server connection is in a bad state,
        # and we don't want that to cost us the trip summary.
        print(f"\nframes={frame_i}  alerts={n_alerts}", flush=True)
        try:
            report = build_report(trip)
            print_report(report)
            save_report(report)
            print(f"trip history -> {HISTORY_PATH}", flush=True)
        except Exception:
            import traceback
            print("[trip_summary] failed to build/save trip report:", flush=True)
            traceback.print_exc()

        log.close()
        print(f"log -> {args.log}", flush=True)
        cv2.destroyAllWindows()
        pygame.quit()
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass
        original.synchronous_mode = False
        original.fixed_delta_seconds = None
        world.apply_settings(original)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
