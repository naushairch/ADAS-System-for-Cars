"""
carla_collect_signs.py — auto-labelled YOLO dataset generator for speed-limit
sign detection.

WHY THIS EXISTS
    Same idea as carla_collect_lanes.py: hand-labelling thousands of frames
    of speed-limit signs is not a good use of anyone's time, and CARLA
    already knows exactly where every sign is and what it says. The map
    spawns real actors named `traffic.speed_limit.<N>` at every sign post -
    read their transform, read N off the actor's type_id, and project the
    sign face into the camera. No annotation, just geometry.

WHAT COUNTS AS A GOOD LABEL
    A speed-limit sign in CARLA is a thin, DOUBLE-SIDED disc: the front has
    the number, the back is a plain grey circle. Projecting the actor's 3D
    position gives you a box whether you are looking at the front or the
    back, so distance/angle alone is not enough - you must also check that
    the camera is on the side the sign actually faces. That check is the
    `facing_dot` test below. It is a geometric assumption (front = the
    actor's forward vector, i.e. the direction traffic reads the sign from)
    that has NOT been verified against a live simulator in this session -
    run with --preview or check_dataset.py on a small sample FIRST. If the
    contact sheet is full of grey backs-of-signs, flip the sign on
    FACING_MIN (make it negative) or swap which vector plays "forward".

CLASSES
    Fixed, not auto-discovered: SIGN_CLASSES = [30, 60, 90], matching every
    speed value this project's default CARLA towns have ever produced. YOLO
    class indices must stay stable across every collection run that feeds
    the same dataset directory, so an unrecognised speed value is skipped
    and counted rather than silently inserted with a new index (that would
    quietly renumber every label file written before it appeared).

WHAT GETS WRITTEN, per frame, into --out (YOLO detection format):
    images/train/000000.jpg, labels/train/000000.txt   (~90%)
    images/val/000000.jpg,   labels/val/000000.txt      (~10%)
    data.yaml     class list, for check_dataset.py / ultralytics training

USAGE
    python carla_collect_signs.py --out sign_yolo_data --minutes 20 --towns Town01,Town03,Town05
    python carla_collect_signs.py --out sign_yolo_data --minutes 2 --preview   # sanity check first
"""

import argparse
import os
import queue
import random
import time
from collections import Counter

import carla
import cv2
import numpy as np

from carla_collect_lanes import build_intrinsic, project_point

IMG_W, IMG_H = 1280, 720
FOV_DEG = 78.0
CAM_X, CAM_Y, CAM_Z = 0.8, 0.0, 1.25
CAM_PITCH = -5.0
FIXED_DT = 0.05

SIGN_CLASSES = [30, 40, 60, 90]       # class index == position in this list.
# Confirmed against Town04 via list_signs.py (30/60/90 already known from the
# old dataset's folder names; 40 was new). If a FUTURE town turns up another
# unknown value AFTER data already exists, append it to the END of this list
# instead of re-sorting - every value before it keeps the same class index
# that's already baked into every .txt label written so far.
SIGN_MAX_DIST_M = 45.0                # ignore signs further than this
SIGN_MIN_DIST_M = 2.0                 # camera-clipping range, not the sign itself
FACING_MIN = 0.3                      # dot(sign_normal, sign->camera); see below
MIN_BOX_PX = 8                        # boxes smaller than this teach nothing
MIN_VISIBLE_FRAC = 0.6                # fraction of the box that must stay in-frame
SIGN_MATCH_MAX_M = 5.0                # sign mesh -> speed-limit actor match radius

VAL_FRACTION = 0.10


def sign_speed_value(type_id):
    """'traffic.speed_limit.60' -> 60, or None if this isn't a speed-limit sign."""
    if not type_id.startswith("traffic.speed_limit."):
        return None
    try:
        return int(type_id.rsplit(".", 1)[-1])
    except ValueError:
        return None


def build_sign_targets(world):
    """Locate the real sign meshes and label each with its speed value.

    WHY NOT JUST USE THE ACTOR TRANSFORM (the original approach, which was
    wrong): a `traffic.speed_limit.*` actor is a TRIGGER VOLUME, not the sign
    you can see. Measured on Town04, its transform sits at road level
    (location.z == 0.00) and its bounding box spans the carriageway - up to
    7 x 55 m. Projecting that put every label on the tarmac and the grass
    verge, never on a sign.

    The visible geometry lives in the level's static meshes instead, via
    get_level_bbs(CityObjectLabel.TrafficSigns): 64 boxes on Town04, sitting
    at z ~= 1.9-2.2 m with sensible extents (~0.3 x 0.0 x 0.38 m). Those carry
    no speed value, so each mesh is matched to its nearest trigger actor,
    which does. Measured match distances on Town04: median 1.2 m, p90 2.6 m -
    so SIGN_MATCH_MAX_M rejects the one 32.7 m outlier (a sign mesh with no
    speed-limit actor near it - some other kind of traffic sign).
    """
    actors = [a for a in world.get_actors().filter("traffic.speed_limit.*")
              if sign_speed_value(a.type_id) in SIGN_CLASSES]
    targets, unmatched = [], 0
    for bb in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns):
        best, best_d = None, float("inf")
        for a in actors:
            al = a.get_transform().location
            d = ((al.x - bb.location.x) ** 2 + (al.y - bb.location.y) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, a
        if best is None or best_d > SIGN_MATCH_MAX_M:
            unmatched += 1
            continue

        # The sign is a thin plate: whichever local axis is flattest is the
        # one it faces along. extent.y == 0.00 on most Town04 signs, so the
        # normal is the box's own right vector; guard the other case anyway.
        rot = bb.rotation
        normal = (rot.get_right_vector() if bb.extent.y <= bb.extent.x
                  else rot.get_forward_vector())
        targets.append(dict(value=sign_speed_value(best.type_id),
                            bb=bb, normal=normal, location=bb.location))
    return targets, unmatched


class OcclusionTester:
    """Rejects signs that geometry says are in front of the camera but that
    the driver cannot actually see - hidden behind a hill, a wall, a building.

    WHY THIS MATTERS: without it, roughly 8% of labels in the first Town04
    collection sat on empty ground (verified on the contact sheet: boxes on
    bare hillside with no sign in the picture). Those teach the detector to
    invent a sign wherever the scenery merely resembles a place one could be -
    the worst possible failure for something that sets a speed limit.

    Casts one ray per candidate sign. Anything solid between camera and sign
    means occluded; the sign's own mesh and its pole obviously do not count.

    UNTESTED at the time of writing - written while a training run held the
    GPU, so no simulator was available to exercise it. Watch the first
    contact sheet after enabling it: too many rejections (a near-empty
    dataset) means the ray is hitting the sign's own geometry and IGNORED_
    LABELS needs widening; too few means cast_ray is not returning what this
    assumes. Falls back to accepting everything if the API misbehaves, so a
    wrong guess here cannot silently empty a collection run.
    """

    IGNORED_LABELS = ("TrafficSigns", "Poles", "None", "NONE", "Other")
    MARGIN_M = 1.0        # ignore hits within this distance of the sign itself

    def __init__(self, world):
        self.world = world
        self.ok = True
        self.failures = 0

    def is_occluded(self, cam_loc, sign_loc):
        if not self.ok:
            return False
        try:
            hits = self.world.cast_ray(cam_loc, sign_loc)
        except Exception as e:
            self.failures += 1
            if self.failures == 1:
                print(f"  [occlusion] cast_ray unavailable ({e}); occlusion "
                      f"checking disabled for this run", flush=True)
            self.ok = False
            return False

        d_sign = cam_loc.distance(sign_loc)
        for h in hits:
            if cam_loc.distance(h.location) >= d_sign - self.MARGIN_M:
                continue      # at or past the sign - not something in the way
            if str(h.label).rsplit(".", 1)[-1] in self.IGNORED_LABELS:
                continue
            return True
        return False


def project_sign_box(target, cam_loc, K, w2c, img_w, img_h):
    """Returns (x1, y1, x2, y2) in pixels, or (None, reason) if unusable."""
    loc = target["location"]
    to_cam = carla.Vector3D(cam_loc.x - loc.x, cam_loc.y - loc.y, cam_loc.z - loc.z)
    dist = (to_cam.x ** 2 + to_cam.y ** 2 + to_cam.z ** 2) ** 0.5
    if dist < SIGN_MIN_DIST_M or dist > SIGN_MAX_DIST_M:
        return None, "range"
    inv = 1.0 / max(dist, 1e-6)
    to_cam_n = carla.Vector3D(to_cam.x * inv, to_cam.y * inv, to_cam.z * inv)

    n = target["normal"]
    facing_dot = n.x * to_cam_n.x + n.y * to_cam_n.y + n.z * to_cam_n.z
    if facing_dot < FACING_MIN:
        return None, "back_of_sign"

    # Project the mesh box's own 8 world vertices. Passing an identity
    # Transform is deliberate: level bbs are already in world space, unlike
    # an actor's bounding box which is relative to its actor.
    pix = []
    for v in target["bb"].get_world_vertices(carla.Transform()):
        p = project_point(v, K, w2c)
        if p is None:
            return None, "behind_camera"
        pix.append(p)
    pix = np.array(pix)
    x1_raw, y1_raw = pix[:, 0].min(), pix[:, 1].min()
    x2_raw, y2_raw = pix[:, 0].max(), pix[:, 1].max()
    raw_area = max(1.0, (x2_raw - x1_raw) * (y2_raw - y1_raw))

    x1, y1 = max(0.0, x1_raw), max(0.0, y1_raw)
    x2, y2 = min(float(img_w), x2_raw), min(float(img_h), y2_raw)
    if x2 <= x1 or y2 <= y1:
        return None, "outside_frame"

    clipped_area = (x2 - x1) * (y2 - y1)
    if clipped_area / raw_area < MIN_VISIBLE_FRAC:
        return None, "mostly_clipped"
    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        return None, "too_small"

    return (x1, y1, x2, y2), None


class Writer:
    def __init__(self, out_dir, val_fraction):
        self.out_dir = out_dir
        self.val_fraction = val_fraction
        self.dirs = {}
        self.counts = {}
        for split in ("train", "val"):
            img_dir = os.path.join(out_dir, "images", split)
            lbl_dir = os.path.join(out_dir, "labels", split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            self.dirs[split] = (img_dir, lbl_dir)
            self.counts[split] = len(
                [f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.total = self.counts["train"] + self.counts["val"]

    def add(self, frame_bgr, yolo_rows):
        split = "val" if random.random() < self.val_fraction else "train"
        img_dir, lbl_dir = self.dirs[split]
        idx = self.counts[split]
        name = f"{idx:06d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(yolo_rows) + ("\n" if yolo_rows else ""))
        self.counts[split] += 1
        self.total += 1
        return split, idx

    def write_data_yaml(self):
        path = os.path.join(self.out_dir, "data.yaml")
        with open(path, "w") as f:
            f.write(f"path: {os.path.abspath(self.out_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(SIGN_CLASSES)}\n")
            f.write("names:\n")
            for i, v in enumerate(SIGN_CLASSES):
                f.write(f"  {i}: '{v}'\n")
        return path


def spawn_camera(world, ego):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMG_W))
    bp.set_attribute("image_size_y", str(IMG_H))
    bp.set_attribute("fov", str(FOV_DEG))
    tf = carla.Transform(carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
                         carla.Rotation(pitch=CAM_PITCH))
    cam = world.spawn_actor(bp, tf, attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    return cam, q


def collect_town(client, town, writer, args, K, deadline):
    world = client.get_world()

    # Only reload the map if we are not already on it. load_world() is the
    # single most fragile call in this whole script on some machines - it has
    # crashed CarlaUE4 outright, and switching straight into synchronous mode
    # after it produced a world whose camera never delivered a frame. If the
    # requested town is already up (very common: the previous run left it
    # there), skipping the call avoids all of that. carla_live.py never calls
    # load_world at all, which is a large part of why it has always worked.
    if args.load_towns and town not in world.get_map().name:
        print(f"[{town}] loading map (currently {world.get_map().name})...",
              flush=True)
        world = client.load_world(town)
        # Let the fresh map settle in ASYNCHRONOUS mode. wait_for_tick() waits
        # on the server's own clock, so this needs no ticking from us and
        # cannot wedge anything if the map is still streaming in.
        for _ in range(20):
            world.wait_for_tick()
        print(f"[{town}] map loaded and settled", flush=True)
    else:
        print(f"[{town}] already loaded ({world.get_map().name}) - "
              f"skipping load_world", flush=True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # EVERYTHING that happens while synchronous mode is on must run under this
    # try/finally. In sync mode the SERVER only advances when a client ticks
    # it, so any exit that skips the restore below - an exception, or even a
    # plain `return` because the map turned out to have no signs - leaves the
    # server wedged, waiting forever for ticks nobody is sending. Every later
    # RPC then blocks until it gives up:
    #     RuntimeError: time-out of 60000ms while waiting for the simulator
    # which reads like CARLA itself is broken, when in fact this script broke
    # it and the only cure is restarting CarlaUE4. That cost a whole debugging
    # session on 2026-08-08: the warm-up loop and the no-signs `return` both
    # sat OUTSIDE the old, narrower try, so both poisoned the server on the way
    # out. Keep this try as the first thing after sync mode goes on.
    actors = []
    try:
        _drive_and_collect(world, tm, actors, town, writer, args, K, deadline)
    finally:
        # Order matters: restore ASYNC first, destroy actors second. destroy()
        # is a blocking RPC, and in synchronous mode with nobody ticking it
        # waits forever - so cleaning up in the other order hangs here and
        # swallows whatever exception sent us into this finally in the first
        # place. That is exactly why a warm-up failure showed up as a silent
        # 150-second hang with no traceback at all.
        try:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
            print(f"[{town}] world restored to asynchronous mode", flush=True)
        except RuntimeError as e:
            print(f"[{town}] could not restore async mode: {e}", flush=True)
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass


def _drive_and_collect(world, tm, actors, town, writer, args, K, deadline):
    """Body of collect_town, split out purely so the caller's try/finally can
    wrap it whole. Every actor spawned here is appended to `actors` so the
    caller can destroy them however this returns - normally, early, or by
    raising."""
    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    sign_targets, unmatched = build_sign_targets(world)
    unknown = {sign_speed_value(a.type_id)
               for a in world.get_actors().filter("traffic.speed_limit.*")
               if sign_speed_value(a.type_id) not in SIGN_CLASSES}
    by_value = Counter(t["value"] for t in sign_targets)
    print(f"[{town}] {len(sign_targets)} sign meshes matched to a speed value "
          f"{dict(sorted(by_value.items()))}; {unmatched} mesh(es) unmatched; "
          f"{len(unknown)} unknown values seen: {unknown or '{}'}")
    if not sign_targets:
        print(f"[{town}] no usable signs on this map - skipping")
        return

    # NB: append to the caller's list, never rebind it - `actors = []` here
    # would leave collect_town's finally with an empty list and leak every
    # actor (including the camera) into the running simulator.
    ego = world.spawn_actor(bl.filter("vehicle.tesla.model3")[0], spawns[0])
    actors.append(ego)
    print(f"[{town}] ego spawned", flush=True)

    for sp in spawns[1:1 + args.traffic]:
        try:
            npc = world.spawn_actor(random.choice(bl.filter("vehicle.*")), sp)
            npc.set_autopilot(True, tm.get_port())
            actors.append(npc)
        except Exception:
            pass
    print(f"[{town}] spawned {len(actors) - 1} traffic vehicles", flush=True)

    cam, q = spawn_camera(world, ego)
    actors.append(cam)
    print(f"[{town}] camera spawned, warming up...", flush=True)

    ego.set_autopilot(True, tm.get_port())
    print(f"[{town}] ego autopilot on", flush=True)
    tm.ignore_lights_percentage(ego, 100)
    print(f"[{town}] traffic manager configured", flush=True)

    # The FIRST tick after a sensor is spawned legitimately delivers no image -
    # the sensor only comes online on the tick AFTER it registers. Measured on
    # this setup: tick 0 empty, ticks 1+ deliver reliably. So an empty queue is
    # only fatal if nothing at all has arrived by the end of warm-up. Treating
    # the first empty window as an error - the previous behaviour - failed
    # every single run at "warm-up 0" while the camera was in fact healthy.
    frames_seen = 0
    for i in range(20):
        # A tick that times out means the SERVER stopped answering; an empty
        # queue after a good tick means the server is fine but the CAMERA is
        # not producing. Keep the two apart, they have different causes.
        try:
            world.tick()
        except RuntimeError as e:
            raise RuntimeError(
                f"[{town}] the server stopped responding to tick() on warm-up "
                f"tick {i}: {e}\n"
                f"If CarlaUE4 has a visible window, check it is not minimised "
                f"or hidden behind other windows - UE4 throttles rendering "
                f"when unfocused, and sync-mode ticks then never complete. "
                f"Launching CarlaUE4.exe with -RenderOffScreen avoids this "
                f"entirely.") from e
        try:
            q.get(timeout=5.0)
            frames_seen += 1
        except queue.Empty:
            pass
    if frames_seen == 0:
        raise RuntimeError(
            f"[{town}] the server ticked 20 times but the camera never "
            f"produced a single frame.")
    print(f"[{town}] warm-up ok ({frames_seen}/20 frames), collecting for "
          f"{args.minutes:.1f} min", flush=True)

    occlusion = None if args.no_occlusion_check else OcclusionTester(world)
    if occlusion is None:
        print(f"[{town}] occlusion checking OFF", flush=True)

    respawn_every = 30.0 / FIXED_DT
    tick = 0
    saved_here = 0
    skip_counts = Counter()
    last_heartbeat = time.time()
    try:
        while time.time() < deadline:
            world.tick()
            try:
                image = q.get(timeout=2.0)
            except Exception:
                continue
            tick += 1

            if tick % args.every != 0:
                continue

            if time.time() - last_heartbeat > 3.0:
                print(f"  [{town}] tick {tick}  saved {saved_here} "
                      f"({writer.total} total)  skipped={dict(skip_counts)}",
                      flush=True)
                last_heartbeat = time.time()

            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()

            ego_tf = ego.get_transform()
            ego_fwd = ego_tf.get_forward_vector()
            cam_tf = cam.get_transform()
            cam_loc = cam_tf.location
            w2c = np.array(cam_tf.get_inverse_matrix())

            rows = []
            vis_boxes = []
            for tgt in sign_targets:
                d = tgt["location"] - ego_tf.location
                along = d.x * ego_fwd.x + d.y * ego_fwd.y
                if along <= 0 or along > SIGN_MAX_DIST_M:
                    continue

                box, reason = project_sign_box(tgt, cam_loc, K, w2c, IMG_W, IMG_H)
                if box is None:
                    skip_counts[reason] += 1
                    continue

                # Last, not first: it is the most expensive test (a ray cast
                # into the world) and the cheap geometric ones above have
                # already discarded the great majority of candidates.
                if occlusion is not None and occlusion.is_occluded(
                        cam_loc, tgt["location"]):
                    skip_counts["occluded"] += 1
                    continue

                x1, y1, x2, y2 = box
                cls = SIGN_CLASSES.index(tgt["value"])
                cx = (x1 + x2) / 2.0 / IMG_W
                cy = (y1 + y2) / 2.0 / IMG_H
                bw = (x2 - x1) / IMG_W
                bh = (y2 - y1) / IMG_H
                rows.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                vis_boxes.append((x1, y1, x2, y2, SIGN_CLASSES[cls]))

            if rows:
                split, idx = writer.add(frame, rows)
                saved_here += 1
            else:
                skip_counts["no_sign_in_view"] += 1

            if args.preview:
                vis = frame.copy()
                for x1, y1, x2, y2, val in vis_boxes:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                                 (0, 0, 255), 2)
                    cv2.putText(vis, str(val), (int(x1), max(18, int(y1) - 6)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("sign collect preview (ESC to stop early)", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if tick % respawn_every == 0:
                ego.set_transform(random.choice(spawns))
    finally:
        # Actor cleanup and the sync-mode restore both live in collect_town's
        # finally now - this one only reports.
        print(f"[{town}] done: saved {saved_here}  skipped={dict(skip_counts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="sign_yolo_data")
    ap.add_argument("--towns", default="", help="comma-separated, e.g. "
                    "Town01,Town03,Town05 - empty uses whatever is already loaded. "
                    "NOTE: this uses client.load_world(), which has been observed "
                    "to crash CarlaUE4 outright on some machines. If it crashes, "
                    "omit --towns, launch CarlaUE4.exe <TownName> directly instead, "
                    "and run this script once per town - the dataset is resumable "
                    "so results across launches accumulate normally.")
    ap.add_argument("--minutes", type=float, default=20.0,
                    help="wall-clock minutes of driving PER TOWN")
    ap.add_argument("--traffic", type=int, default=8)
    ap.add_argument("--every", type=int, default=4,
                    help="save every Nth simulator tick (20Hz / N)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--no-occlusion-check", action="store_true",
                    help="keep signs hidden behind terrain/buildings. Only for "
                    "comparing against the pre-occlusion-filter dataset - "
                    "those labels sit on empty ground and teach the model to "
                    "hallucinate signs")
    ap.add_argument("--preview", action="store_true",
                    help="show a live box-overlay window while collecting - "
                    "use this on a short run FIRST, see module docstring")
    args = ap.parse_args()
    args.load_towns = bool(args.towns)

    towns = [t.strip() for t in args.towns.split(",") if t.strip()] or ["current"]

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    K = build_intrinsic(IMG_W, IMG_H, FOV_DEG)
    writer = Writer(args.out, args.val_fraction)
    print(f"writing to {args.out} (resuming from {writer.total} frames: "
          f"{writer.counts['train']} train / {writer.counts['val']} val)")
    try:
        for town in towns:
            print(f"[{town}] collecting for {args.minutes:.1f} min")
            deadline = time.time() + args.minutes * 60.0
            collect_town(client, town, writer, args, K, deadline)
    finally:
        path = writer.write_data_yaml()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"\ndone. {writer.total} labelled frames in {args.out}")
        print(f"wrote {path}")
        print("\nNow run: python check_dataset.py   (look BEFORE you train)")


if __name__ == "__main__":
    main()
