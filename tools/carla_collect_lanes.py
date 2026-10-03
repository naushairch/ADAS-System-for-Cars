"""
carla_collect_lanes.py — auto-labelled dataset generator for the SCNN lane
model (scnn_lane/model.py).

WHY THIS EXISTS
    SCNN needs, per frame: where the lane lines are (a pixel mask) and
    whether each one is solid or broken. Nobody has hand-labelled that for
    this project, and hand-labelling thousands of frames is not a good use
    of anyone's time. CARLA already knows the answer exactly - every
    waypoint carries the true lane marking type:

        wp.left_lane_marking.type / wp.right_lane_marking.type

    So instead of annotating, we DRIVE and read the ground truth off the
    map API every frame. The only work left is projecting the lane
    boundary's 3D position into the camera image, which is plain pinhole
    geometry.

WHAT GETS WRITTEN, per frame, into --out:
    images/000000.jpg   the bird's-eye warp of the camera frame (same warp
                         lane_detect.py uses at inference, so training and
                         inference see identical geometry)
    masks/000000.png    single-channel label image: 0 background,
                         1 left line, 2 right line
    labels.csv          frame, left_label, right_label (0 absent /
                         1 broken / 2 solid), town

DRIVING STRATEGY
    Plain autopilot mostly hugs lane centre, which barely touches a line.
    We specifically want frames where the car is CLOSE to or ON a line -
    that is the exact condition the model has to be good at - so the
    traffic manager is told to make random lane changes fairly often.

USAGE
    python carla_collect_lanes.py --out lane_data --minutes 20 --towns Town01,Town03,Town05
"""

import argparse
import csv
import os
import queue
import random
import time
from collections import Counter

import carla
import cv2
import numpy as np

from lane_detect import (BEV_W, BEV_H, build_warp_matrix)

IMG_W, IMG_H = 1280, 720
FOV_DEG = 78.0
CAM_X, CAM_Y, CAM_Z = 0.8, 0.0, 1.25
CAM_PITCH = -5.0
FIXED_DT = 0.05

LANE_WAYPOINT_STEP_M = 1.5
LANE_HORIZON_M = 45.0
LINE_THICKNESS_PX = 11     # ~ paint width in BEV pixels, see HALF_LINE_WIDTH_M

# absent / broken / solid
LABEL_ABSENT, LABEL_BROKEN, LABEL_SOLID = 0, 1, 2


def _classify_marking(marking_type):
    """Same coarse classification carla_live.py already uses for its ground
    truth column, kept identical so the two stay comparable."""
    t = str(marking_type)
    if "Solid" in t:
        return LABEL_SOLID
    if "Broken" in t:
        return LABEL_BROKEN
    return LABEL_ABSENT


def build_intrinsic(w, h, fov_deg):
    K = np.identity(3, dtype=np.float64)
    focal = w / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def project_point(loc, K, w2c):
    """3D world point -> 2D pixel, or None if behind the camera."""
    p = np.array([loc.x, loc.y, loc.z, 1.0])
    p_cam = w2c @ p
    # UE4 world axes (x fwd, y right, z up) -> camera optical axes
    # (x right, y down, z forward).
    p_cam = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    if p_cam[2] <= 0.05:
        return None
    p_img = K @ p_cam
    return p_img[:2] / p_img[2]


def lane_boundary_chain(wp0, step=LANE_WAYPOINT_STEP_M, horizon=LANE_HORIZON_M):
    """
    Walk forward from the ego's waypoint, returning parallel lists of
    (left_3d_point, right_3d_point) for each step, plus the label pair read
    at the START of the chain (marking type can change further down the
    road, but the frame is labelled by what the ego is looking at now).
    """
    left_label = _classify_marking(wp0.left_lane_marking.type)
    right_label = _classify_marking(wp0.right_lane_marking.type)

    chain = [wp0]
    travelled = 0.0
    base_width = wp0.lane_width
    while travelled < horizon:
        nxt = chain[-1].next(step)
        if not nxt:
            break
        cand = nxt[0]
        # Junctions - and the crosswalks painted at their approach - break the
        # "just walk forward along one lane" assumption behind this whole
        # function: next() can return an ambiguous successor (a connecting
        # or crossing road), lane width can jump, and the boundary point
        # swings sideways as a result. That is exactly what painted a lane
        # line across a zebra crossing during collection. Stop the chain
        # rather than project on through it - better a shorter, correct
        # chain than a long, wrong one.
        if cand.is_junction or cand.lane_type != carla.LaneType.Driving:
            break
        if abs(cand.lane_width - base_width) > 0.5:
            break
        chain.append(cand)
        travelled += step

    lefts, rights = [], []
    for wp in chain:
        half_w = wp.lane_width / 2.0
        rv = wp.transform.get_right_vector()
        loc = wp.transform.location
        lefts.append(carla.Location(x=loc.x - rv.x * half_w,
                                    y=loc.y - rv.y * half_w, z=loc.z))
        rights.append(carla.Location(x=loc.x + rv.x * half_w,
                                     y=loc.y + rv.y * half_w, z=loc.z))
    return lefts, rights, left_label, right_label


def rasterize_mask(bev_pts_left, bev_pts_right, left_label, right_label):
    mask = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    if left_label != LABEL_ABSENT and len(bev_pts_left) >= 2:
        cv2.polylines(mask, [bev_pts_left.astype(np.int32)], False, 1,
                     thickness=LINE_THICKNESS_PX)
    if right_label != LABEL_ABSENT and len(bev_pts_right) >= 2:
        cv2.polylines(mask, [bev_pts_right.astype(np.int32)], False, 2,
                     thickness=LINE_THICKNESS_PX)
    return mask


def project_chain_to_bev(points_3d, K, w2c, M):
    """3D world points -> image pixels -> BEV pixels.

    The chain starts at the EGO'S OWN position and walks forward, so it is
    the FIRST point(s) that are most likely to fail to project - not the
    last. The camera sits 0.8 m forward of the vehicle origin, so the
    boundary point right at the car is often behind the camera or closer
    than its near-clip distance. Skip those and keep going rather than
    give up on the whole chain: everything further down the road only gets
    MORE in front of the camera, not less.
    """
    img_pts = []
    for p in points_3d:
        px = project_point(p, K, w2c)
        if px is None:
            continue
        img_pts.append(px)
    if len(img_pts) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    bev_pts = cv2.perspectiveTransform(img_pts, M)
    return bev_pts.reshape(-1, 2)


class Writer:
    def __init__(self, out_dir):
        self.images_dir = os.path.join(out_dir, "images")
        self.masks_dir = os.path.join(out_dir, "masks")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "labels.csv")
        # Not just "does the file exist" - an interrupted earlier run can
        # leave a zero-byte labels.csv behind (created, header never
        # flushed before the process died). Resuming into that would skip
        # the header and let the first data row silently masquerade as one
        # when the dataset loader reads it back.
        new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self.csv_f = open(csv_path, "a", newline="")
        self.w = csv.writer(self.csv_f)
        if new_file:
            self.w.writerow(["frame", "left_label", "right_label", "town"])
        self.n = len(os.listdir(self.images_dir))

    def add(self, bev_bgr, mask, left_label, right_label, town):
        idx = self.n
        cv2.imwrite(os.path.join(self.images_dir, f"{idx:06d}.jpg"), bev_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(self.masks_dir, f"{idx:06d}.png"), mask)
        self.w.writerow([idx, left_label, right_label, town])
        # cv2.imwrite() above is already durable on disk the instant it
        # returns; the CSV row is not - csv.writer sits on top of a
        # regular buffered file object, so without this it can sit in
        # memory for a long time and vanish if the process is ever killed
        # rather than exited cleanly through Writer.close(). That produces
        # exactly the confusing failure mode of images/masks existing on
        # disk with no matching label row, silently dropped by the dataset
        # loader later. A flush per row is cheap next to a CARLA tick.
        self.csv_f.flush()
        self.n += 1
        return idx

    def close(self):
        self.csv_f.close()


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


def collect_town(client, town, writer, args, K, M, deadline):
    world = client.get_world()

    # Skip load_world() when the town is already up. That call has crashed
    # CarlaUE4 outright on some machines, and switching straight into
    # synchronous mode after it produced a world whose camera never delivered
    # a frame. Same reasoning as carla_collect_signs.py.
    if args.load_towns and town not in world.get_map().name:
        print(f"[{town}] loading map (currently {world.get_map().name})...",
              flush=True)
        world = client.load_world(town)
        # Settle in ASYNCHRONOUS mode: wait_for_tick() rides the server's own
        # clock, so it needs nothing from us and cannot wedge anything while
        # the fresh map is still streaming in.
        for _ in range(20):
            world.wait_for_tick()
    else:
        print(f"[{town}] already loaded ({world.get_map().name}) - "
              f"skipping load_world", flush=True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # EVERYTHING below runs under this try/finally. In synchronous mode the
    # SERVER only advances when a client ticks it, so any exit that skips the
    # restore - an exception, an early return - leaves it wedged, waiting for
    # ticks nobody is sending, and every later RPC dies with
    #     RuntimeError: time-out of 60000ms while waiting for the simulator
    # looking for all the world like CARLA itself broke. The warm-up loop used
    # to sit outside the try and cost an entire debugging session that way.
    actors = []
    try:
        _drive_and_collect(world, tm, actors, town, writer, args, K, M, deadline)
    finally:
        # Async FIRST, destroy second: destroy() is a blocking RPC and hangs
        # forever in sync mode with nobody ticking, which would swallow
        # whatever exception sent us here.
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


def _drive_and_collect(world, tm, actors, town, writer, args, K, M, deadline):
    """Body of collect_town, split out so the caller's try/finally wraps it
    whole. Appends every spawned actor to `actors` - never rebinds it, or the
    caller's cleanup would have an empty list and leak the camera."""
    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)

    ego = world.spawn_actor(bl.filter("vehicle.tesla.model3")[0], spawns[0])
    actors.append(ego)

    for sp in spawns[1:1 + args.traffic]:
        try:
            npc = world.spawn_actor(random.choice(bl.filter("vehicle.*")), sp)
            npc.set_autopilot(True, tm.get_port())
            actors.append(npc)
        except Exception:
            pass

    cam, q = spawn_camera(world, ego)
    actors.append(cam)

    ego.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(ego, 100)
    # Deliberately trigger lane changes so we get plenty of frames with the
    # car close to / on top of a line - the exact condition the model needs
    # to be good at, and the one plain lane-centred driving barely produces.
    tm.random_left_lanechange_percentage(ego, 45)
    tm.random_right_lanechange_percentage(ego, 45)

    # The FIRST tick after a sensor spawns delivers no image - it only comes
    # online on the tick after registration (measured: tick 0 empty, ticks 1+
    # fine). So an empty queue is only fatal if nothing arrives at all; a bare
    # blocking q.get() here would instead hang forever with no output.
    frames_seen = 0
    for i in range(20):
        try:
            world.tick()
        except RuntimeError as e:
            raise RuntimeError(
                f"[{town}] server stopped responding to tick() on warm-up "
                f"tick {i}: {e}\nIf CarlaUE4 was launched with -dx11, drop "
                f"that flag - forcing D3D11 makes the server hang as soon as "
                f"a camera renders.") from e
        try:
            q.get(timeout=5.0)
            frames_seen += 1
        except queue.Empty:
            pass
    if frames_seen == 0:
        raise RuntimeError(
            f"[{town}] server ticked 20 times but the camera never produced "
            f"a frame.")
    print(f"[{town}] warm-up ok ({frames_seen}/20 frames)", flush=True)

    respawn_every = 25.0 / FIXED_DT   # ~25s of driving per spawn point
    tick = 0
    saved_here = 0
    skip_counts = Counter()
    last_heartbeat = time.time()
    try:
        while time.time() < deadline:
            world.tick()
            try:
                image = q.get(timeout=2.0)
            except queue.Empty:
                continue
            tick += 1

            if tick % args.every != 0:
                continue

            # A time-based heartbeat, not a saved-frame-count one: saved
            # frames can legitimately go quiet for a long stretch (a red
            # light at a junction, say), and a count-based print would go
            # just as quiet right alongside it - telling you nothing about
            # whether the script is still alive.
            if time.time() - last_heartbeat > 3.0:
                print(f"  [{town}] tick {tick}  saved {saved_here} "
                      f"({writer.n} total)  skipped={dict(skip_counts)}",
                      flush=True)
                last_heartbeat = time.time()

            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = arr.reshape((image.height, image.width, 4))[:, :, :3]

            skip_reason, mask, left_label, right_label = None, None, None, None
            wp0 = world.get_map().get_waypoint(
                ego.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving)
            # Inside a junction the ego's own lane markings are ambiguous
            # (multiple crossing/connecting lanes overlap there), so the
            # frame has no trustworthy label to begin with - skip it rather
            # than risk teaching the model a wrong one.
            if wp0 is None or wp0.is_junction:
                skip_reason = "junction"
            else:
                lefts3d, rights3d, left_label, right_label = lane_boundary_chain(wp0)
                if left_label == LABEL_ABSENT and right_label == LABEL_ABSENT:
                    skip_reason = "no lines"
                else:
                    w2c = np.array(cam.get_transform().get_inverse_matrix())
                    bev_left = project_chain_to_bev(lefts3d, K, w2c, M)
                    bev_right = project_chain_to_bev(rights3d, K, w2c, M)
                    mask = rasterize_mask(bev_left, bev_right, left_label, right_label)
                    if not mask.any():
                        skip_reason = "empty mask"

            if skip_reason is not None:
                skip_counts[skip_reason] += 1

            # The preview window is shown on EVERY processed tick, whether or
            # not the frame was saved. Tying imshow/waitKey to saved frames
            # only (the old behaviour) meant a stretch of skipped frames -
            # exactly what the junction/crosswalk fix above produces more
            # of - starved the window's message pump long enough for Windows
            # to mark it "Not Responding", even though the script was
            # working correctly underneath.
            bev_bgr = None
            if skip_reason is None:
                bev_bgr = cv2.warpPerspective(frame, M, (BEV_W, BEV_H))
                writer.add(bev_bgr, mask, left_label, right_label, town)
                saved_here += 1

            if args.preview:
                if bev_bgr is None:
                    bev_bgr = cv2.warpPerspective(frame, M, (BEV_W, BEV_H))
                vis = bev_bgr.copy()
                if skip_reason is None:
                    tint = np.zeros_like(vis)
                    tint[mask == 1] = (0, 0, 255)
                    tint[mask == 2] = (0, 220, 0)
                    cv2.addWeighted(tint, 0.5, vis, 1.0, 0, vis)
                else:
                    # A thin corner label is too easy to miss at a glance -
                    # a full border plus a filled text banner is not.
                    cv2.rectangle(vis, (0, 0), (BEV_W - 1, BEV_H - 1), (0, 165, 255), 6)
                    cv2.rectangle(vis, (0, 0), (BEV_W, 34), (0, 165, 255), -1)
                    cv2.putText(vis, f"SKIPPED: {skip_reason}", (8, 24),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow("collect preview (ESC to stop early)", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if tick % respawn_every == 0:
                ego.set_transform(random.choice(spawns))
    finally:
        # Actor cleanup and the sync-mode restore both live in collect_town's
        # finally now - this one only reports.
        print(f"[{town}] done: saved {saved_here}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="lane_data")
    ap.add_argument("--towns", default="", help="comma-separated, e.g. "
                    "Town01,Town03,Town05 - empty uses whatever is already loaded")
    ap.add_argument("--minutes", type=float, default=20.0,
                    help="wall-clock minutes of driving PER TOWN")
    ap.add_argument("--traffic", type=int, default=8)
    ap.add_argument("--every", type=int, default=4,
                    help="save every Nth simulator tick (20Hz / N)")
    ap.add_argument("--preview", action="store_true",
                    help="show a live mask-overlay window while collecting")
    args = ap.parse_args()
    args.load_towns = bool(args.towns)

    towns = [t.strip() for t in args.towns.split(",") if t.strip()] or ["current"]

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    K = build_intrinsic(IMG_W, IMG_H, FOV_DEG)
    M = build_warp_matrix(IMG_W, IMG_H)

    writer = Writer(args.out)
    print(f"writing to {args.out} (resuming from frame {writer.n})")
    try:
        for town in towns:
            print(f"[{town}] collecting for {args.minutes:.1f} min")
            deadline = time.time() + args.minutes * 60.0
            collect_town(client, town, writer, args, K, M, deadline)
    finally:
        writer.close()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"\ndone. {writer.n} labelled frames in {args.out}")


if __name__ == "__main__":
    main()
