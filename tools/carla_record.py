"""
CARLA scenario recorder for the advisory ADAS project.

Produces, per scenario:
    out/<scenario>/frames/000000.jpg ...        RGB camera, matched to phone geometry
    out/<scenario>/ground_truth.csv             per-frame TRUE distance / speeds / TTC
    out/<scenario>/meta.json                    camera intrinsics + scenario params

Those frames are then replayed through the real Android app (see ReplaySource.java),
and the app's estimates are compared against ground_truth.csv by evaluate.py.

WHY THIS MATTERS: CARLA gives exact focal length, exact distance and exact TTC.
That lets you separate two error sources that are impossible to untangle on a real
road: is the DISTANCE ESTIMATOR wrong, or is the ALERT LOGIC wrong? Fix the maths
here, then take only the domain gap to the road.

--------------------------------------------------------------------------------
SETUP (Windows 11)

1. Download CARLA 0.9.15 precompiled Windows build (CARLA_0.9.15.zip) from
   https://github.com/carla-simulator/carla/releases
   Extract to e.g. C:\\CARLA_0.9.15

2. Python 3.8-3.10 (CARLA's wheel does not support 3.12):
       py -3.10 -m venv carlaenv
       carlaenv\\Scripts\\activate
       pip install carla==0.9.15 numpy opencv-python

3. Launch the simulator in low-quality mode (much lighter on an 8GB card):
       C:\\CARLA_0.9.15\\CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

4. In a second terminal, with the venv active:
       python carla_record.py --scenario all

--------------------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import queue
import shutil

import carla
import numpy as np
import cv2

# --- Camera geometry: MUST match what the phone actually produces -------------
# The Android app analyses frames at 1280x720 (ImageAnalysis targetResolution).
# Set FOV to your OnePlus 8T's actual horizontal field of view. ~78 deg is a good
# starting value for the main camera; measure yours (README step 6) and correct it.
IMG_W, IMG_H = 1280, 720
FOV_DEG = 78.0

# Dashboard mount position relative to the vehicle origin, metres.
CAM_X, CAM_Y, CAM_Z = 0.8, 0.0, 1.25
CAM_PITCH = -5.0          # slight downward tilt, as a dash mount usually sits

FIXED_DT = 0.05           # 20 Hz, matching realistic phone throughput
WARMUP_TICKS = 20


def focal_px(width, fov_deg):
    """Ground-truth focal length. Compare this with your calibrated FOCAL_PX."""
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def speed_ms(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def gap_metres(ego, other):
    """
    Bumper-to-bumper longitudinal gap, not centre-to-centre.
    Centre distance would overstate the true gap by ~4-5 m for two cars, which
    would make every TTC number wrong in the same direction and hide real bugs.
    """
    e_loc = ego.get_transform().location
    o_loc = other.get_transform().location
    centre = e_loc.distance(o_loc)
    half = ego.bounding_box.extent.x + other.bounding_box.extent.x
    return max(0.0, centre - half)


class Recorder:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.frames_dir = os.path.join(out_dir, "frames")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(self.frames_dir)
        self.rows = []

    def add(self, idx, image, t, dist, ego_v, lead_v, in_lane):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
        cv2.imwrite(os.path.join(self.frames_dir, f"{idx:06d}.jpg"), arr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        closing = ego_v - lead_v
        ttc = dist / closing if closing > 0.3 and dist > 0 else -1.0
        self.rows.append((idx, round(t, 3), round(dist, 3), round(ego_v, 3),
                          round(lead_v, 3), round(closing, 3), round(ttc, 3),
                          int(in_lane)))

    def finish(self, meta):
        with open(os.path.join(self.out_dir, "ground_truth.csv"), "w") as f:
            f.write("frame,t_s,true_dist_m,ego_ms,lead_ms,closing_ms,true_ttc_s,in_ego_lane\n")
            for r in self.rows:
                f.write(",".join(str(x) for x in r) + "\n")
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {len(self.rows)} frames written to {self.out_dir}")


def setup_world(client, sync=True):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = FIXED_DT if sync else None
    world.apply_settings(settings)
    return world


def spawn_camera(world, ego):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMG_W))
    bp.set_attribute("image_size_y", str(IMG_H))
    bp.set_attribute("fov", str(FOV_DEG))
    bp.set_attribute("sensor_tick", str(FIXED_DT))
    tf = carla.Transform(carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z),
                         carla.Rotation(pitch=CAM_PITCH))
    cam = world.spawn_actor(bp, tf, attach_to=ego)
    q = queue.Queue()
    cam.listen(q.put)
    return cam, q


def clean(actors):
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass


# ------------------------------------------------------------------ scenarios

def scenario_approach(world, rec, duration=14.0):
    """S1 - Steady approach. Lead cruises at 40 km/h, ego closes at 70 km/h.
    Expect: COLLISION_WARN then COLLISION_URGENT, once each, cleanly timed."""
    return _lead_scenario(world, rec, duration,
                          ego_target=70 / 3.6, lead_target=40 / 3.6,
                          brake_at=None, initial_gap=60.0)


def scenario_emergency_brake(world, rec, duration=12.0):
    """S2 - The classic FCW test. Lead matches speed, then brakes hard at t=5s.
    Expect: URGENT alert. This is your headline latency measurement."""
    return _lead_scenario(world, rec, duration,
                          ego_target=60 / 3.6, lead_target=60 / 3.6,
                          brake_at=5.0, initial_gap=25.0)


def scenario_roadside_parked(world, rec, duration=14.0):
    """S3 - FALSE POSITIVE TEST. Ego drives a clear lane past many parked cars.
    Expect: TOTAL SILENCE. This scenario is more important than the positive
    ones - it is where naive threshold logic embarrasses itself."""
    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    ego_bp = bl.filter("vehicle.tesla.model3")[0]
    ego = world.spawn_actor(ego_bp, spawns[0])
    actors = [ego]

    fwd = spawns[0].get_forward_vector()
    right = carla.Location(x=-fwd.y, y=fwd.x, z=0)
    base = spawns[0].location
    for i in range(10):
        d = 25 + i * 12
        side = 3.4 if i % 2 == 0 else -3.4
        loc = carla.Location(x=base.x + fwd.x * d + right.x * side,
                             y=base.y + fwd.y * d + right.y * side,
                             z=base.z + 0.3)
        bp = bl.filter("vehicle.*")[i % 8]
        try:
            a = world.spawn_actor(bp, carla.Transform(loc, spawns[0].rotation))
            actors.append(a)
        except Exception:
            pass

    cam, q = spawn_camera(world, ego)
    actors.append(cam)

    for _ in range(WARMUP_TICKS):
        world.tick(); q.get()

    idx, t = 0, 0.0
    while t < duration:
        ego.apply_control(_throttle_for(ego, 50 / 3.6))
        world.tick()
        img = q.get()
        # No lead vehicle: distance is "infinite", nothing in the ego lane.
        rec.add(idx, img, t, 999.0, speed_ms(ego), 0.0, in_lane=False)
        idx += 1; t += FIXED_DT
    return actors


def _lead_scenario(world, rec, duration, ego_target, lead_target,
                   brake_at, initial_gap):
    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    start = spawns[0]

    ego = world.spawn_actor(bl.filter("vehicle.tesla.model3")[0], start)

    fwd = start.get_forward_vector()
    lead_tf = carla.Transform(
        carla.Location(x=start.location.x + fwd.x * initial_gap,
                       y=start.location.y + fwd.y * initial_gap,
                       z=start.location.z),
        start.rotation)
    lead = world.spawn_actor(bl.filter("vehicle.audi.tt")[0], lead_tf)

    cam, q = spawn_camera(world, ego)
    actors = [ego, lead, cam]

    for _ in range(WARMUP_TICKS):
        world.tick(); q.get()

    idx, t = 0, 0.0
    while t < duration:
        ego.apply_control(_throttle_for(ego, ego_target))
        if brake_at is not None and t >= brake_at:
            lead.apply_control(carla.VehicleControl(throttle=0.0, brake=0.85))
        else:
            lead.apply_control(_throttle_for(lead, lead_target))

        world.tick()
        img = q.get()
        rec.add(idx, img, t, gap_metres(ego, lead),
                speed_ms(ego), speed_ms(lead), in_lane=True)
        idx += 1; t += FIXED_DT
    return actors


def _throttle_for(actor, target_ms):
    """Crude P controller. Good enough - we need repeatability, not realism."""
    err = target_ms - speed_ms(actor)
    if err > 0:
        return carla.VehicleControl(throttle=min(0.85, 0.25 + err * 0.12), brake=0.0)
    return carla.VehicleControl(throttle=0.0, brake=min(0.5, -err * 0.15))


SCENARIOS = {
    "s1_approach": scenario_approach,
    "s2_emergency_brake": scenario_emergency_brake,
    "s3_roadside_parked": scenario_roadside_parked,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", help="all | " + " | ".join(SCENARIOS))
    ap.add_argument("--out", default="out")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = setup_world(client, sync=True)
    original = world.get_settings()

    meta = {
        "img_w": IMG_W, "img_h": IMG_H, "fov_deg": FOV_DEG,
        "ground_truth_focal_px": round(focal_px(IMG_W, FOV_DEG), 2),
        "fixed_dt": FIXED_DT,
        "cam_pos": [CAM_X, CAM_Y, CAM_Z], "cam_pitch": CAM_PITCH,
    }
    print(f"Ground-truth focal length: {meta['ground_truth_focal_px']} px")
    print("Put this in DistanceEstimator.FOCAL_PX when replaying CARLA frames.\n")

    try:
        for name in names:
            print(f"[{name}]")
            rec = Recorder(os.path.join(args.out, name))
            actors = []
            try:
                actors = SCENARIOS[name](world, rec)
            finally:
                clean(actors)
                for _ in range(5):
                    world.tick()
            rec.finish({**meta, "scenario": name})
    finally:
        original.synchronous_mode = False
        original.fixed_delta_seconds = None
        world.apply_settings(original)
        print("\nWorld restored to async mode.")


if __name__ == "__main__":
    main()
