"""
gen_parity_golden.py — freezes the Python engine's behaviour into a golden file
that the Java engine must reproduce exactly.

WHY THIS EXISTS
    adas_core.py and app/src/main/java/com/adas/logic/*.java are meant to be the
    same code in two languages. They silently drifted apart once already: the
    Python gained an oncoming-traffic filter, a corridor fix, a stopping-distance
    obstacle path and a hits gate, and none of it reached the Java. Every hour
    spent tuning thresholds in CARLA after that point was buying nothing for the
    app that actually ships.

    A comment saying "keep these in sync" did not prevent that and never will.
    This does: run the same synthetic scenarios through both engines and assert
    the alerts land on the same frames.

WHAT IS COMPARED
    Alert TYPE and FRAME INDEX only — not the reason strings, which contain
    formatted floats whose last digit can legitimately differ between Python and
    Java. The contract being enforced is "same decision, same moment".

    The scenarios drive the WHOLE chain, not just AlertEngine: raw boxes go
    through IouTracker and DistanceEstimator first, so a divergence in tracking
    or distance is caught here too.

USAGE
    python gen_parity_golden.py
        -> ../app/src/androidTest/assets/parity_golden.json

    Re-run whenever you change a threshold in adas_core.py, then run
    AlertEngineParityTest on the phone. If it fails, the Java did not get the
    change.
"""

import json
import math
import os

from adas_core import AlertEngine, DistanceEstimator, IouTracker, Track

# Pinned so both sides compute identical distances. The real value is calibrated
# per phone+mount; for parity purposes it only has to MATCH.
PARITY_FOCAL_PX = 750.0

FRAME_W, FRAME_H = 1280, 720
STEP_MS = 50           # 20 Hz
START_MS = 100_000

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "src",
                        "androidTest", "assets", "parity_golden.json")


def box_for(dist_m, class_id, cx=FRAME_W / 2.0, cy=FRAME_H * 0.55):
    """Bounding box a vehicle of this class would project to at this distance."""
    w = DistanceEstimator.real_width_for(class_id) * PARITY_FOCAL_PX / dist_m
    h = w * 0.85
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


# ---------------------------------------------------------------- scenarios

def scn_approaching_stopped_car():
    """Ego at 20 m/s closing on a stationary car. Must escalate WARN -> URGENT."""
    frames = []
    for i in range(75):
        dist = 80.0 - i * 1.0          # 20 m/s at 20 Hz = 1 m per frame
        if dist < 5:
            break
        frames.append(dict(
            dets=[dict(box=box_for(dist, 2), class_id=2, score=0.9)],
            ego_speed_ms=20.0, speed_limit_ms=27.8,
            obstacle_dist=None, lane_cross=None))
    return dict(name="approaching_stopped_car", frames=frames)


def scn_oncoming_vehicle():
    """
    A car approaching in the OPPOSITE lane, dead centre in frame.

    This is the regression test for the missing is_oncoming() filter. Without it
    the engine fires COLLISION_URGENT at a vehicle we are never going to hit.
    Expected: silence throughout.
    """
    frames = []
    for i in range(60):
        dist = 100.0 - i * 1.5         # closing at 30 m/s while we do 15
        if dist < 8:
            break
        frames.append(dict(
            dets=[dict(box=box_for(dist, 2), class_id=2, score=0.9)],
            ego_speed_ms=15.0, speed_limit_ms=27.8,
            obstacle_dist=None, lane_cross=None))
    return dict(name="oncoming_vehicle", frames=frames)


def scn_roadside_parked_car():
    """Parked car well off to the side. Outside the corridor, so silent."""
    frames = []
    for i in range(50):
        dist = 60.0 - i * 1.0
        if dist < 6:
            break
        frames.append(dict(
            dets=[dict(box=box_for(dist, 2, cx=FRAME_W / 2.0 + 260),
                       class_id=2, score=0.9)],
            ego_speed_ms=20.0, speed_limit_ms=27.8,
            obstacle_dist=None, lane_cross=None))
    return dict(name="roadside_parked_car", frames=frames)


def scn_adjacent_lane_vehicle():
    """
    Overtaking a slower car in the NEXT LANE at 3.5 m lateral offset.

    Regression test for the corridor constants. The old values (1.9 m half-width,
    6% frame-width floor) put a corridor almost 4 m wide at range, which swallowed
    the adjacent lane and fired a collision warning at a car we are simply passing.
    The oncoming filter does not save us here — the car is going our way, so only
    the corridor geometry can reject it.
    """
    frames = []
    for i in range(85):
        dist = 90.0 - i * 1.0          # closing at 20 m/s
        if dist < 8:
            break
        lateral_px = 3.5 * PARITY_FOCAL_PX / dist
        frames.append(dict(
            dets=[dict(box=box_for(dist, 2, cx=FRAME_W / 2.0 + lateral_px),
                       class_id=2, score=0.9)],
            ego_speed_ms=30.0, speed_limit_ms=33.3,
            obstacle_dist=None, lane_cross=None))
    return dict(name="adjacent_lane_vehicle", frames=frames)


def scn_speeding():
    """No traffic, 44% over the limit. Must fire SPEED_LIMIT once, then cool down."""
    frames = []
    for _ in range(90):
        frames.append(dict(dets=[], ego_speed_ms=20.0, speed_limit_ms=13.9,
                           obstacle_dist=None, lane_cross=None))
    return dict(name="speeding", frames=frames)


def scn_obstacle_approach():
    """
    Closing on a static unnamed object at 10 m/s. Exercises the stopping-distance
    trigger and the per-object latch: warn once, urgent once, then silence.
    """
    frames = []
    for i in range(80):
        dist = 40.0 - i * 0.5
        if dist < 2:
            break
        frames.append(dict(dets=[], ego_speed_ms=10.0, speed_limit_ms=13.9,
                           obstacle_dist=round(dist, 3), lane_cross=None))
    return dict(name="obstacle_approach", frames=frames)


def scn_obstacle_retreating():
    """Backing away from a wall. Must stay silent — we are escaping it."""
    frames = []
    for i in range(60):
        dist = 8.0 + i * 0.3
        frames.append(dict(dets=[], ego_speed_ms=3.0, speed_limit_ms=13.9,
                           obstacle_dist=round(dist, 3), lane_cross=None))
    return dict(name="obstacle_retreating", frames=frames)


def scn_lane_crossing():
    """A solid line crossed on one frame. Fires immediately, bypassing confirmation."""
    frames = []
    for i in range(40):
        frames.append(dict(dets=[], ego_speed_ms=15.0, speed_limit_ms=27.8,
                           obstacle_dist=None,
                           lane_cross=("left" if i == 10 else None)))
    return dict(name="lane_crossing", frames=frames)


def scn_stop_start_traffic():
    """Crawling behind a car at 2 m/s. Below MIN_SPEED_MS, so vehicle alerts stay off."""
    frames = []
    for i in range(50):
        dist = 6.0 - i * 0.02
        frames.append(dict(
            dets=[dict(box=box_for(dist, 2), class_id=2, score=0.9)],
            ego_speed_ms=2.0, speed_limit_ms=13.9,
            obstacle_dist=None, lane_cross=None))
    return dict(name="stop_start_traffic", frames=frames)


SCENARIOS = [
    scn_approaching_stopped_car,
    scn_oncoming_vehicle,
    scn_roadside_parked_car,
    scn_adjacent_lane_vehicle,
    scn_speeding,
    scn_obstacle_approach,
    scn_obstacle_retreating,
    scn_lane_crossing,
    scn_stop_start_traffic,
]


# ------------------------------------------------------------------- runner

def run_scenario(scn):
    """Drives the full chain exactly as carla_live.py does, and records alerts."""
    Track._next_id = 1
    tracker = IouTracker()
    engine = AlertEngine()

    expected = []
    sim_ms = START_MS

    for i, f in enumerate(scn["frames"]):
        dets = [(tuple(d["box"]), d["class_id"], d["score"]) for d in f["dets"]]
        tracks = tracker.update(dets)
        for t in tracks:
            if t.missed == 0:
                t.update_distance(
                    DistanceEstimator.estimate(t.box, t.class_id), sim_ms)

        alert, _subject, _reason = engine.evaluate(
            tracks, FRAME_W, f["ego_speed_ms"], f["speed_limit_ms"], None, sim_ms,
            obstacle_ttc=None, obstacle_dist=f["obstacle_dist"],
            lane_cross=f["lane_cross"])

        if alert is not None:
            expected.append(dict(frame=i, alert=alert.name))

        sim_ms += STEP_MS

    return expected


def main():
    DistanceEstimator.FOCAL_PX = PARITY_FOCAL_PX

    out = dict(
        focal_px=PARITY_FOCAL_PX,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        step_ms=STEP_MS,
        start_ms=START_MS,
        scenarios=[],
    )

    for build in SCENARIOS:
        scn = build()
        expected = run_scenario(scn)
        scn["expected"] = expected
        out["scenarios"].append(scn)

        summary = ", ".join(f"{e['alert']}@{e['frame']}" for e in expected) or "silent"
        print(f"  {scn['name']:26s} {len(scn['frames']):4d} frames -> {summary}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)

    total = sum(len(s["expected"]) for s in out["scenarios"])
    print(f"\n{len(out['scenarios'])} scenarios, {total} expected alerts")
    print(f"-> {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
