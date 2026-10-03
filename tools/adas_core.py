"""
adas_core.py — Python mirror of the Java pipeline.

This is a DELIBERATE line-for-line port of:
    Track.java  IouTracker.java  DistanceEstimator.java  AlertEngine.java

Same class names, same constant names, same values, same logic order.

WHY: you tune thresholds here against CARLA ground truth, then copy the numbers
straight into the Java files. If you change something here, change it there too.
Any divergence between the two and your simulator results stop meaning anything
about the app you actually ship.

THIS IS NOW ENFORCED, NOT REQUESTED. The two sides silently drifted apart once
already — the Python gained the oncoming-traffic filter, the corridor fix, the
stopping-distance obstacle path and MIN_HITS_FOR_TTC, and none of it reached the
Java. Everything tuned after that point bought nothing for the shipping app.

    After changing ANY constant or branch in this file:

        python gen_parity_golden.py          # regenerates the golden trace
        gradlew connectedDebugAndroidTest    # asserts the Java still matches

    The test replays nine synthetic scenarios through both engines and compares
    alert type and frame index. Two of them exist specifically to catch the
    regressions listed above.

Requires: nothing outside the standard library.
"""

import math
from enum import Enum

# ============================================================ DistanceEstimator

class DistanceEstimator:
    """Mirror of DistanceEstimator.java"""

    # Calibrate. In CARLA use the ground-truth focal printed by carla_record.py:
    #   focal = img_width / (2 * tan(fov/2))
    FOCAL_PX = 750.0

    W_CAR = 1.75
    W_MOTORCYCLE = 0.75
    W_BUS = 2.50
    W_TRUCK = 2.45
    W_PERSON = 0.50
    W_BICYCLE = 0.60

    @classmethod
    def real_width_for(cls, class_id):
        return {0: cls.W_PERSON, 1: cls.W_BICYCLE, 2: cls.W_CAR,
                3: cls.W_MOTORCYCLE, 5: cls.W_BUS, 7: cls.W_TRUCK}.get(class_id, cls.W_CAR)

    @classmethod
    def estimate(cls, box, class_id):
        """box = (x1, y1, x2, y2). Returns metres, or nan if too small to trust."""
        wpx = box[2] - box[0]
        if wpx < 12.0:
            return float("nan")
        return (cls.real_width_for(class_id) * cls.FOCAL_PX) / wpx

    @classmethod
    def in_ego_corridor(cls, box, frame_width, dist_m, corridor_half_m=1.6):
        """
        Is this object roughly in our path?

        We take the BOTTOM-CENTRE of the box - where the wheels meet the road -
        because that point sits on the ground plane, so its sideways position
        maps cleanly to real metres. The top of a tall truck does not.

        The corridor narrows with distance exactly as perspective demands. The
        floor on half_px used to be 6% of the frame width, which at 40 m meant a
        corridor almost 4 m wide - wide enough to include the oncoming lane.
        That was the cause of alerts on approaching traffic.
        """
        cx = (box[0] + box[2]) * 0.5
        centre = frame_width * 0.5
        half_px = (corridor_half_m * cls.FOCAL_PX) / max(dist_m, 3.0)
        half_px = max(frame_width * 0.012, min(frame_width * 0.40, half_px))
        return abs(cx - centre) <= half_px

    @staticmethod
    def is_oncoming(closing_speed_ms, ego_speed_ms):
        """
        True if this object is travelling TOWARDS us rather than away.

        A vehicle ahead of us closes at, at most, our own speed - that is the
        case where it is parked. Anything closing FASTER than we are moving must
        be coming the other way. A forward collision system should stay quiet
        about those: they are in the opposite lane and we are not going to rear-
        end them.

        The margin absorbs noise in the distance estimate.
        """
        return closing_speed_ms > (ego_speed_ms * 1.15) + 3.5


# ==================================================================== Track

class Track:
    """Mirror of Track.java"""

    _next_id = 1
    DIST_ALPHA = 0.35
    SPEED_ALPHA = 0.25

    def __init__(self, box, class_id, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.class_id = class_id
        self.score = score
        self.age = 0
        self.missed = 0
        self.hits = 0
        self.distance_m = float("nan")
        self.closing_speed_ms = 0.0
        self.ttc_s = float("inf")
        self._last_raw = float("nan")
        self._last_ts = 0.0

    def update(self, box, class_id, score):
        self.box = box
        self.class_id = class_id
        self.score = score
        self.missed = 0
        self.hits += 1

    def update_distance(self, raw_dist_m, ts_ms):
        if math.isnan(raw_dist_m):
            return

        if math.isnan(self.distance_m):
            self.distance_m = raw_dist_m
        else:
            self.distance_m = (self.DIST_ALPHA * raw_dist_m
                               + (1 - self.DIST_ALPHA) * self.distance_m)

        if not math.isnan(self._last_raw) and self._last_ts > 0:
            dt = (ts_ms - self._last_ts) / 1000.0
            if dt > 0.02:
                inst = (self._last_raw - self.distance_m) / dt
                inst = max(-55.0, min(55.0, inst))
                self.closing_speed_ms = (self.SPEED_ALPHA * inst
                                         + (1 - self.SPEED_ALPHA) * self.closing_speed_ms)
                self._last_raw = self.distance_m
                self._last_ts = ts_ms
        else:
            self._last_raw = self.distance_m
            self._last_ts = ts_ms

        self.ttc_s = (self.distance_m / self.closing_speed_ms
                      if self.closing_speed_ms > 0.3 else float("inf"))

    @property
    def is_confirmed(self):
        return self.hits >= 3


# =============================================================== IouTracker

def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if uni <= 0 else inter / uni


def _is_vehicle(c):
    return c in (2, 3, 5, 7)


class IouTracker:
    """Mirror of IouTracker.java"""

    MATCH_IOU = 0.30
    MAX_MISSED = 6

    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """detections: list of (box, class_id, score)"""
        used = [False] * len(detections)

        for t in sorted(self.tracks, key=lambda x: -x.hits):
            best_i, best_iou = -1, self.MATCH_IOU
            for i, (box, cid, sc) in enumerate(detections):
                if used[i]:
                    continue
                if not (t.class_id == cid or (_is_vehicle(t.class_id) and _is_vehicle(cid))):
                    continue
                v = _iou(t.box, box)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i >= 0:
                box, cid, sc = detections[best_i]
                t.update(box, cid, sc)
                used[best_i] = True
            else:
                t.missed += 1
            t.age += 1

        for i, (box, cid, sc) in enumerate(detections):
            if not used[i]:
                self.tracks.append(Track(box, cid, sc))

        self.tracks = [t for t in self.tracks if t.missed <= self.MAX_MISSED]
        return self.tracks

    def reset(self):
        self.tracks = []


# ================================================================= AlertType

class AlertType(Enum):
    COLLISION_URGENT = (0, "brake", 5000)
    OBSTACLE_AHEAD = (1, "obstacle", 4500)
    COLLISION_WARN = (2, "vehicle_close", 3500)
    LANE_SOLID = (3, "lane", 5000)
    SPEED_LIMIT = (4, "slow_down", 12000)

    def __init__(self, priority, audio_key, cooldown_ms):
        self.priority = priority
        self.audio_key = audio_key
        self.cooldown_ms = cooldown_ms


# =============================================================== AlertEngine

class AlertEngine:
    """
    Mirror of AlertEngine.java.

    Four guards against an unusable, chattering system:
      1. temporal confirmation  2. hysteresis  3. cooldown  4. priority arbitration
    Plus context gating below MIN_SPEED_MS.
    """

    # ---- tunables: change here, then copy into AlertEngine.java ----
    TTC_URGENT_SET = 1.6
    TTC_URGENT_CLEAR = 2.2
    TTC_WARN_SET = 2.8
    TTC_WARN_CLEAR = 3.6

    SPEED_OVER_SET = 1.10
    SPEED_OVER_CLEAR = 1.03

    MIN_SPEED_MS = 4.2            # ~15 km/h, for VEHICLE collision alerts

    # Obstacles get a far lower gate. The 15 km/h floor exists to stop vehicle
    # warnings firing continuously in stop-start traffic, where you are always
    # a few metres from the car in front. A tree does not behave that way: if
    # you are closing on one at walking pace you still want to be told, and
    # the corridor test already excludes anything at the roadside.
    OBSTACLE_MIN_SPEED_MS = 0.9   # ~3 km/h

    # --- obstacle trigger: STOPPING DISTANCE, not time-to-contact -----------
    #
    # Time-to-contact for an obstacle was computed by differentiating the depth
    # measurement. Depth is noisy, and the derivative of a noisy signal is far
    # noisier still, so TTC flickered across the threshold and the warning
    # re-fired over and over. That was the real fault, not the threshold value.
    #
    # An obstacle is STATIC. A tree is not going anywhere, so the closing speed
    # simply IS our own speed - a number we measure directly and accurately.
    # No differentiation, no noise.
    #
    # The distance to warn at is then the distance we still need in order to
    # stop: reaction time plus braking. That scales properly with speed, which
    # a fixed threshold cannot - 10 m is half a second of warning at 80 km/h
    # and a nagging 3.6 seconds at walking pace.
    #
    #     warn_distance = v * t_reaction  +  v^2 / (2 * decel)
    REACT_WARN_S = 1.2
    DECEL_WARN_MS2 = 4.5          # comfortable braking
    REACT_URGENT_S = 0.6
    DECEL_URGENT_MS2 = 7.0        # hard braking

    # Floors, so the speed formula never delays a warning past a sensible
    # distance. At 5 km/h the formula alone asks for barely 4 m, which means an
    # object six metres ahead is silently ignored while you roll towards it.
    # Twelve metres is roughly where a driver wants to be told regardless of
    # how slowly they are going.
    WARN_FLOOR_M = 12.0
    URGENT_FLOOR_M = 6.0

    OBSTACLE_CONFIRM = 3          # frames below threshold before speaking
    OBSTACLE_ABSENT_FRAMES = 12   # frames with no obstacle before the latch clears
    OBSTACLE_RELEASE_M = 6.0      # backing off this far re-arms the same object
    # Approach is judged over a WINDOW, not frame to frame. Depth jitters by
    # about a metre, so a single-frame comparison decides you are retreating
    # several times a second and resets the confirmation counters each time -
    # which is why a genuine approach never accumulated enough frames to fire.
    OBSTACLE_APPROACH_M = 0.8     # net closing over the window
    OBSTACLE_APPROACH_WINDOW = 15 # frames (~0.75 s)
    CONFIRM_FRAMES = 3
    SPEED_CONFIRM_FRAMES = 25
    MIN_HITS_FOR_TTC = 6        # frames of history before we trust closing speed

    # Unnamed obstacles - trees, poles, postboxes, walls - go through the SAME
    # collision thresholds as vehicles. An object you are about to hit is an
    # object you are about to hit; it does not matter whether the detector has
    # a word for it. Giving obstacles their own, later thresholds meant the
    # warning only arrived when the car was nearly touching them.
    OBSTACLE_CONFIRM_FRAMES = 5

    def __init__(self):
        self._consec = {t: 0 for t in AlertType}
        self._latched = {t: False for t in AlertType}
        self._last_fired = {t: 0 for t in AlertType}
        self._suppress_until = 0
        self._last_was_obstacle = False
        # Per-obstacle latch. Stage 0 = nothing said, 1 = warned, 2 = urgent.
        # We only ever escalate; the same tree is never announced twice.
        self._obs_stage = 0
        self._obs_absent = 0
        self._obs_min_dist = float("inf")
        self._obs_warn_count = 0
        self._obs_urgent_count = 0
        self._obs_smooth = None
        self._obs_history = []
        # Populated every frame so the caller can show exactly why the obstacle
        # path did or did not speak. Silence has many possible causes, and
        # guessing between them wastes far more time than reporting them.
        self.obs_debug = {}
        self._obs_smooth = None
        self._obs_history = []
        # Populated every frame so the caller can show exactly why the obstacle
        # path did or did not speak. Silence has many possible causes and
        # guessing between them wastes far more time than reporting them.
        self.obs_debug = {}

    def evaluate(self, tracks, frame_width, ego_speed_ms,
                 speed_limit_ms, lane_offset, now_ms,
                 obstacle_ttc=None, lane_cross=None, obstacle_dist=None):
        """
        Returns (AlertType | None, subject_track | None, reason_str).

        obstacle_ttc : seconds to contact with an UNNAMED object, from optical
                       flow, or None when nothing credible is looming.
        lane_cross   : 'left' / 'right' on the single frame a solid line is
                       crossed, otherwise None.
        """

        if now_ms < self._suppress_until:
            for t in AlertType:
                self._consec[t] = 0
            return None, None, "suppressed"

        worst, worst_ttc = None, float("inf")
        moving = ego_speed_ms >= self.MIN_SPEED_MS
        obstacle_ok = ego_speed_ms >= self.OBSTACLE_MIN_SPEED_MS

        if moving:
            for t in tracks:
                if not t.is_confirmed or math.isnan(t.distance_m):
                    continue
                if not DistanceEstimator.in_ego_corridor(t.box, frame_width, t.distance_m):
                    continue
                # Ignore traffic coming the other way. We will not rear-end a
                # vehicle that is in the opposite lane heading towards us.
                if DistanceEstimator.is_oncoming(t.closing_speed_ms, ego_speed_ms):
                    continue
                # A closing-speed estimate needs a few frames of history before
                # it means anything. Without this, a car that has just appeared
                # produces a wild TTC on its second frame.
                if t.hits < self.MIN_HITS_FOR_TTC:
                    continue
                if t.ttc_s < worst_ttc:
                    worst_ttc, worst = t.ttc_s, t

        # Fold the unnamed obstacle into the SAME comparison as vehicles, BEFORE
        # the thresholds are applied. Whichever threat is closest in time wins,
        # whatever it happens to be. A tree you are about to hit is a tree you
        # are about to hit; it does not deserve its own, later warning.
        self._reset(AlertType.OBSTACLE_AHEAD)
        obstacle = False
        obstacle_is_worst = False
        # Obstacles are judged on stopping distance and latched per object.
        obs_alert = self._obstacle_alert(obstacle_dist, ego_speed_ms)
        self._last_was_obstacle = obs_alert is not None
        self._obs_reason = ""
        if obs_alert is not None:
            self._obs_reason = (
                f"obstacle at {obstacle_dist:.1f} m, "
                f"stopping needs {self.warn_distance(ego_speed_ms):.1f} m "
                f"at {ego_speed_ms * 3.6:.0f} km/h")

        urgent = moving and self._hyst(
            AlertType.COLLISION_URGENT, worst_ttc,
            self.TTC_URGENT_SET, self.TTC_URGENT_CLEAR)
        warn = moving and self._hyst(
            AlertType.COLLISION_WARN, worst_ttc,
            self.TTC_WARN_SET, self.TTC_WARN_CLEAR)

        # Crossing a solid line is a single EVENT, not a sustained condition, so
        # it bypasses the frame-confirmation counter - by the time you confirmed
        # it over five frames the wheels are already in the next lane. The
        # detector does its own multi-frame voting before reporting a crossing.
        lane = bool(lane_cross) and moving
        if lane:
            self._consec[AlertType.LANE_SOLID] = self.CONFIRM_FRAMES
        else:
            self._reset(AlertType.LANE_SOLID)

        if speed_limit_ms > 0 and ego_speed_ms > 0:
            speeding = self._hyst_above(AlertType.SPEED_LIMIT,
                                        ego_speed_ms / speed_limit_ms,
                                        self.SPEED_OVER_SET, self.SPEED_OVER_CLEAR)
        else:
            self._reset(AlertType.SPEED_LIMIT)
            speeding = False

        self._tick(AlertType.COLLISION_URGENT, urgent)
        self._tick(AlertType.COLLISION_WARN, warn and not urgent)

        self._tick(AlertType.SPEED_LIMIT, speeding)

        # An obstacle alert bypasses the frame counters and cooldowns. The latch
        # has already guaranteed it speaks at most once per object, so a
        # cooldown on top could only ever silence a genuine SECOND obstacle.
        # A vehicle URGENT still outranks an obstacle WARN.
        if obs_alert is AlertType.COLLISION_URGENT:
            self._last_fired[obs_alert] = now_ms
            return obs_alert, None, self._obs_reason
        if obs_alert is AlertType.COLLISION_WARN:
            veh_urgent_ready = (
                self._consec[AlertType.COLLISION_URGENT] >= self.CONFIRM_FRAMES
                and (now_ms - self._last_fired[AlertType.COLLISION_URGENT])
                >= AlertType.COLLISION_URGENT.cooldown_ms)
            if not veh_urgent_ready:
                self._last_fired[obs_alert] = now_ms
                return obs_alert, None, self._obs_reason

        for t in sorted(AlertType, key=lambda x: x.priority):
            if t is AlertType.SPEED_LIMIT:
                need = self.SPEED_CONFIRM_FRAMES
            elif t is AlertType.OBSTACLE_AHEAD:
                need = self.OBSTACLE_CONFIRM_FRAMES
            else:
                need = self.CONFIRM_FRAMES
            if self._consec[t] >= need and (now_ms - self._last_fired[t]) >= t.cooldown_ms:
                self._last_fired[t] = now_ms
                self._consec[t] = 0
                return t, worst, self._describe(t, worst, worst_ttc, ego_speed_ms,
                                                speed_limit_ms, lane_cross,
                                                obstacle_ttc)
        return None, None, "silent"

    @classmethod
    def warn_distance(cls, speed_ms):
        """Distance needed to stop comfortably, plus reaction time."""
        d = speed_ms * cls.REACT_WARN_S + (speed_ms ** 2) / (2 * cls.DECEL_WARN_MS2)
        return max(cls.WARN_FLOOR_M, d)

    @classmethod
    def urgent_distance(cls, speed_ms):
        """Distance needed to stop under hard braking. Below this you are late."""
        d = (speed_ms * cls.REACT_URGENT_S
             + (speed_ms ** 2) / (2 * cls.DECEL_URGENT_MS2))
        return max(cls.URGENT_FLOOR_M, d)

    def _obstacle_alert(self, obstacle_dist, ego_speed_ms):
        """
        Decide what to say about an unnamed obstacle, and say it ONCE.

        The latch is the important part. A time-based cooldown re-fires while
        you are still approaching the same object, which is what made the old
        behaviour so noisy. Instead we remember what we have already said about
        THIS obstacle and only ever escalate - warn, then urgent, then nothing
        more until the object is gone or we have backed away from it.
        """
        self.obs_debug = dict(dist=obstacle_dist, speed=ego_speed_ms,
                              stage=self._obs_stage, reason="")

        if obstacle_dist is None:
            self.obs_debug["reason"] = "no obstacle detected"
            self._obs_absent += 1
            if self._obs_absent >= self.OBSTACLE_ABSENT_FRAMES:
                self._obs_stage = 0
                self._obs_min_dist = float("inf")
                self._obs_warn_count = self._obs_urgent_count = 0
                self._obs_smooth = None
                self._obs_history = []
            return None
        self._obs_absent = 0

        # Moving well clear of it means the next approach is a fresh event -
        # whether it is a different object or the same one on a second pass.
        if obstacle_dist > self._obs_min_dist + self.OBSTACLE_RELEASE_M:
            self._obs_stage = 0
            self._obs_min_dist = obstacle_dist
            self._obs_warn_count = self._obs_urgent_count = 0
            self._obs_history = []
        self._obs_min_dist = min(self._obs_min_dist, obstacle_dist)

        if ego_speed_ms < self.OBSTACLE_MIN_SPEED_MS:
            self.obs_debug["reason"] = (
                f"too slow ({ego_speed_ms * 3.6:.1f} < "
                f"{self.OBSTACLE_MIN_SPEED_MS * 3.6:.1f} km/h)")
            return None

        # Only warn about something we are actually CLOSING ON. Without this,
        # reversing away from a wall re-arms the latch and then immediately
        # fires again as the distance passes back through the threshold - a
        # warning about an object we are escaping.
        #
        # Judged on a smoothed distance, because raw depth jitters by tens of
        # centimetres and a single-frame comparison would call that a retreat.
        self._obs_smooth = (obstacle_dist if self._obs_smooth is None
                            else 0.4 * obstacle_dist + 0.6 * self._obs_smooth)
        self._obs_history.append(self._obs_smooth)
        if len(self._obs_history) > self.OBSTACLE_APPROACH_WINDOW:
            self._obs_history.pop(0)

        approaching = (
            len(self._obs_history) >= 3
            and self._obs_smooth
            < self._obs_history[0] - self.OBSTACLE_APPROACH_M)
        self.obs_debug["approaching"] = approaching
        self.obs_debug["closed"] = (
            round(self._obs_history[0] - self._obs_smooth, 2)
            if self._obs_history else 0.0)
        if not approaching:
            self.obs_debug["reason"] = ("not closing (distance steady or "
                                        "increasing)")
            self._obs_warn_count = self._obs_urgent_count = 0
            return None

        d_warn = self.warn_distance(ego_speed_ms)
        d_urgent = self.urgent_distance(ego_speed_ms)

        # A short confirmation still applies, so one bad depth frame cannot
        # trigger anything on its own.
        self._obs_urgent_count = (self._obs_urgent_count + 1
                                  if obstacle_dist <= d_urgent else 0)
        self._obs_warn_count = (self._obs_warn_count + 1
                                if obstacle_dist <= d_warn else 0)

        self.obs_debug.update(warn_d=d_warn, urgent_d=d_urgent,
                              warn_count=self._obs_warn_count,
                              urgent_count=self._obs_urgent_count)
        if obstacle_dist > d_warn:
            self.obs_debug["reason"] = (
                f"still {obstacle_dist - d_warn:.1f} m beyond the warn "
                f"distance ({d_warn:.1f} m at {ego_speed_ms * 3.6:.0f} km/h)")
        elif self._obs_stage >= 1 and obstacle_dist > d_urgent:
            self.obs_debug["reason"] = "already warned about this object"
        elif self._obs_stage >= 2:
            self.obs_debug["reason"] = "already at urgent for this object"

        if self._obs_stage < 2 and self._obs_urgent_count >= self.OBSTACLE_CONFIRM:
            self._obs_stage = 2
            return AlertType.COLLISION_URGENT
        if self._obs_stage < 1 and self._obs_warn_count >= self.OBSTACLE_CONFIRM:
            self._obs_stage = 1
            return AlertType.COLLISION_WARN
        return None

    def suppress_for(self, ms, now_ms):
        self._suppress_until = now_ms + ms

    # ---- internals ----
    def _hyst(self, t, value, set_v, clear_v):
        """True while value is BELOW set; releases only above clear."""
        was = self._latched[t]
        now = (value < clear_v) if was else (value < set_v)
        self._latched[t] = now
        return now

    def _hyst_above(self, t, value, set_v, clear_v):
        was = self._latched[t]
        now = (value > clear_v) if was else (value > set_v)
        self._latched[t] = now
        return now

    def _tick(self, t, active):
        self._consec[t] = self._consec[t] + 1 if active else 0

    def _reset(self, t):
        self._consec[t] = 0
        self._latched[t] = False

    @staticmethod
    def _describe(t, s, ttc, ego, limit, lane_cross, obstacle_ttc):
        if t in (AlertType.COLLISION_URGENT, AlertType.COLLISION_WARN):
            if s is None:
                return f"ttc={ttc:.2f} OBSTACLE (unnamed object in path)"
            return (f"ttc={ttc:.2f} d={s.distance_m:.1f} "
                    f"close={s.closing_speed_ms:.1f} id={s.id}")
        if t is AlertType.OBSTACLE_AHEAD:
            return f"looming_ttc={obstacle_ttc:.2f}"
        if t is AlertType.LANE_SOLID:
            return f"crossed solid line on {lane_cross}"
        if t is AlertType.SPEED_LIMIT:
            return f"ego={ego:.1f} limit={limit:.1f}"
        return ""


def ground_truth_focal(width_px, fov_deg):
    """Exact focal length for a simulated camera. No calibration needed in CARLA."""
    return width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
