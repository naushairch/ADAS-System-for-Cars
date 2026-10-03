"""
lane_detect.py — finds lane lines, works out whether they are SOLID or
BROKEN using a trained SCNN (Spatial CNN, Pan et al. 2018) model, and warns
when the car crosses a solid one.

WHY A TRAINED MODEL, AND WHY SCNN SPECIFICALLY
    The previous version of this file classified solid-vs-broken from the
    fill ratio of a strip of pixels around the fitted line, one frame at a
    time. That measurement is exactly the thing a tyre destroys: the moment
    the car is close enough to the line to need the warning, the tyre and
    its shadow sit on top of the very paint being measured, and the
    single-frame read comes back "unsure" or - worse - confidently wrong.
    That produced intermittent missed warnings that no amount of tuning the
    thresholds could fully fix, because the fault was structural: local,
    single-frame evidence disappears exactly when it matters most.

    SCNN's defining idea directly targets that: a thin, long structure like
    a lane line is easy to lose locally but obvious if you can reason about
    the whole row/column it lives in. Its spatial message-passing module
    (see scnn_lane/model.py) propagates features top-to-bottom,
    bottom-to-top, left-to-right and right-to-left across the WHOLE feature
    map before anything is classified, so a few occluded rows under a tyre
    still get filled in from the clear rows above and below - in the same
    forward pass, not papered over afterwards.

WHY BIRD'S-EYE VIEW (unchanged from the classical version)
    Seen from the windscreen, lane lines converge towards a vanishing point,
    so a line's width and spacing change with distance. Warping the road
    into a top-down rectangle first makes "how far am I from the line" a
    straight pixel count, and keeps training/inference geometry identical to
    what carla_collect_lanes.py generated the training data in.

WHAT COUNTS AS A CROSSING (unchanged from the classical version)
    Not "am I near the line" but "did the line move from one side of me to
    the other". Tracking the SIGN of the offset makes this robust: drifting
    close to a line and correcting is normal driving and must stay silent.
    A rolling majority vote and a "latched" verdict (the last confident
    classification while the line was still clearly separate from the car)
    add a second layer of defence against a single bad frame - now backing
    up a model that is itself far more occlusion-robust than a fill ratio.

TRAINING
    This file loads scnn_lane/weights/scnn_lane.pt. If it is missing:
        1) python carla_collect_lanes.py --out lane_data --towns Town01,Town03,Town05 --minutes 20
        2) python train_lane_scnn.py --data lane_data --epochs 30
"""

import os

import numpy as np
import cv2

# --- bird's-eye warp -------------------------------------------------------
# Source points trace a trapezoid on the road ahead; destination is a rectangle.
# These are fractions of the frame so they survive a resolution change.
SRC_TOP_Y = 0.62
SRC_BOT_Y = 0.95
SRC_TOP_HALF = 0.10
SRC_BOT_HALF = 0.48

BEV_W, BEV_H = 400, 500

# Real-world scale of the warped image. A standard lane is ~3.5 m and we warp
# roughly one lane width plus margins into BEV_W pixels.
METRES_PER_BEV_PX = 3.5 / 260.0

# --- TOUCH detection ---------------------------------------------------------
# The warning fires the moment a TYRE is on the paint - not when the whole car
# has crossed over. That is much earlier, and it is the moment a driver would
# actually want to know.
#
# Geometry: the offset we measure is from the CAR'S CENTRELINE to the line. A
# tyre is on the paint when
#
#     |offset|  <=  half_track  +  half_line_width  +  tolerance
#
# half_track comes from the vehicle's own bounding box at runtime, so it adapts
# to whatever is spawned rather than assuming a particular car.
DEFAULT_HALF_TRACK_M = 0.78     # replaced at runtime from the CARLA bounding box
TYRE_INSET = 0.88               # tyres sit slightly inboard of the body edge
HALF_LINE_WIDTH_M = 0.075       # road paint is roughly 15 cm across

# Balanced tolerance. Larger fires earlier but risks alerting when merely close.
TOUCH_TOLERANCE_M = 0.10

# Keep warning while the car stays on the line, but not constantly.
TOUCH_REPEAT_FRAMES = 60        # 3 s at 20 fps
TOUCH_RELEASE_M = 0.18          # must move this much clear before it counts as
                                # a new touch, so jitter on the boundary cannot
                                # retrigger it

# Hard floor between any two alerts. Needed because the "nearest line" can
# switch from one side of the road to the other as the car moves across, and the
# new line reads as far away - which looks like a release and would otherwise
# re-arm the alert instantly. 0.7 s is short enough that a genuine new touch
# still feels immediate.
MIN_GAP_FRAMES = 14

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__),
                            "scnn_lane", "weights", "scnn_lane.pt")

# Segmentation classes (must match carla_collect_lanes.py / scnn_lane/model.py)
SEG_BG, SEG_LEFT, SEG_RIGHT = 0, 1, 2
# Existence classes
EXIST_ABSENT, EXIST_BROKEN, EXIST_SOLID = 0, 1, 2


def build_warp_matrix(frame_w, frame_h):
    """The perspective transform from camera frame to bird's-eye view.

    Pulled out as a free function so carla_collect_lanes.py can build the
    IDENTICAL warp when generating training data, without having to
    instantiate a LaneDetector (which loads model weights that don't exist
    yet at collection time).
    """
    src = np.float32([
        [frame_w * (0.5 - SRC_TOP_HALF), frame_h * SRC_TOP_Y],
        [frame_w * (0.5 + SRC_TOP_HALF), frame_h * SRC_TOP_Y],
        [frame_w * (0.5 + SRC_BOT_HALF), frame_h * SRC_BOT_Y],
        [frame_w * (0.5 - SRC_BOT_HALF), frame_h * SRC_BOT_Y],
    ])
    dst = np.float32([[0, 0], [BEV_W, 0], [BEV_W, BEV_H], [0, BEV_H]])
    return cv2.getPerspectiveTransform(src, dst)


class LaneDetector:

    def __init__(self, frame_w, frame_h, weights_path=None, n_windows=12):
        self.w, self.h = frame_w, frame_h
        self.M = build_warp_matrix(frame_w, frame_h)
        self.n_windows = n_windows

        weights_path = weights_path or WEIGHTS_PATH
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"no trained lane model at {weights_path}\n\n"
                "Train one first:\n"
                "  1) python carla_collect_lanes.py --out lane_data "
                "--towns Town01,Town03,Town05 --minutes 20\n"
                "  2) python train_lane_scnn.py --data lane_data --epochs 30\n\n"
                "That writes scnn_lane/weights/scnn_lane.pt, which this class "
                "loads automatically.")

        import torch
        from scnn_lane.model import SCNNLaneNet
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SCNNLaneNet().to(self.device).eval()
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        if self.device == "cuda":
            self.model.half()
        print(f"lane model loaded: {weights_path} on {self.device}")

        # Rolling memory so one bad frame cannot flip a verdict.
        self._left_solid_votes = []
        self._right_solid_votes = []
        # Crossing is tracked on the NEAREST line, whichever side it is on.
        # Watching a fixed "left slot" fails: the instant a line drifts past the
        # car it gets reclassified as the right line, so the sign never flips
        # and the crossing is invisible.
        self._near_solid_votes = []
        # Latched verdict: the last CONFIDENT reading taken while the line was
        # still at a comfortable distance. Used at crossing time so a tyre
        # sitting on the paint - which even a spatially-aware model can find
        # harder than a clear, separate line - can't silently suppress a
        # warning it already earned a frame earlier.
        self._latched_solid = None
        self._latched_age = 0
        self._half_track_m = DEFAULT_HALF_TRACK_M
        self._touching = False
        self._touch_timer = 0
        self.last_debug = None

    def set_vehicle_half_width(self, half_width_m):
        """
        Call once with the ego vehicle's bounding-box half width. Tyres sit a
        little inboard of the widest point of the body, hence TYRE_INSET.
        """
        self._half_track_m = max(0.5, half_width_m * TYRE_INSET)

    @property
    def touch_threshold_m(self):
        return self._half_track_m + HALF_LINE_WIDTH_M + TOUCH_TOLERANCE_M

    # ------------------------------------------------------------- inference

    def _infer(self, bev_bgr):
        """Runs the model on one BEV frame.

        Returns (seg_mask [BEV_H,BEV_W] uint8 of {0,1,2},
                 exist_probs [2,3] float — rows are (left, right), columns
                 are (absent, broken, solid)).
        """
        from scnn_lane.preprocess import bev_to_chw
        torch = self._torch
        chw = bev_to_chw(bev_bgr)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            seg_logits, exist_logits = self.model(tensor)
        seg_mask = seg_logits.argmax(1)[0].byte().cpu().numpy()
        exist_probs = torch.softmax(exist_logits[0].float(), dim=-1).cpu().numpy()
        return seg_mask, exist_probs

    def _line_xs(self, seg_mask, class_id):
        """
        Per-row-band x position of a line, nearest-window first (index 0 =
        bottom of the BEV image = nearest the car), matching the convention
        the touch/offset logic and the overlay both expect.

        A band with no pixels of this class carries forward the nearest
        band's x - the model's segmentation can have small gaps even on a
        line it correctly called present, and one pixel gap should not make
        the offset undefined.
        """
        win_h = BEV_H // self.n_windows
        xs = []
        last = None
        for i in range(self.n_windows):
            y_hi = BEV_H - i * win_h
            y_lo = y_hi - win_h
            cols = np.where(seg_mask[y_lo:y_hi, :] == class_id)[1]
            if cols.size > 0:
                last = float(cols.mean())
            xs.append(last)
        if all(x is None for x in xs):
            return None
        first_known = next(x for x in xs if x is not None)
        return [first_known if x is None else x for x in xs]

    # ----------------------------------------------------------------- votes

    @staticmethod
    def _vote(votes, value, keep=15):
        votes.append(value)
        if len(votes) > keep:
            votes.pop(0)
        if len(votes) < 3:
            return None
        solid = sum(1 for v in votes if v is True)
        broken = sum(1 for v in votes if v is False)
        # 2:1 majority protects against a couple of noisy frames.
        if solid >= broken * 2 and solid >= 3:
            return True
        if broken >= solid * 2 and broken >= 3:
            return False
        return None

    # ---------------------------------------------------------------- public

    def process(self, bgr):
        """
        Returns a dict:
            left_solid / right_solid : True, False, or None (unsure)
            left_offset_m / right_offset_m : signed sideways distance to each
                line. Negative = line is left of us, positive = right of us.
            crossed_solid : 'left', 'right', or None - fired ONCE at the moment
                a solid line passes underneath the car
            confidence : 0..1, the model's confidence in the nearer line's
                existence classification
        """
        bev = cv2.warpPerspective(bgr, self.M, (BEV_W, BEV_H))
        seg_mask, exist_probs = self._infer(bev)

        left_cls = int(exist_probs[0].argmax())
        right_cls = int(exist_probs[1].argmax())
        left_conf = float(exist_probs[0, left_cls])
        right_conf = float(exist_probs[1, right_cls])

        left_xs = self._line_xs(seg_mask, SEG_LEFT) if left_cls != EXIST_ABSENT else None
        right_xs = self._line_xs(seg_mask, SEG_RIGHT) if right_cls != EXIST_ABSENT else None

        left_solid_raw = None if left_cls == EXIST_ABSENT else (left_cls == EXIST_SOLID)
        right_solid_raw = None if right_cls == EXIST_ABSENT else (right_cls == EXIST_SOLID)

        left_solid = self._vote(self._left_solid_votes, left_solid_raw)
        right_solid = self._vote(self._right_solid_votes, right_solid_raw)

        car_x = BEV_W / 2.0
        left_off = (left_xs[0] - car_x) * METRES_PER_BEV_PX if left_xs else None
        right_off = (right_xs[0] - car_x) * METRES_PER_BEV_PX if right_xs else None

        # --- TOUCH detection, on whichever line is nearest --------------------
        candidates = []
        if left_xs:
            candidates.append((abs(left_off), left_off, left_solid_raw))
        if right_xs:
            candidates.append((abs(right_off), right_off, right_solid_raw))

        crossed = None
        if self._touch_timer > 0:
            self._touch_timer -= 1

        if candidates:
            _, near_off, near_solid_raw = min(candidates, key=lambda c: c[0])
            near_solid = self._vote(self._near_solid_votes, near_solid_raw)

            on_line = abs(near_off) <= self.touch_threshold_m
            clear_of_line = abs(near_off) > self.touch_threshold_m + TOUCH_RELEASE_M

            # Latch a confident verdict taken while the line is still clearly
            # separate from the car. Once a tyre is on the paint, even this
            # model's read is less reliable than it was a moment ago - so once
            # on the line we trust the latch over the live read entirely, not
            # only when the live read happens to come back "unsure".
            if not on_line and near_solid is not None:
                self._latched_solid = near_solid
                self._latched_age = 0
            else:
                self._latched_age += 1

            if on_line and self._latched_age < 90:
                effective_solid = self._latched_solid
            else:
                effective_solid = near_solid

            # ONE timer governs every alert, first contact included. Without
            # that, an offset hovering on the threshold flickers in and out of
            # contact and fires a fresh "first touch" every couple of frames.
            if on_line and effective_solid is True:
                self._touching = True
                if self._touch_timer == 0:
                    crossed = "right" if near_off > 0 else "left"
                    self._touch_timer = TOUCH_REPEAT_FRAMES
            elif clear_of_line:
                # Genuinely clear of the paint: re-arm for the next real touch,
                # but never below the hard floor.
                self._touching = False
                self._touch_timer = min(self._touch_timer, MIN_GAP_FRAMES)
        else:
            self._touching = False

        conf = max(left_conf if left_xs else 0.0, right_conf if right_xs else 0.0)
        self.last_debug = (bev, seg_mask, left_xs, right_xs)

        return dict(left_solid=left_solid, right_solid=right_solid,
                    latched_solid=self._latched_solid,
                    touching=self._touching,
                    touch_threshold_m=self.touch_threshold_m,
                    left_offset_m=left_off, right_offset_m=right_off,
                    left_fill=left_conf, right_fill=right_conf,
                    crossed_solid=crossed, confidence=conf)

    def overlay(self, frame, result):
        """Small corner panel showing the warped road and the model's mask."""
        if self.last_debug is None:
            return frame
        bev, seg_mask, lxs, rxs = self.last_debug
        vis = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        vis[seg_mask == SEG_LEFT] = (60, 60, 60)
        vis[seg_mask == SEG_RIGHT] = (60, 60, 60)

        for xs, solid in ((lxs, result["left_solid"]), (rxs, result["right_solid"])):
            if not xs:
                continue
            col = (0, 0, 255) if solid is True else (
                (0, 220, 0) if solid is False else (140, 140, 140))
            step = BEV_H // len(xs)
            for i, x in enumerate(xs):
                y = BEV_H - i * step - step // 2
                cv2.circle(vis, (int(x), int(y)), 4, col, -1)

        cv2.line(vis, (BEV_W // 2, 0), (BEV_W // 2, BEV_H), (255, 200, 0), 1)
        small = cv2.resize(vis, (160, 200))
        h, w = frame.shape[:2]
        frame[h - 210:h - 10, w - 170:w - 10] = small
        cv2.rectangle(frame, (w - 170, h - 210), (w - 10, h - 10), (200, 200, 200), 1)
        cv2.putText(frame, "lanes", (w - 166, h - 216),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return frame
