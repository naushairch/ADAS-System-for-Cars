"""
obstacle_flow.py — class-agnostic obstacle warning from image expansion.

You do not need to know WHAT something is to know you are closing on it. As you
approach any object its image EXPANDS, and the rate of that expansion gives
time-to-contact directly. No object class, no depth model, no calibration.

HOW IT MEASURES EXPANSION
    Not by tracking each point's outward speed individually - that is
    mathematically correct and practically hopeless, because one mistracked
    point produces a wild answer and no amount of median filtering rescues a set
    where half the points are rubbish.

    Instead the image is split into overlapping windows, and for each one a
    similarity transform (scale + rotation + translation) is fitted between the
    old and new point positions using RANSAC - which discards bad
    correspondences as PART of the fit rather than after it. The scale s falls
    straight out, and for an object approaching at constant speed:

        TTC = dt / (s - 1)

    A patch of road, or scenery sliding past, does not produce a clean uniform
    scale-up: RANSAC finds few inliers and the window is discarded. A real
    object approaching head-on does, because every point on it expands about the
    same centre by the same factor.

WHAT IT DELIBERATELY IGNORES
    - Vehicles YOLO already found. They have a better warning path; double
      alerting is worse than not alerting.
    - The road surface. Its texture expands purely because you are driving, so
      the search region stops well above the bonnet.
    - Anything while turning. Rotation moves every point and destroys the
      expansion signal.
"""

import numpy as np
import cv2

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

FEATURE_PARAMS = dict(maxCorners=500, qualityLevel=0.01,
                      minDistance=6, blockSize=7)

MIN_CLUSTER = 14
MAX_TTC_S = 5.0
ROTATION_LIMIT = 3.0        # px/frame of uniform sideways drift = we are turning
MIN_EGO_SPEED_MS = 5.0

MIN_SCALE_RATE = 0.004      # per-frame growth to count as "approaching"
MIN_WINDOW_POINTS = 6
MIN_INLIER_FRACTION = 0.60

SMOOTH_FRAMES = 5
TREND_EMA = 0.45


def set_sensitivity(level):
    """
    One dial for the whole detector, because tuning several constants blind is
    hopeless.

      1  very strict  - almost never speaks; use if you get false alarms
      2  strict       - the default
      3  moderate
      4  sensitive    - use if it misses real obstacles
    """
    global MIN_CLUSTER, MIN_INLIER_FRACTION, MIN_SCALE_RATE
    presets = {
        1: (22, 0.70, 0.008),
        2: (14, 0.60, 0.005),
        3: (10, 0.52, 0.0035),
        4: (7, 0.45, 0.002),
    }
    MIN_CLUSTER, MIN_INLIER_FRACTION, MIN_SCALE_RATE = presets.get(level, presets[2])


class LoomingDetector:

    def __init__(self, frame_w, frame_h, dt):
        self.w, self.h = frame_w, frame_h
        self.dt = dt
        self.prev_gray = None
        self.prev_pts = None
        self.last_points = []
        self._history = []

    def _region_mask(self, exclude_boxes):
        """
        Where we may look: a narrow trapezoid that STOPS well above the bonnet.
        The lower band is road surface, whose texture expands purely because we
        are moving, and including it guaranteed constant false alarms.
        """
        mask = np.zeros((self.h, self.w), np.uint8)
        top_y = int(self.h * 0.40)
        bot_y = int(self.h * 0.72)
        pts = np.array([[
            (int(self.w * 0.42), top_y),
            (int(self.w * 0.58), top_y),
            (int(self.w * 0.70), bot_y),
            (int(self.w * 0.30), bot_y),
        ]], np.int32)
        cv2.fillPoly(mask, pts, 255)
        for (x1, y1, x2, y2) in exclude_boxes:
            cv2.rectangle(mask, (int(x1) - 6, int(y1) - 6),
                          (int(x2) + 6, int(y2) + 6), 0, -1)
        return mask

    def process(self, bgr, exclude_boxes=(), ego_speed_ms=0.0):
        """Returns (ttc_seconds or None, n_supporting_points)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.last_points = []

        if ego_speed_ms < MIN_EGO_SPEED_MS:
            self.prev_gray = gray
            self.prev_pts = None
            self._history.clear()
            return None, 0

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=self._region_mask(exclude_boxes), **FEATURE_PARAMS)
            return None, 0

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS)
        if nxt is None:
            self.prev_gray = gray
            self.prev_pts = None
            return None, 0

        good_old = self.prev_pts[status.flatten() == 1].reshape(-1, 2)
        good_new = nxt[status.flatten() == 1].reshape(-1, 2)

        result, n = None, 0
        if len(good_old) >= MIN_CLUSTER:
            flow = good_new - good_old
            # Nearly every point sliding the same way means we are turning, not
            # approaching. The expansion signal is unusable; stay silent.
            if abs(float(np.median(flow[:, 0]))) < ROTATION_LIMIT:
                result, n = self._ttc_from_expansion(good_old, good_new,
                                                     exclude_boxes)

        self.prev_gray = gray
        if len(good_new) < 160:
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=self._region_mask(exclude_boxes), **FEATURE_PARAMS)
        else:
            self.prev_pts = good_new.reshape(-1, 1, 2)
        return result, n

    def _ttc_from_expansion(self, old, new, exclude_boxes):
        h, w = self.h, self.w
        x0, x1 = int(w * 0.30), int(w * 0.70)
        y0, y1 = int(h * 0.38), int(h * 0.74)

        best_ttc, best_inliers = None, 0

        # Several window sizes: object size varies enormously. A wall fills the
        # corridor and needs a big window; a lamp post is 50 px wide and is
        # swamped by background in anything larger than a small one.
        grids = [(3, 3, 0.55, 0.60), (5, 4, 0.28, 0.42), (7, 4, 0.18, 0.36)]
        windows = []
        for nx, ny, fw, fh in grids:
            win_w = max(40, int((x1 - x0) * fw))
            win_h = max(40, int((y1 - y0) * fh))
            step_x = max(1, (x1 - x0 - win_w) // max(1, nx - 1))
            step_y = max(1, (y1 - y0 - win_h) // max(1, ny - 1))
            for gy in range(ny):
                for gx in range(nx):
                    windows.append((x0 + gx * step_x, y0 + gy * step_y,
                                    win_w, win_h))

        for (cx0, cy0, win_w, win_h) in windows:
            cx1, cy1 = cx0 + win_w, cy0 + win_h
            sel = ((old[:, 0] >= cx0) & (old[:, 0] < cx1)
                   & (old[:, 1] >= cy0) & (old[:, 1] < cy1))
            need = max(MIN_WINDOW_POINTS, int(MIN_CLUSTER * 0.5))
            if sel.sum() < need:
                continue

            o = old[sel].astype(np.float32)
            nn = new[sel].astype(np.float32)
            M, inliers = cv2.estimateAffinePartial2D(
                o, nn, method=cv2.RANSAC, ransacReprojThreshold=1.6,
                maxIters=800, confidence=0.985)
            if M is None or inliers is None:
                continue

            n_in = int(inliers.sum())
            if n_in < need:
                continue
            # Most of the window's points must agree, otherwise the fit is
            # describing a mixture of object and background.
            if n_in / float(sel.sum()) < MIN_INLIER_FRACTION:
                continue

            scale = float(np.hypot(M[0, 0], M[1, 0]))
            if scale <= 1.0 + MIN_SCALE_RATE:
                continue
            ttc = self.dt / (scale - 1.0)
            if not (0.25 < ttc < MAX_TTC_S):
                continue

            if best_ttc is None or ttc < best_ttc:
                best_ttc = ttc
                best_inliers = n_in
                self.last_points = [(int(p[0]), int(p[1]))
                                    for p, k in zip(nn, inliers.flatten()) if k]

        if best_ttc is None:
            self._history.clear()
            return None, 0

        # SMOOTHING ONLY, not gating. Temporal confirmation belongs in one
        # place, and AlertEngine already does it with confirmation frames,
        # hysteresis and a cooldown. Requiring a falling trend here as well meant
        # two independent gates had to agree on the same frames, and any flicker
        # in either killed the alert.
        self._history.append(best_ttc)
        if len(self._history) > SMOOTH_FRAMES:
            self._history.pop(0)
        ema = self._history[0]
        for v in self._history:
            ema = TREND_EMA * v + (1 - TREND_EMA) * ema
        return float(ema), best_inliers

    def overlay(self, frame, ttc, n):
        for (x, y) in self.last_points:
            cv2.circle(frame, (x, y), 2, (0, 200, 255), -1)
        if ttc is not None:
            cv2.putText(frame, f"looming {ttc:.1f}s ({n} pts)",
                        (16, frame.shape[0] - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        return frame
