"""
depth_obstacle.py — warns about ANY object in your path, named or not.

Trees, poles, postboxes, walls, barriers, debris. Nothing here knows what those
things are, and that is the point.

THE IDEA
    A depth model gives a distance for every pixel. Separately, simple geometry
    tells us how far the ROAD SURFACE should be at each image row - knowing the
    camera's height and pitch, a pixel halfway down the image looks at tarmac a
    predictable distance away.

    Reconstruct each pixel's real 3D position, then ask two questions: how far
    to the side is it, and how tall does it stand above the road? Anything in
    the driving corridor standing more than a third of a metre proud of the
    surface is an obstacle.

WHY RELATIVE DEPTH
    Depth models come as metric (metres directly, heavier, domain-tied) or
    relative (scale-free, far lighter, and what actually ships on mobile -
    MiDaS Small is ~16 MB as TFLite). We use relative depth and recover the
    scale ourselves each frame by fitting against road geometry we already know.

    The consequence is that the same algorithm runs unchanged whether depth
    comes from Depth Anything V2 on a desktop GPU or MiDaS Small on a phone.
    Only the model swaps out.

INSTALL
    pip install transformers
"""

import math

import numpy as np
import cv2

INFER_SIZE = 252

MIN_OBSTACLE_HEIGHT_M = 0.35
MAX_OBSTACLE_HEIGHT_M = 4.5

MIN_BLOB_PX = 90
MIN_RANGE_M = 0.8
BONNET_FRACTION = 0.10
MAX_RANGE_M = 60.0
MIN_EGO_SPEED_MS = 0.5

# Scale calibration is fitted on road pixels. When something is close enough to
# FILL that sample region we would be measuring the obstacle and calling it
# road, and the scale collapses at exactly the moment it matters most. So a
# fitted scale that jumps far from the running value is rejected as
# contaminated, and the previous one is kept.
SCALE_JUMP_LIMIT = 1.28
SCALE_MIN_SAMPLES = 200


class DepthObstacleDetector:

    def __init__(self, frame_w, frame_h, focal_px, cam_height_m=1.25,
                 cam_pitch_deg=-5.0, dt=0.05, model_name=None,
                 corridor_half_m=1.7):
        self.w, self.h = frame_w, frame_h
        self.focal = focal_px
        self.cam_h = cam_height_m
        self.pitch = math.radians(-cam_pitch_deg)
        self.dt = dt
        self.corridor_half_m = corridor_half_m

        self.ok = False
        self._model = None
        self._proc = None
        self._torch = None
        self.scale = None
        self.scale_held = False
        self.last_depth_m = None
        self.last_mask = None

        self._prev_dist = None
        self._smoothed_dist = None
        self._closing = 0.0

        self._road_depth = self._road_plane_depths()
        self._u, self._v = self._pixel_grids()
        self._load(model_name or "depth-anything/Depth-Anything-V2-Small-hf")

    # ------------------------------------------------------------------ model

    def _load(self, name):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._torch = torch
            self._proc = AutoImageProcessor.from_pretrained(name)
            self._model = AutoModelForDepthEstimation.from_pretrained(name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device).eval()
            if self.device == "cuda":
                self._model.half()
            self.ok = True
            print(f"depth model loaded: {name} on {self.device}")
        except ImportError:
            print("depth obstacles need transformers:  pip install transformers")
        except Exception as e:
            print(f"depth model not loaded ({e})")

    # --------------------------------------------------------------- geometry

    def _road_plane_depths(self):
        """
        How far away is the tarmac at each image row?

        A pixel at row y looks downward at angle (pitch + atan((y - cy) / f)).
        A camera at height h sees the ground at h / tan(angle) along that ray.
        Rows above the horizon get infinity - there is no ground there.
        """
        cy = self.h / 2.0
        rows = np.arange(self.h, dtype=np.float32)
        angle = self.pitch + np.arctan((rows - cy) / self.focal)
        depth = np.full(self.h, np.inf, dtype=np.float32)
        below = angle > 1e-3
        depth[below] = self.cam_h / np.tan(angle[below])
        depth[depth > 300.0] = np.inf
        return depth

    def _pixel_grids(self):
        """Pre-computed (u - cx) and (v - cy) so per-frame maths is cheap."""
        cx, cy = self.w / 2.0, self.h / 2.0
        u = np.arange(self.w, dtype=np.float32)[None, :] - cx
        v = np.arange(self.h, dtype=np.float32)[:, None] - cy
        return u, v

    def _reconstruct(self, depth_m):
        """
        Turn a depth map into real 3D positions, then ask of every pixel: how
        far to the side is it, and how tall is it?

        Working in metres rather than image rows means a pixel 30 m away is
        handled by the same arithmetic as one 5 m away.
        """
        u, v = self._u, self._v
        Z = depth_m
        X = u * Z / self.focal                   # right
        Y = v * Z / self.focal                   # down

        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)
        down = Y * cos_p + Z * sin_p
        forward = Z * cos_p - Y * sin_p

        height = self.cam_h - down
        return X, height, forward

    # ------------------------------------------------------------- inference

    def _predict_relative(self, bgr):
        torch = self._torch
        small = cv2.resize(bgr, (INFER_SIZE, INFER_SIZE),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inputs = self._proc(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device == "cuda":
            pixel_values = pixel_values.half()
        with torch.no_grad():
            pred = self._model(pixel_values=pixel_values).predicted_depth
        d = pred.squeeze().float().cpu().numpy()
        return cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _fit_scale(self, relative, corridor=None):
        """
        Recover metres from a scale-free prediction, using the road itself.

        Depth Anything outputs INVERSE relative depth: bigger value = nearer, so
        metres is proportional to 1 / value. We compare against the known road
        distance and take the median ratio.

        Sampled in THREE separate bands rather than one. If an obstacle is close
        enough to cover the nearest band, the further bands are still looking at
        clear road and the median across bands survives.
        """
        bands = ((0.86, 0.97), (0.78, 0.86), (0.70, 0.78))
        x0 = int(self.w * 0.36)
        x1 = int(self.w * 0.64)

        per_band = []
        for (fy0, fy1) in bands:
            y0, y1 = int(self.h * fy0), int(self.h * fy1)
            if y1 - y0 < 4:
                continue
            inv = np.maximum(relative[y0:y1, x0:x1], 1e-6)
            expected = np.repeat(self._road_depth[y0:y1, None], x1 - x0, axis=1)
            good = np.isfinite(expected) & (expected < MAX_RANGE_M)
            if good.sum() < 120:
                continue
            ratios = expected[good] * inv[good]
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            if ratios.size >= 80:
                per_band.append(float(np.median(ratios)))

        if not per_band:
            return None
        fitted = float(np.median(per_band))

        if self.scale is not None:
            ratio = fitted / self.scale
            if ratio > SCALE_JUMP_LIMIT or ratio < 1.0 / SCALE_JUMP_LIMIT:
                self.scale_held = True
                return None
        self.scale_held = False
        return fitted

    # ---------------------------------------------------------------- public

    def process(self, bgr, ego_speed_ms=0.0, exclude_boxes=()):
        """Returns (distance_m or None, ttc_s or None, n_pixels)."""
        self.last_mask = None
        if not self.ok:
            return None, None, 0
        if ego_speed_ms < MIN_EGO_SPEED_MS:
            self._prev_dist = self._smoothed_dist = None
            self._closing = 0.0
            return None, None, 0

        relative = self._predict_relative(bgr)

        scale = self._fit_scale(relative)
        if scale is None and self.scale is None:
            return None, None, 0
        if scale is not None:
            self.scale = (0.3 * scale + 0.7 * self.scale) if self.scale else scale

        depth_m = self.scale / np.maximum(relative, 1e-6)
        self.last_depth_m = depth_m

        lateral, height, forward = self._reconstruct(depth_m)

        obstacle = (
            (np.abs(lateral) <= self.corridor_half_m)
            & (height >= MIN_OBSTACLE_HEIGHT_M)
            & (height <= MAX_OBSTACLE_HEIGHT_M)
            & (forward > MIN_RANGE_M)
            & (forward < MAX_RANGE_M)
        )

        mask = obstacle.astype(np.uint8)
        # The car's own bonnet sits at the bottom of the frame and is, quite
        # correctly, "something solid in front of the camera". Ignore that band.
        mask[int(self.h * (1.0 - BONNET_FRACTION)):, :] = 0
        for (x1, y1, x2, y2) in exclude_boxes:
            mask[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        n_lab, labels, stats, _cent = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        best_dist, best_px = None, 0
        for i in range(1, n_lab):
            if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
                continue
            blob = labels == i
            # 20th percentile, not the minimum: one bad pixel should not decide
            # how far away a tree is.
            d = float(np.percentile(forward[blob], 20))
            if not np.isfinite(d):
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_px = int(stats[i, cv2.CC_STAT_AREA])
                self.last_mask = blob

        if best_dist is None:
            self._prev_dist = self._smoothed_dist = None
            self._closing = 0.0
            return None, None, 0

        if self._smoothed_dist is None:
            self._smoothed_dist = best_dist
        else:
            self._smoothed_dist = 0.4 * best_dist + 0.6 * self._smoothed_dist

        ttc = None
        if self._prev_dist is not None:
            inst = (self._prev_dist - self._smoothed_dist) / self.dt
            inst = max(-40.0, min(40.0, inst))
            self._closing = 0.3 * inst + 0.7 * self._closing
            if self._closing > 0.4:
                ttc = self._smoothed_dist / self._closing
                if not (0.2 < ttc < 12.0):
                    ttc = None
        self._prev_dist = self._smoothed_dist

        return self._smoothed_dist, ttc, best_px

    def overlay(self, frame, dist, ttc, n_px, show_depth=True):
        if self.last_mask is not None:
            tint = frame.copy()
            tint[self.last_mask] = (0, 90, 255)
            cv2.addWeighted(tint, 0.45, frame, 0.55, 0, frame)

        if show_depth and self.last_depth_m is not None:
            d = np.clip(self.last_depth_m, 2, MAX_RANGE_M)
            vis = ((1.0 - (d - 2) / (MAX_RANGE_M - 2)) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
            small = cv2.resize(vis, (200, 112))
            frame[10:122, frame.shape[1] - 210:frame.shape[1] - 10] = small
            cv2.rectangle(frame, (frame.shape[1] - 210, 10),
                          (frame.shape[1] - 10, 122), (200, 200, 200), 1)
            cv2.putText(frame, "depth", (frame.shape[1] - 206, 136),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if dist is not None:
            txt = f"obstacle {dist:.1f} m"
            if self.scale_held:
                txt += "  [scale held]"
            if ttc is not None:
                txt += f"  ttc {ttc:.1f}s"
            cv2.putText(frame, txt, (16, frame.shape[0] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        return frame
