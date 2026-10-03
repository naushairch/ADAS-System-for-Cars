package com.adas.logic;

import android.graphics.RectF;

/**
 * Monocular distance from bounding-box WIDTH via the pinhole model:
 *
 *     distance = (realWidthMetres * focalLengthPx) / boxWidthPx
 *
 * Width is used rather than height because the bottom of a vehicle box is
 * frequently clipped by the dashboard, by road furniture, or by the vehicle in
 * front of it, whereas the left/right edges are usually clean.
 *
 * MIRRORS tools/adas_core.py :: DistanceEstimator. Keep the constants identical.
 *
 * FOCAL_PX MUST BE CALIBRATED FOR YOUR PHONE + MOUNT. Until it is, every distance
 * is plausible and wrong, which is worse than obviously wrong. See Calibration.
 */
public class DistanceEstimator {

    /**
     * Calibrate this. Placeholder value gives plausible-but-wrong distances.
     *
     * IMPORTANT: this is expressed in the pixel coordinates of the ANALYSED
     * BITMAP (e.g. 1280 wide), not the 640px model input — Detector.postprocess()
     * already unmaps the letterbox before boxes reach here.
     */
    public static float FOCAL_PX = 750f;

    // ---- corridor geometry (mirrors adas_core.py) ----
    /** Half-width of the driving corridor, metres. */
    public static final float CORRIDOR_HALF_M = 1.6f;
    /**
     * Floor on the corridor's pixel half-width. This used to be 0.06f, which at
     * 40 m produced a corridor almost 4 m wide — wide enough to include the
     * oncoming lane, and the direct cause of alerts on approaching traffic.
     */
    private static final float CORRIDOR_MIN_FRAC = 0.012f;
    private static final float CORRIDOR_MAX_FRAC = 0.40f;

    // Typical vehicle widths on Pakistani roads, metres.
    private static final float W_CAR = 1.75f;
    private static final float W_MOTORCYCLE = 0.75f;
    private static final float W_BUS = 2.50f;
    private static final float W_TRUCK = 2.45f;
    private static final float W_PERSON = 0.50f;
    private static final float W_BICYCLE = 0.60f;

    public static float realWidthFor(int classId) {
        switch (classId) {
            case 0: return W_PERSON;
            case 1: return W_BICYCLE;
            case 2: return W_CAR;
            case 3: return W_MOTORCYCLE;
            case 5: return W_BUS;
            case 7: return W_TRUCK;
            default: return W_CAR;
        }
    }

    /** Returns NaN when the box is too small for the estimate to mean anything. */
    public static float estimate(RectF box, int classId) {
        float wpx = box.width();
        if (wpx < 12f) return Float.NaN;   // beyond useful range
        return (realWidthFor(classId) * FOCAL_PX) / wpx;
    }

    public static boolean inEgoCorridor(RectF box, int frameWidth, float distM) {
        return inEgoCorridor(box, frameWidth, distM, CORRIDOR_HALF_M);
    }

    /**
     * Is this object roughly in our path?
     *
     * We take the BOTTOM-CENTRE of the box — where the wheels meet the road —
     * because that point sits on the ground plane, so its sideways position maps
     * cleanly to real metres. The top of a tall truck does not.
     *
     * The corridor narrows with distance exactly as perspective demands.
     *
     * @param frameWidth width of the analysed frame in px
     * @param distM      estimated distance
     */
    public static boolean inEgoCorridor(RectF box, int frameWidth, float distM,
                                        float corridorHalfM) {
        float cx = (box.left + box.right) * 0.5f;
        float centre = frameWidth * 0.5f;

        float halfPx = (corridorHalfM * FOCAL_PX) / Math.max(distM, 3f);
        halfPx = Math.max(frameWidth * CORRIDOR_MIN_FRAC,
                 Math.min(frameWidth * CORRIDOR_MAX_FRAC, halfPx));

        return Math.abs(cx - centre) <= halfPx;
    }

    /**
     * True if this object is travelling TOWARDS us rather than away.
     *
     * A vehicle ahead of us closes at, at most, our own speed — that is the case
     * where it is parked. Anything closing FASTER than we are moving must be
     * coming the other way. A forward-collision system should stay quiet about
     * those: they are in the opposite lane and we are not going to rear-end them.
     *
     * The margin absorbs noise in the distance estimate.
     *
     * Without this filter the system alerts on every car approaching in the
     * opposite lane, which on an undivided road makes it unusable.
     */
    public static boolean isOncoming(float closingSpeedMs, float egoSpeedMs) {
        return closingSpeedMs > (egoSpeedMs * 1.15f) + 3.5f;
    }
}
