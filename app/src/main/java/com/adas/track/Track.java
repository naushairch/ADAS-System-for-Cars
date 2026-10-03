package com.adas.track;

import android.graphics.RectF;

/**
 * A tracked object persisting across frames.
 *
 * Distance is exponentially smoothed; closing speed is estimated from the
 * smoothed series so that per-frame bounding-box jitter does not dominate.
 */
public class Track {
    private static int NEXT_ID = 1;

    public final int id;
    public int classId;
    public RectF box;
    public float score;

    public int age;            // frames since created
    public int missed;         // consecutive frames without a match
    public int hits;           // total matched frames

    public float distanceM = Float.NaN;      // smoothed distance
    public float closingSpeedMs = 0f;        // positive = getting closer
    public float ttcS = Float.POSITIVE_INFINITY;

    private float lastRawDist = Float.NaN;
    private long lastTsMs = 0L;

    private static final float DIST_ALPHA = 0.35f;   // smoothing on distance
    private static final float SPEED_ALPHA = 0.25f;  // smoothing on derivative

    public Track(RectF box, int classId, float score) {
        this.id = NEXT_ID++;
        this.box = box;
        this.classId = classId;
        this.score = score;
    }

    public void update(RectF box, int classId, float score) {
        this.box = box;
        this.classId = classId;
        this.score = score;
        this.missed = 0;
        this.hits++;
    }

    /** Feed a fresh raw distance measurement. */
    public void updateDistance(float rawDistM, long tsMs) {
        if (Float.isNaN(rawDistM)) return;

        if (Float.isNaN(distanceM)) {
            distanceM = rawDistM;
        } else {
            distanceM = DIST_ALPHA * rawDistM + (1 - DIST_ALPHA) * distanceM;
        }

        if (!Float.isNaN(lastRawDist) && lastTsMs > 0) {
            float dt = (tsMs - lastTsMs) / 1000f;
            if (dt > 0.02f) {   // ignore absurdly small dt
                float instClosing = (lastRawDist - distanceM) / dt;
                // clamp: nothing on a road closes faster than ~55 m/s relative
                instClosing = Math.max(-55f, Math.min(55f, instClosing));
                closingSpeedMs = SPEED_ALPHA * instClosing + (1 - SPEED_ALPHA) * closingSpeedMs;
                lastRawDist = distanceM;
                lastTsMs = tsMs;
            }
        } else {
            lastRawDist = distanceM;
            lastTsMs = tsMs;
        }

        ttcS = (closingSpeedMs > 0.3f)
                ? distanceM / closingSpeedMs
                : Float.POSITIVE_INFINITY;
    }

    /** A track is trustworthy only after it has been seen a few times. */
    public boolean isConfirmed() { return hits >= 3; }
}
