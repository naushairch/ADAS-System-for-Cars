package com.adas.calib;

import com.adas.track.Track;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Measures FOCAL_PX from a target of known width at a known distance.
 *
 * THE MATHS
 *     The pinhole model the whole system rests on is
 *
 *         distance = realWidth * focal / widthPx
 *
 *     so with distance and realWidth both known by tape measure, focal falls out:
 *
 *         focal = distance * widthPx / realWidth
 *
 * WHY IT SAMPLES MANY FRAMES INSTEAD OF ONE
 *     A YOLO box jitters by several pixels frame to frame. At 20 m a car is only
 *     ~66 px wide, so three pixels of jitter is a 5% error, straight into every
 *     distance the system will ever report. Taking the MEDIAN over tens of frames
 *     removes that, and the interquartile spread reports whether the measurement
 *     can be trusted at all.
 *
 * WHY REAL WIDTH IS ENTERED RATHER THAN LOOKED UP BY CLASS
 *     DistanceEstimator assumes 1.75 m for anything car-like. Real cars vary by
 *     30 cm either way, and calibrating against an assumption bakes that error in
 *     permanently. Measure the actual target with the same tape measure.
 *
 * WHY ONLY CENTRED TARGETS COUNT
 *     The calibration target is parked straight ahead. Anything off to the side
 *     is a different object at a different distance, and averaging it in would
 *     quietly corrupt the result.
 */
public class FocalSampler {

    /** Target must sit within this fraction of the frame width from centre. */
    private static final float CENTRE_TOLERANCE = 0.20f;

    private final float knownDistanceM;
    private final float targetWidthM;
    private final int needed;
    private final List<Float> widthsPx = new ArrayList<>();

    public FocalSampler(float knownDistanceM, float targetWidthM, int needed) {
        this.knownDistanceM = knownDistanceM;
        this.targetWidthM = targetWidthM;
        this.needed = needed;
    }

    /** Feed one frame's tracks. Picks the largest confirmed, centred target. */
    public void offer(List<Track> tracks, int frameWidth) {
        if (isDone()) return;

        Track best = null;
        for (Track t : tracks) {
            if (!t.isConfirmed()) continue;
            float cx = (t.box.left + t.box.right) * 0.5f;
            if (Math.abs(cx - frameWidth * 0.5f) > frameWidth * CENTRE_TOLERANCE) {
                continue;
            }
            if (best == null || t.box.width() > best.box.width()) best = t;
        }
        if (best != null && best.box.width() > 1f) {
            widthsPx.add(best.box.width());
        }
    }

    public int count() { return widthsPx.size(); }

    public int needed() { return needed; }

    public boolean isDone() { return widthsPx.size() >= needed; }

    private float percentile(List<Float> sorted, float p) {
        if (sorted.isEmpty()) return Float.NaN;
        int idx = Math.round((sorted.size() - 1) * p);
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, idx)));
    }

    /** Median box width in pixels, or NaN with no samples. */
    public float medianWidthPx() {
        if (widthsPx.isEmpty()) return Float.NaN;
        List<Float> s = new ArrayList<>(widthsPx);
        Collections.sort(s);
        return percentile(s, 0.5f);
    }

    /** The measured focal length, in this frame's pixels. */
    public float focalPx() {
        float w = medianWidthPx();
        if (Float.isNaN(w) || targetWidthM <= 0f) return Float.NaN;
        return knownDistanceM * w / targetWidthM;
    }

    /**
     * Interquartile spread as a percentage of the median — the trust indicator.
     * Under ~5% is a solid measurement. Much above that means the detector was
     * not locked onto one steady target and the result should be discarded.
     */
    public float spreadPct() {
        if (widthsPx.size() < 4) return Float.NaN;
        List<Float> s = new ArrayList<>(widthsPx);
        Collections.sort(s);
        float med = percentile(s, 0.5f);
        if (med <= 0f) return Float.NaN;
        return 100f * (percentile(s, 0.75f) - percentile(s, 0.25f)) / med;
    }

    public String describe() {
        float spread = spreadPct();
        return String.format(Locale.US,
                "focal = %.1f px%n%nfrom %d samples, median box %.1f px%n"
                        + "target %.2f m wide at %.2f m%n"
                        + "spread %s",
                focalPx(), count(), medianWidthPx(), targetWidthM, knownDistanceM,
                Float.isNaN(spread) ? "n/a"
                        : String.format(Locale.US, "%.1f%% %s", spread,
                                spread < 5f ? "(good)"
                                        : spread < 12f ? "(usable)"
                                        : "(POOR - re-measure)"));
    }
}
