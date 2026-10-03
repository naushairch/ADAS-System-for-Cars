package com.adas.calib;

import android.content.Context;
import android.content.SharedPreferences;

import com.adas.logic.DistanceEstimator;

/**
 * Persisted camera calibration.
 *
 * WHY THIS EXISTS RATHER THAN CONSTANTS IN THE SOURCE
 *     DistanceEstimator.FOCAL_PX ships as a placeholder (750). Until it is
 *     measured for this phone in this mount, every distance and every
 *     time-to-collision is plausible and wrong — which is worse than obviously
 *     wrong, because it looks like it is working.
 *
 *     Measuring it means sitting in a car park with a tape measure. Re-flashing
 *     the APK to change one number, in a car, is a bad workflow, and the mount
 *     WILL move between sessions. So the measurement lives in SharedPreferences
 *     and is taken with the app itself.
 *
 * THE RESOLUTION TRAP
 *     FOCAL_PX is expressed in ANALYSIS-FRAME pixels. MainActivity only requests
 *     1280x720 — CameraX is free to hand back something else, and
 *     setTargetResolution is deprecated precisely because it was never a
 *     guarantee. A focal measured at one resolution is simply wrong at another.
 *
 *     Focal scales linearly with frame width, so the width it was measured at is
 *     stored alongside it and applyForFrameWidth() rescales when they differ.
 *     Without that, changing phone, camera or CameraX version would introduce a
 *     silent scale error into every distance in the system.
 *
 * HEIGHT AND PITCH are stored but not yet consumed — the depth-obstacle module
 * (Phase 4) needs them to reconstruct the road plane. They are captured now so
 * one session in the car collects everything, rather than two.
 */
public class Calibration {

    private static final String PREFS = "adas_calibration";
    private static final String K_FOCAL = "focal_px";
    private static final String K_FOCAL_W = "focal_frame_width";
    private static final String K_HEIGHT = "cam_height_m";
    private static final String K_PITCH = "cam_pitch_deg";
    private static final String K_STAMP = "calibrated_at";
    private static final String K_BONNET = "bonnet_top_frac";

    /** Dashboard mount on a mid-size SUV. A placeholder until measured. */
    public static final float DEFAULT_HEIGHT_M = 1.25f;
    public static final float DEFAULT_PITCH_DEG = -5.0f;

    /**
     * Where the focal currently in DistanceEstimator came from, worst to best.
     * Surfaced on the diagnostics line so the driver is never guessing which of
     * these is in force.
     */
    public enum Source {
        /** Nobody measured anything and the lens told us nothing. Distances are fiction. */
        PLACEHOLDER,
        /** Computed from the camera hardware. Good to a few percent, zero effort. */
        LENS,
        /** Tape-measured in a car park. The most accurate option. */
        MANUAL
    }

    private final SharedPreferences prefs;
    private Source source = Source.PLACEHOLDER;

    public Calibration(Context ctx) {
        prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public Source source() { return source; }

    /**
     * Decide which focal to use for this frame size and push it into
     * DistanceEstimator. Call once the real analysis resolution is known.
     *
     * PRECEDENCE: a manual measurement always wins. It is strictly better
     * information — it captures the actual lens at the actual mount, distortion
     * included — so a user who has done the car park work never has it silently
     * overridden by the hardware estimate.
     *
     * @param lensFocalPx focal from LensIntrinsics for this frame size, or NaN
     * @return the focal actually applied
     */
    public float resolve(int frameWidth, float lensFocalPx) {
        if (isCalibrated()) {
            source = Source.MANUAL;
            return applyForFrameWidth(frameWidth);
        }
        if (!Float.isNaN(lensFocalPx) && lensFocalPx > 0f) {
            source = Source.LENS;
            DistanceEstimator.FOCAL_PX = lensFocalPx;
            return lensFocalPx;
        }
        source = Source.PLACEHOLDER;
        DistanceEstimator.FOCAL_PX = 750f;
        return 750f;
    }

    public boolean isCalibrated() { return prefs.contains(K_FOCAL); }

    public float focalPx() { return prefs.getFloat(K_FOCAL, 750f); }

    public int focalFrameWidth() { return prefs.getInt(K_FOCAL_W, 0); }

    public float camHeightM() { return prefs.getFloat(K_HEIGHT, DEFAULT_HEIGHT_M); }

    public float camPitchDeg() { return prefs.getFloat(K_PITCH, DEFAULT_PITCH_DEG); }

    public long calibratedAt() { return prefs.getLong(K_STAMP, 0L); }

    /**
     * Where the car's own bonnet starts, as a fraction down the frame.
     * 1.0 means no bonnet is visible and nothing is ignored — the safe default,
     * because wrongly ignoring the bottom of the frame would suppress a genuinely
     * close vehicle.
     *
     * Measured on real dashcam footage: YOLO detects the ego bonnet as a car in
     * half of all frames. It is a huge box, dead centre, that never moves, so its
     * width makes it read as roughly a metre away and it occupies the ego corridor
     * permanently. depth_obstacle.py has always masked this band; the detection
     * path did not.
     */
    public float bonnetTopFrac() { return prefs.getFloat(K_BONNET, 1.0f); }

    public void saveBonnetTopFrac(float frac) {
        prefs.edit().putFloat(K_BONNET, Math.max(0.5f, Math.min(1.0f, frac))).apply();
    }

    public void saveFocal(float focalPx, int frameWidth) {
        prefs.edit()
                .putFloat(K_FOCAL, focalPx)
                .putInt(K_FOCAL_W, frameWidth)
                .putLong(K_STAMP, System.currentTimeMillis())
                .apply();
    }

    public void saveGeometry(float heightM, float pitchDeg) {
        prefs.edit()
                .putFloat(K_HEIGHT, heightM)
                .putFloat(K_PITCH, pitchDeg)
                .apply();
    }

    public void clear() { prefs.edit().clear().apply(); }

    /**
     * Push the stored focal into DistanceEstimator, rescaled if the analysis
     * resolution differs from the one it was measured at.
     *
     * @return the focal actually applied, in this frame's pixels
     */
    public float applyForFrameWidth(int frameWidth) {
        float f = focalPx();
        int measuredAt = focalFrameWidth();
        if (measuredAt > 0 && frameWidth > 0 && measuredAt != frameWidth) {
            f = f * (frameWidth / (float) measuredAt);
        }
        DistanceEstimator.FOCAL_PX = f;
        return f;
    }

    /** One-line summary for the calibration dialog. */
    public String summary(int currentFrameWidth) {
        if (!isCalibrated()) {
            if (source == Source.LENS) {
                return String.format(java.util.Locale.US,
                        "Using the camera's own lens data: focal %.1f px.%n"
                        + "Good to a few percent and needs nothing from you.%n"
                        + "Measuring below is optional and only slightly better.",
                        DistanceEstimator.FOCAL_PX);
            }
            return "NOT CALIBRATED - the lens reported nothing usable, so distances "
                    + "are placeholder values and will be wrong. Measure focal below.";
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(java.util.Locale.US,
                "focal %.1f px (measured at %d px wide)",
                focalPx(), focalFrameWidth()));
        if (focalFrameWidth() > 0 && currentFrameWidth > 0
                && focalFrameWidth() != currentFrameWidth) {
            sb.append(String.format(java.util.Locale.US,
                    "\nrescaled to %.1f px for this %d px frame",
                    applyForFrameWidth(currentFrameWidth), currentFrameWidth));
        }
        sb.append(String.format(java.util.Locale.US,
                "\nheight %.2f m, pitch %.1f deg",
                camHeightM(), camPitchDeg()));
        return sb.toString();
    }
}
