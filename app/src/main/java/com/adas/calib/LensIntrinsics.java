package com.adas.calib;

import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.util.SizeF;

/**
 * Focal length derived from the camera hardware, so no car park is required.
 *
 * WHY THIS EXISTS
 *     DistanceEstimator.FOCAL_PX shipped as 750 — a placeholder nobody measured.
 *     The only way to replace it was a tape-measure session with a second car at
 *     a known distance. That is a genuine barrier, and the consequence of
 *     skipping it is not a silent system but a confidently mistimed one: every
 *     time-to-collision threshold is derived from this number, so a focal that is
 *     25% low makes "brake" arrive a quarter of a second late.
 *
 *     But a pinhole focal in pixels is not a property of the car or the mount. It
 *     is a property of the lens and the sensor, both of which the camera stack
 *     already reports:
 *
 *         focal_px = focal_mm * frame_width_px / sensor_width_mm
 *
 *     Verified present on the target device (OnePlus 8T) via
 *     `dumpsys media.camera`: android.lens.info.availableFocalLengths and
 *     android.sensor.info.physicalSize.
 *
 * ACCURACY VERSUS A MANUAL MEASUREMENT
 *     This is typically within a few percent. It cannot capture lens distortion,
 *     and it trusts the vendor's reported sensor size, which is occasionally
 *     rounded. A tape-measure calibration remains more accurate and still wins
 *     when present — see Calibration.resolve(). This is the floor, not the
 *     ceiling: it exists so that the WORST case is "a few percent out" rather
 *     than "an unmeasured constant".
 *
 * WHAT IT DOES NOT GIVE YOU
 *     Mount height and pitch. Those are geometry of the car, not the lens, and
 *     no amount of querying the camera will reveal them. The obstacle detector
 *     (Phase 4) needs them; forward-collision does not.
 */
public final class LensIntrinsics {

    public final float focalMm;
    public final float sensorWidthMm;
    public final float sensorHeightMm;

    private LensIntrinsics(float focalMm, float sensorWidthMm, float sensorHeightMm) {
        this.focalMm = focalMm;
        this.sensorWidthMm = sensorWidthMm;
        this.sensorHeightMm = sensorHeightMm;
    }

    /**
     * Reads the first back-facing camera that reports both values.
     *
     * On a multi-camera phone this is the main sensor, because the vendor lists
     * it first — the ultrawide and telephoto are additional ids after it. If a
     * device ever ordered them differently the focal would be wrong rather than
     * absent, so the resulting number is surfaced on the diagnostics line for a
     * sanity check rather than trusted blindly.
     *
     * @return null when the device reports nothing usable; the caller then falls
     *         back to the placeholder and says so on screen.
     */
    public static LensIntrinsics readBackCamera(Context ctx) {
        try {
            CameraManager cm = (CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
            if (cm == null) return null;
            for (String id : cm.getCameraIdList()) {
                CameraCharacteristics ch = cm.getCameraCharacteristics(id);
                Integer facing = ch.get(CameraCharacteristics.LENS_FACING);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue;

                float[] focal = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                SizeF phys = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                if (focal == null || focal.length == 0 || phys == null) continue;
                if (focal[0] <= 0f || phys.getWidth() <= 0f || phys.getHeight() <= 0f) continue;

                return new LensIntrinsics(focal[0], phys.getWidth(), phys.getHeight());
            }
        } catch (Throwable ignored) {
            // Any camera-service failure is non-fatal: we fall back to the
            // placeholder. A missing focal must never stop the app starting.
        }
        return null;
    }

    /**
     * Focal length in the pixels of an analysis frame of this size.
     *
     * THE CROP CORRECTION
     *     physicalSize describes the WHOLE sensor, but a 16:9 analysis stream off
     *     a 4:3 sensor does not use the whole sensor. The stack crops to reach the
     *     requested aspect ratio, and cropping changes the field of view — so
     *     dividing by the full sensor width would understate the focal.
     *
     *     A wider-than-sensor output (16:9 from 4:3) is cropped vertically and
     *     keeps the full sensor width. A narrower one is cropped horizontally, and
     *     only that fraction of the sensor width contributes.
     */
    public float focalPxFor(int frameW, int frameH) {
        if (frameW <= 0 || frameH <= 0) return Float.NaN;
        float sensorAspect = sensorWidthMm / sensorHeightMm;
        float outAspect = frameW / (float) frameH;
        float usedWidthMm = sensorWidthMm * Math.min(1f, outAspect / sensorAspect);
        if (usedWidthMm <= 0f) return Float.NaN;
        return focalMm * frameW / usedWidthMm;
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.US,
                "lens %.2fmm, sensor %.2fx%.2fmm", focalMm, sensorWidthMm, sensorHeightMm);
    }
}
