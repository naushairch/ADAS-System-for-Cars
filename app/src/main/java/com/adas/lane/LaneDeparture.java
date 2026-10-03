package com.adas.lane;

import com.adas.lane.LaneDetector.Lane;
import com.adas.lane.LaneDetector.Result;

/**
 * Turns LaneDetector output into the value AlertEngine wants for laneCross:
 * "left", "right", or null - null meaning "no opinion", which is the answer
 * this class is designed to give often.
 *
 * WHY THIS IS SEPARATE FROM LaneDetector
 *   The detector answers a perception question: where are the lines. This
 *   answers a driving question: am I leaving my lane, and is that worth
 *   telling the driver about. Those fail differently and are tuned against
 *   different evidence, so they are not one class.
 *
 * THE FOUR REASONS THIS STAYS SILENT
 *   1. The boundary was not confidently located (Lane.isLine() false).
 *   2. The located points do not lie on a line - see MAX_FIT_RESIDUAL. A
 *      wet road at night produces reflections that the network will happily
 *      report as scattered lane points; they never form a straight line.
 *   3. The boundary is DASHED. Crossing a dashed line is an ordinary lane
 *      change and warning about it is simply wrong.
 *   4. Nothing changed - the warning fires on the TRANSITION into a
 *      crossing, not every frame while it lasts.
 *
 *   On a Pakistani road with erased markings, reasons 1 and 2 are expected to
 *   fire almost continuously, and that is the intended behaviour, not a
 *   degraded one. A lane departure system that says nothing is a lane
 *   departure system that is not lying to you.
 *
 * EXTRAPOLATION, HONESTLY LABELLED
 *   BDD100K's lane labels stop around 86% of frame height (median lowest
 *   annotated point y=545 of 720), so the model was never taught what the
 *   road looks like at the very bottom of the frame - which is exactly where
 *   the car is. The fitted line is therefore extrapolated from the labelled
 *   band down to BUMPER_Y_FRAC. Extrapolation is the least trustworthy step
 *   here, which is why the fit quality gate above it is strict.
 */
public class LaneDeparture {

    /**
     * Lowest row we are willing to measure at, when nothing blocks the view.
     *
     * Was 1.0 - the very bottom of the frame. That is wrong on a bonnet mount:
     * the bottom of the frame is our own car, so the fitted line was being
     * extrapolated past the last real road pixel into bodywork, and the further
     * it is extrapolated the more a small slope error becomes a large sideways
     * one. evaluate() now measures at the bonnet line instead, which is the
     * closest road actually visible, and clamps to this when the bonnet is
     * unset (Calibration returns 1.0 until the driver measures it).
     */
    private static final float MAX_MEASURE_Y_FRAC = 0.95f;

    /** Lowest anchor row the model reports on; nothing below this is evidence. */
    private static final float LOWEST_ANCHOR_Y_FRAC = 620f / 720f;

    /**
     * Camera's horizontal position in the frame. 0.5 assumes the phone is
     * mounted on the centreline. A mount biased to one side puts a constant
     * offset on every measurement, so this is settable.
     */
    private float cameraXFrac = 0.5f;

    /**
     * A typical car is ~1.8 m in a ~3.5 m lane. Used to convert "distance
     * from the camera to the line" into "distance from the WHEEL to the
     * line", which is what actually matters for a departure.
     */
    private static final float CAR_TO_LANE_WIDTH = 0.52f;

    /** Fallback when only one boundary is visible and width cannot be measured. */
    private static final float DEFAULT_LANE_WIDTH_FRAC = 0.55f;

    /**
     * RMS distance (fraction of frame width) between the located points and
     * their best-fit line, above which the "line" is not a line.
     */
    private static final float MAX_FIT_RESIDUAL = 0.02f;

    /** Remembered so a one-frame dropout does not reset the fallback width. */
    private float lastLaneWidthFrac = DEFAULT_LANE_WIDTH_FRAC;

    private boolean crossingLeft = false;
    private boolean crossingRight = false;

    public void setCameraXFrac(float f) { cameraXFrac = f; }

    /** Straight-line fit of x against y over the located anchors. */
    private static final class Fit {
        float slope, intercept, residual;
        boolean ok;

        float xAt(float yFrac) { return slope * yFrac + intercept; }
    }

    private static Fit fit(Lane lane) {
        Fit f = new Fit();
        if (!lane.isLine()) return f;

        int n = 0;
        float sy = 0, sx = 0, syy = 0, sxy = 0;
        for (int a = 0; a < LaneDetector.NUM_ANCHORS; a++) {
            float x = lane.x[a];
            if (Float.isNaN(x)) continue;
            float y = LaneDetector.anchorYFraction(a);
            n++; sy += y; sx += x; syy += y * y; sxy += x * y;
        }
        if (n < LaneDetector.MIN_ANCHORS_FOR_LINE) return f;

        float denom = n * syy - sy * sy;
        if (Math.abs(denom) < 1e-9f) return f;
        f.slope = (n * sxy - sy * sx) / denom;
        f.intercept = (sx - f.slope * sy) / n;

        float sumSq = 0;
        for (int a = 0; a < LaneDetector.NUM_ANCHORS; a++) {
            float x = lane.x[a];
            if (Float.isNaN(x)) continue;
            float d = x - f.xAt(LaneDetector.anchorYFraction(a));
            sumSq += d * d;
        }
        f.residual = (float) Math.sqrt(sumSq / n);
        f.ok = f.residual <= MAX_FIT_RESIDUAL;
        return f;
    }

    /**
     * @return "left", "right", or null. Non-null only on the frame the
     *         crossing BEGINS, matching what AlertEngine expects.
     */
    public String evaluate(Result r, float bonnetTopFrac) {
        Lane left = r.lanes[LaneDetector.SLOT_EGO_LEFT];
        Lane right = r.lanes[LaneDetector.SLOT_EGO_RIGHT];

        Fit fl = fit(left);
        Fit fr = fit(right);

        // Measure where the road is still visible: at the bonnet line, or at
        // the lowest row the model reports on, whichever is higher up the
        // frame. Never below MAX_MEASURE_Y_FRAC.
        float measureY = Math.min(Math.min(bonnetTopFrac, MAX_MEASURE_Y_FRAC),
                                  LOWEST_ANCHOR_Y_FRAC);

        float xl = fl.ok ? fl.xAt(measureY) : Float.NaN;
        float xr = fr.ok ? fr.xAt(measureY) : Float.NaN;

        if (!Float.isNaN(xl) && !Float.isNaN(xr) && xr > xl) {
            lastLaneWidthFrac = xr - xl;
        }
        float halfCar = 0.5f * CAR_TO_LANE_WIDTH * lastLaneWidthFrac;

        // A boundary only counts if it is SOLID. Dashed is a legal crossing.
        boolean leftFires = fl.ok && left.isSolid()
                && (cameraXFrac - xl) < halfCar;
        boolean rightFires = fr.ok && right.isSolid()
                && (xr - cameraXFrac) < halfCar;

        String out = null;
        if (leftFires && !crossingLeft) out = "left";
        else if (rightFires && !crossingRight) out = "right";

        crossingLeft = leftFires;
        crossingRight = rightFires;
        return out;
    }

    /** Diagnostics for the on-screen overlay and the session log. */
    public String describe(Result r) {
        Lane l = r.lanes[LaneDetector.SLOT_EGO_LEFT];
        Lane rr = r.lanes[LaneDetector.SLOT_EGO_RIGHT];
        Fit fl = fit(l), fr = fit(rr);
        return "L " + styleName(l) + "/" + l.present + (fl.ok ? "" : "!")
                + "  R " + styleName(rr) + "/" + rr.present + (fr.ok ? "" : "!");
    }

    private static String styleName(Lane l) {
        if (!l.isLine()) return "-";
        switch (l.style) {
            case LaneDetector.STYLE_SOLID: return "solid";
            case LaneDetector.STYLE_DASHED: return "dash";
            default: return "-";
        }
    }

    public void reset() { crossingLeft = crossingRight = false; }
}
