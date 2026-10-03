package com.adas.lane;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;

/**
 * Exercises the logic that decides whether a lane warning fires.
 *
 * WHY THIS TEST EXISTS
 *   Everything else about the lane feature is checked by looking at pictures
 *   or at validation statistics. The final step - "given where the model says
 *   the lines are, do we wake the driver up?" - had no check at all, and it is
 *   the only part whose output the driver actually hears. It is also pure
 *   geometry, so it can be run on a desktop in a second rather than by driving
 *   a car at a solid white line.
 *
 *   The cases below are the four ways this is allowed to stay silent plus the
 *   one way it is allowed to speak, because for this feature a false alarm is
 *   the expensive failure: it teaches a driver to ignore alerts, and one of
 *   the alerts they would then ignore is the collision warning.
 */
public class LaneDepartureTest {

    private static final float BONNET_NONE = 1.0f;
    /** Row the fit is evaluated at when the bonnet is unset: the lowest anchor. */
    private static final float MEASURE_Y = 620f / 720f;

    /** A perfectly straight line passing through xAtMeasure at the measure row. */
    private static LaneDetector.Lane line(float xAtMeasure, float slope,
                                          int style, int nAnchors) {
        LaneDetector.Lane l = new LaneDetector.Lane();
        for (int a = 0; a < LaneDetector.NUM_ANCHORS; a++) {
            l.x[a] = Float.NaN;
        }
        // Fill from the bottom up so the anchors used are the ones nearest the car.
        int filled = 0;
        for (int a = LaneDetector.NUM_ANCHORS - 1; a >= 0 && filled < nAnchors; a--) {
            float y = LaneDetector.anchorYFraction(a);
            l.x[a] = xAtMeasure + slope * (y - MEASURE_Y);
            l.conf[a] = 0.9f;
            filled++;
        }
        l.present = filled;
        l.style = style;
        l.styleConf = 0.9f;
        return l;
    }

    private static LaneDetector.Result frame(LaneDetector.Lane left,
                                             LaneDetector.Lane right) {
        LaneDetector.Result r = new LaneDetector.Result();
        r.lanes[LaneDetector.SLOT_EGO_LEFT] = left;
        r.lanes[LaneDetector.SLOT_EGO_RIGHT] = right;
        return r;
    }

    private static LaneDetector.Lane absent() {
        LaneDetector.Lane l = new LaneDetector.Lane();
        for (int a = 0; a < LaneDetector.NUM_ANCHORS; a++) l.x[a] = Float.NaN;
        l.present = 0;
        l.style = LaneDetector.STYLE_ABSENT;
        return l;
    }

    // --- the one case that should speak ------------------------------------

    @Test
    public void firesWhenCrossingASolidLine() {
        LaneDeparture d = new LaneDeparture();

        // Centred in a lane 0.20 wide: camera at 0.50, lines at 0.40 and 0.60.
        assertNull("centred in lane must be silent",
                d.evaluate(frame(line(0.40f, 0f, LaneDetector.STYLE_SOLID, 30),
                                 line(0.60f, 0f, LaneDetector.STYLE_SOLID, 30)),
                           BONNET_NONE));

        // Drifted left until the wheel is over the line.
        assertEquals("drifting onto a solid line must warn", "left",
                d.evaluate(frame(line(0.48f, 0f, LaneDetector.STYLE_SOLID, 30),
                                 line(0.68f, 0f, LaneDetector.STYLE_SOLID, 30)),
                           BONNET_NONE));
    }

    @Test
    public void firesOnceNotEveryFrame() {
        LaneDeparture d = new LaneDeparture();
        d.evaluate(frame(line(0.40f, 0f, LaneDetector.STYLE_SOLID, 30),
                         line(0.60f, 0f, LaneDetector.STYLE_SOLID, 30)), BONNET_NONE);

        LaneDetector.Result crossing =
                frame(line(0.48f, 0f, LaneDetector.STYLE_SOLID, 30),
                      line(0.68f, 0f, LaneDetector.STYLE_SOLID, 30));

        assertEquals("left", d.evaluate(crossing, BONNET_NONE));
        // AlertType.LANE_SOLID has a cooldown, but the contract with
        // AlertEngine is one non-null on the frame the crossing BEGINS.
        assertNull("must not re-fire while still crossing",
                d.evaluate(crossing, BONNET_NONE));
        assertNull(d.evaluate(crossing, BONNET_NONE));
    }

    // --- the four ways it must stay quiet ----------------------------------

    @Test
    public void silentWhenTheLineIsDashed() {
        LaneDeparture d = new LaneDeparture();
        d.evaluate(frame(line(0.40f, 0f, LaneDetector.STYLE_DASHED, 30),
                         line(0.60f, 0f, LaneDetector.STYLE_DASHED, 30)), BONNET_NONE);
        assertNull("crossing a dashed line is a normal lane change",
                d.evaluate(frame(line(0.48f, 0f, LaneDetector.STYLE_DASHED, 30),
                                 line(0.68f, 0f, LaneDetector.STYLE_DASHED, 30)),
                           BONNET_NONE));
    }

    @Test
    public void silentWhenTooFewAnchorsWereLocated() {
        LaneDeparture d = new LaneDeparture();
        // 5 anchors is below MIN_ANCHORS_FOR_LINE (8): not a line, just specks.
        assertNull("a handful of points is not a lane boundary",
                d.evaluate(frame(line(0.48f, 0f, LaneDetector.STYLE_SOLID, 5),
                                 line(0.68f, 0f, LaneDetector.STYLE_SOLID, 5)),
                           BONNET_NONE));
    }

    @Test
    public void silentWhenPointsDoNotFormALine() {
        LaneDeparture d = new LaneDeparture();
        LaneDetector.Lane noisy = line(0.48f, 0f, LaneDetector.STYLE_SOLID, 30);
        // Scatter them well beyond MAX_FIT_RESIDUAL - wet tarmac at night
        // produces exactly this: plenty of confident points, no actual line.
        for (int a = 0; a < LaneDetector.NUM_ANCHORS; a++) {
            if (!Float.isNaN(noisy.x[a])) {
                noisy.x[a] += ((a % 2 == 0) ? 0.06f : -0.06f);
            }
        }
        assertNull("scattered points must not become a warning",
                d.evaluate(frame(noisy, line(0.68f, 0f, LaneDetector.STYLE_SOLID, 30)),
                           BONNET_NONE));
    }

    @Test
    public void silentWhenTheBoundaryIsNotSeenAtAll() {
        LaneDeparture d = new LaneDeparture();
        assertNull("no lines, no opinion",
                d.evaluate(frame(absent(), absent()), BONNET_NONE));
    }

    // --- the bonnet fix ----------------------------------------------------

    @Test
    public void bonnetMaskDoesNotBreakAGenuineCrossing() {
        // A bonnet at 80% of frame height still leaves anchors above it, so the
        // feature must keep working - just measured higher up the frame.
        LaneDeparture d = new LaneDeparture();
        float bonnet = 0.80f;
        d.evaluate(frame(line(0.40f, 0f, LaneDetector.STYLE_SOLID, 30),
                         line(0.60f, 0f, LaneDetector.STYLE_SOLID, 30)), bonnet);
        assertEquals("a real crossing must still warn with a masked bonnet",
                "left",
                d.evaluate(frame(line(0.48f, 0f, LaneDetector.STYLE_SOLID, 30),
                                 line(0.68f, 0f, LaneDetector.STYLE_SOLID, 30)),
                           bonnet));
    }
}
