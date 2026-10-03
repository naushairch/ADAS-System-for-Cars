package com.adas.logic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.graphics.RectF;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.adas.detect.Detection;
import com.adas.track.IouTracker;
import com.adas.track.Track;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Enforces that this Java pipeline behaves identically to tools/adas_core.py.
 *
 * WHY
 *     The two implementations drifted apart once already, and the drift was
 *     invisible: every threshold tuned in CARLA after that point bought nothing
 *     for the shipping app. A comment saying "keep these in sync" did not stop
 *     it. This does.
 *
 * WHAT IS ASSERTED
 *     Alert TYPE and FRAME INDEX, for the whole chain — IouTracker, then
 *     DistanceEstimator, then AlertEngine — replaying the exact synthetic
 *     scenarios that tools/gen_parity_golden.py ran through the Python.
 *
 *     Reason strings are deliberately NOT compared: they contain formatted
 *     floats whose last digit can legitimately differ between the two languages.
 *     The contract is "same decision, same moment".
 *
 * HOW TO RUN
 *     Needs a connected device or emulator (RectF is a real framework class):
 *         gradlew connectedDebugAndroidTest
 *     or right-click the class in Android Studio -> Run.
 *
 * WHEN IT FAILS
 *     A threshold changed on one side only. Regenerate and re-check:
 *         python tools/gen_parity_golden.py
 *     then port whatever moved in adas_core.py into logic/*.java.
 *
 * Two of these scenarios are regression tests for bugs found in the Java:
 *   - oncoming_vehicle      : without DistanceEstimator.isOncoming() this fired
 *                             COLLISION_WARN then COLLISION_URGENT at a car in
 *                             the opposite lane.
 *   - adjacent_lane_vehicle : with the old corridor constants (1.9 m half-width,
 *                             6% frame-width floor) this fired COLLISION_WARN at
 *                             a car being overtaken in the next lane.
 */
@RunWith(AndroidJUnit4.class)
public class AlertEngineParityTest {

    private static class Fired {
        final int frame;
        final String alert;

        Fired(int frame, String alert) {
            this.frame = frame;
            this.alert = alert;
        }

        @Override
        public String toString() {
            return alert + "@" + frame;
        }
    }

    @Test
    public void javaEngineMatchesPythonGolden() throws Exception {
        JSONObject golden = new JSONObject(readAsset("parity_golden.json"));

        final float focalPx = (float) golden.getDouble("focal_px");
        final int frameWidth = golden.getInt("frame_width");
        final long stepMs = golden.getLong("step_ms");
        final long startMs = golden.getLong("start_ms");

        // Pinned so both sides compute identical distances.
        DistanceEstimator.FOCAL_PX = focalPx;

        JSONArray scenarios = golden.getJSONArray("scenarios");
        assertTrue("golden file has no scenarios", scenarios.length() > 0);

        List<String> failures = new ArrayList<>();

        for (int s = 0; s < scenarios.length(); s++) {
            JSONObject scn = scenarios.getJSONObject(s);
            String name = scn.getString("name");

            List<Fired> actual = replay(scn, frameWidth, startMs, stepMs);
            List<Fired> expected = parseExpected(scn.getJSONArray("expected"));

            if (!sameSequence(expected, actual)) {
                failures.add(String.format(
                        "%n  %s%n    python expected : %s%n    java produced   : %s",
                        name, expected, actual));
            }
        }

        assertEquals("Java engine diverged from the Python golden trace:"
                + String.join("", failures) + "\n", 0, failures.size());
    }

    /** Drives the full chain exactly as MainActivity.analyze() does. */
    private List<Fired> replay(JSONObject scn, int frameWidth, long startMs, long stepMs)
            throws Exception {

        IouTracker tracker = new IouTracker();
        AlertEngine engine = new AlertEngine();
        List<Fired> fired = new ArrayList<>();

        JSONArray frames = scn.getJSONArray("frames");
        long simMs = startMs;

        for (int i = 0; i < frames.length(); i++) {
            JSONObject f = frames.getJSONObject(i);

            JSONArray detsJson = f.getJSONArray("dets");
            List<Detection> dets = new ArrayList<>(detsJson.length());
            for (int k = 0; k < detsJson.length(); k++) {
                JSONObject d = detsJson.getJSONObject(k);
                JSONArray b = d.getJSONArray("box");
                RectF box = new RectF(
                        (float) b.getDouble(0), (float) b.getDouble(1),
                        (float) b.getDouble(2), (float) b.getDouble(3));
                dets.add(new Detection(box, d.getInt("class_id"),
                        (float) d.getDouble("score")));
            }

            List<Track> tracks = tracker.update(dets);
            for (Track t : tracks) {
                if (t.missed == 0) {
                    t.updateDistance(DistanceEstimator.estimate(t.box, t.classId), simMs);
                }
            }

            float obstacleDist = f.isNull("obstacle_dist")
                    ? Float.NaN : (float) f.getDouble("obstacle_dist");
            String laneCross = f.isNull("lane_cross") ? null : f.getString("lane_cross");

            AlertEngine.Decision d = engine.evaluate(
                    tracks, frameWidth,
                    (float) f.getDouble("ego_speed_ms"),
                    (float) f.getDouble("speed_limit_ms"),
                    simMs, obstacleDist, laneCross);

            if (d.fire != null) fired.add(new Fired(i, d.fire.name()));

            simMs += stepMs;
        }
        return fired;
    }

    private static List<Fired> parseExpected(JSONArray arr) throws Exception {
        List<Fired> out = new ArrayList<>(arr.length());
        for (int i = 0; i < arr.length(); i++) {
            JSONObject e = arr.getJSONObject(i);
            out.add(new Fired(e.getInt("frame"), e.getString("alert")));
        }
        return out;
    }

    private static boolean sameSequence(List<Fired> a, List<Fired> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i).frame != b.get(i).frame) return false;
            if (!a.get(i).alert.equals(b.get(i).alert)) return false;
        }
        return true;
    }

    /** Reads from the ANDROID TEST apk's assets, not the app's. */
    private static String readAsset(String name) throws Exception {
        Context testCtx = InstrumentationRegistry.getInstrumentation().getContext();
        try (InputStream in = testCtx.getAssets().open(name)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) bos.write(buf, 0, n);
            return bos.toString("UTF-8");
        }
    }
}
