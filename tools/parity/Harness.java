import android.graphics.RectF;

import com.adas.detect.Detection;
import com.adas.logic.AlertEngine;
import com.adas.logic.DistanceEstimator;
import com.adas.track.IouTracker;
import com.adas.track.Track;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Offline parity harness. Runs the REAL com.adas logic classes on a desktop JVM
 * against the golden trace produced by tools/gen_parity_golden.py.
 *
 * Same contract as the on-device AlertEngineParityTest, but runnable without a
 * phone, an emulator or Gradle.
 */
public class Harness {

    static class Frame {
        float ego, limit, obstacle;
        String lane;
        List<Detection> dets = new ArrayList<>();
    }

    static class Scen {
        String name;
        List<Frame> frames = new ArrayList<>();
        List<String> expected = new ArrayList<>();   // "ALERT@frame"
    }

    public static void main(String[] args) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get(args[0]));

        float focal = 750f;
        int frameW = 1280;
        long stepMs = 50, startMs = 100000;

        List<Scen> scens = new ArrayList<>();
        Scen cur = null;
        Frame curFrame = null;

        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty()) continue;
            String[] p = line.split("\\s+");
            switch (p[0]) {
                case "FOCAL":  focal = Float.parseFloat(p[1]); break;
                case "FRAMEW": frameW = Integer.parseInt(p[1]); break;
                case "STEP":   stepMs = Long.parseLong(p[1]); break;
                case "START":  startMs = Long.parseLong(p[1]); break;
                case "SCEN":
                    cur = new Scen();
                    cur.name = p[1];
                    scens.add(cur);
                    break;
                case "F":
                    curFrame = new Frame();
                    curFrame.ego = Float.parseFloat(p[1]);
                    curFrame.limit = Float.parseFloat(p[2]);
                    curFrame.obstacle = p[3].equals("NaN") ? Float.NaN : Float.parseFloat(p[3]);
                    curFrame.lane = p[4].equals("-") ? null : p[4];
                    cur.frames.add(curFrame);
                    break;
                case "D":
                    curFrame.dets.add(new Detection(
                            new RectF(Float.parseFloat(p[1]), Float.parseFloat(p[2]),
                                      Float.parseFloat(p[3]), Float.parseFloat(p[4])),
                            Integer.parseInt(p[5]), Float.parseFloat(p[6])));
                    break;
                case "E":
                    cur.expected.add(p[2] + "@" + p[1]);
                    break;
                case "ENDSCEN":
                    break;
                default:
                    throw new IllegalStateException("bad line: " + line);
            }
        }

        DistanceEstimator.FOCAL_PX = focal;

        int failures = 0;
        for (Scen s : scens) {
            List<String> actual = replay(s, frameW, startMs, stepMs);
            boolean ok = actual.equals(s.expected);
            if (!ok) failures++;
            System.out.printf("%-4s %-26s python=%-42s java=%s%n",
                    ok ? "OK" : "FAIL", s.name,
                    s.expected.isEmpty() ? "silent" : String.join(",", s.expected),
                    actual.isEmpty() ? "silent" : String.join(",", actual));
        }

        System.out.println();
        if (failures == 0) {
            System.out.println("PARITY OK - " + scens.size()
                    + " scenarios, Java matches Python exactly");
        } else {
            System.out.println("PARITY FAILED - " + failures + " scenario(s) diverged");
            System.exit(1);
        }
    }

    static List<String> replay(Scen s, int frameW, long startMs, long stepMs) {
        IouTracker tracker = new IouTracker();
        AlertEngine engine = new AlertEngine();
        List<String> fired = new ArrayList<>();
        long simMs = startMs;

        for (int i = 0; i < s.frames.size(); i++) {
            Frame f = s.frames.get(i);
            List<Track> tracks = tracker.update(f.dets);
            for (Track t : tracks) {
                if (t.missed == 0) {
                    t.updateDistance(DistanceEstimator.estimate(t.box, t.classId), simMs);
                }
            }
            AlertEngine.Decision d = engine.evaluate(
                    tracks, frameW, f.ego, f.limit, simMs, f.obstacle, f.lane);
            if (d.fire != null) fired.add(d.fire.name() + "@" + i);
            simMs += stepMs;
        }
        return fired;
    }
}
