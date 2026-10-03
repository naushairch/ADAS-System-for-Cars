package com.adas.replay;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.SystemClock;

import com.adas.detect.Detection;
import com.adas.detect.Detector;
import com.adas.logic.AlertEngine;
import com.adas.logic.DistanceEstimator;
import com.adas.track.IouTracker;
import com.adas.track.Track;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Runs a recorded frame sequence through the SAME detector, tracker and
 * AlertEngine that the live camera path uses, and writes what the app estimated
 * for every frame.
 *
 * This is the whole point: we validate the shipping code, not a Python
 * re-implementation of it that could silently drift from the real thing.
 *
 * Input layout (push with adb, see README):
 *   <externalFiles>/replay/<name>/frames/000000.jpg ...
 *
 * Output:
 *   <externalFiles>/replay/<name>/app_output.csv
 *
 * Frames are stepped at a FIXED simulated timestep rather than wall-clock, so
 * the result is deterministic and re-runnable while you tune thresholds.
 */
public class ReplayRunner {

    public interface Progress {
        void onProgress(int done, int total);
        void onDone(String outputPath, int alertCount);
        void onError(String message);
    }

    /**
     * Used only for legacy recordings with no ego.csv. Deliberately high so the
     * MIN_SPEED_MS gate does not suppress everything — but note that at this
     * speed the oncoming filter and the obstacle stopping-distance maths are
     * both meaningless. Record with SessionRecorder to get real speeds.
     */
    private static final float FALLBACK_EGO_MS = 999f;

    private final Context ctx;
    private final float dtSeconds;

    public ReplayRunner(Context ctx, float dtSeconds) {
        this.ctx = ctx;
        this.dtSeconds = dtSeconds;
    }

    /**
     * Reads ego.csv as written by SessionRecorder.
     *
     * @return {egoMsPerFrame, limitMsPerFrame}, NaN where a frame has no row, or
     *         null when the file is absent (a legacy recording).
     */
    private static float[][] loadEgo(File sessionDir, int nFrames) {
        File csv = new File(sessionDir, "ego.csv");
        if (!csv.isFile()) return null;

        float[] ego = new float[nFrames];
        float[] limit = new float[nFrames];
        Arrays.fill(ego, Float.NaN);
        Arrays.fill(limit, Float.NaN);

        try (BufferedReader r = new BufferedReader(new FileReader(csv))) {
            r.readLine();   // header
            String line;
            while ((line = r.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length < 6) continue;
                try {
                    int idx = Integer.parseInt(p[0].trim());
                    if (idx < 0 || idx >= nFrames) continue;
                    ego[idx] = Float.parseFloat(p[2].trim());
                    limit[idx] = Float.parseFloat(p[5].trim());
                } catch (NumberFormatException ignored) { }
            }
        } catch (Exception e) {
            return null;
        }
        return new float[][]{ego, limit};
    }

    public void run(String sessionName, float speedLimitMs, Progress cb) {
        File base = new File(ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS),
                "replay/" + sessionName);
        File framesDir = new File(base, "frames");
        if (!framesDir.isDirectory()) {
            cb.onError("No frames at " + framesDir.getAbsolutePath());
            return;
        }

        File[] files = framesDir.listFiles((d, n) -> n.toLowerCase(Locale.US).endsWith(".jpg"));
        if (files == null || files.length == 0) {
            cb.onError("Empty frames directory");
            return;
        }
        Arrays.sort(files);

        Detector detector = null;
        FileWriter out = null;
        int alertCount = 0;

        try {
            // Single thread, NNAPI off: replay is about determinism, not speed.
            detector = new Detector(ctx, "yolov8n_float32.tflite", 4, false);
            IouTracker tracker = new IouTracker();
            AlertEngine engine = new AlertEngine();

            // Per-frame ego speed and speed limit, if SessionRecorder wrote them.
            float[][] ego = loadEgo(base, files.length);

            File outFile = new File(base, "app_output.csv");
            out = new FileWriter(outFile);
            out.write("frame,sim_t_s,ego_ms,est_dist_m,est_closing_ms,est_ttc_s,"
                    + "n_tracks,near_id,in_corridor,alert,reason,infer_ms\n");

            // Simulated clock. AlertEngine cooldowns are in ms, so we advance a
            // virtual millisecond counter rather than reading the system clock.
            long simMs = 100_000L;
            final long stepMs = (long) (dtSeconds * 1000f);

            for (int i = 0; i < files.length; i++) {
                Bitmap bmp = BitmapFactory.decodeFile(files[i].getAbsolutePath());
                if (bmp == null) continue;

                long t0 = SystemClock.elapsedRealtime();
                List<Detection> dets = detector.detect(bmp);
                long inferMs = SystemClock.elapsedRealtime() - t0;

                List<Track> tracks = tracker.update(dets);
                for (Track t : tracks) {
                    if (t.missed == 0) {
                        t.updateDistance(DistanceEstimator.estimate(t.box, t.classId), simMs);
                    }
                }

                Track near = null;
                for (Track t : tracks) {
                    if (!t.isConfirmed() || Float.isNaN(t.distanceM)) continue;
                    if (!DistanceEstimator.inEgoCorridor(t.box, bmp.getWidth(), t.distanceM)) continue;
                    if (near == null || t.distanceM < near.distanceM) near = t;
                }

                // Real per-frame speed when SessionRecorder captured it. Without
                // it, every speed-dependent path — the MIN_SPEED_MS gate, the
                // oncoming test, obstacle stopping distance, the speed-limit
                // check — is bypassed or meaningless, which made replay useless
                // for tuning exactly the thresholds it exists to tune.
                float egoMs = (ego != null && !Float.isNaN(ego[0][i]))
                        ? ego[0][i] : FALLBACK_EGO_MS;
                float limitMs = (ego != null && !Float.isNaN(ego[1][i]))
                        ? ego[1][i] : speedLimitMs;

                AlertEngine.Decision d = engine.evaluate(
                        tracks, bmp.getWidth(), egoMs, limitMs, simMs,
                        Float.NaN, null);

                if (d.fire != null) alertCount++;

                out.write(String.format(Locale.US,
                        "%d,%.3f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%s,\"%s\",%d\n",
                        i, i * dtSeconds, egoMs,
                        near == null ? -1f : near.distanceM,
                        near == null ? 0f : near.closingSpeedMs,
                        near == null || Float.isInfinite(near.ttcS) ? -1f : near.ttcS,
                        tracks.size(),
                        near == null ? -1 : near.id,
                        near == null ? 0 : 1,
                        d.fire == null ? "" : d.fire.name(),
                        d.reason,
                        inferMs));

                bmp.recycle();
                simMs += stepMs;

                if ((i % 10) == 0) cb.onProgress(i, files.length);
            }

            out.flush();
            cb.onDone(outFile.getAbsolutePath(), alertCount);

        } catch (Exception e) {
            cb.onError(e.getClass().getSimpleName() + ": " + e.getMessage());
        } finally {
            try { if (out != null) out.close(); } catch (Exception ignored) { }
            if (detector != null) detector.close();
        }
    }
}
