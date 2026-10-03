package com.adas.io;

import android.content.Context;
import android.os.Environment;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Writes two CSVs per session into app-external files dir:
 *
 *   frames_<ts>.csv  - one row per processed frame (fps, latency, ego speed,
 *                      nearest in-corridor object, thermal status)
 *   alerts_<ts>.csv  - one row per fired alert
 *
 * These are your evaluation data. False-alerts-per-km, alert latency and fps
 * decay over time are all computed from frames_*.csv, so do not skip this.
 * Pull them with:  adb pull /sdcard/Android/data/com.adas/files/
 */
public class SessionLogger {

    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private FileWriter frames, alerts;
    private final String stamp;
    private final File dir;

    public SessionLogger(Context ctx) {
        stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        dir = ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        try {
            if (dir != null && !dir.exists()) dir.mkdirs();
            frames = new FileWriter(new File(dir, "frames_" + stamp + ".csv"));
            alerts = new FileWriter(new File(dir, "alerts_" + stamp + ".csv"));
            frames.write("t_ms,fps,infer_ms,total_ms,ego_ms,lat,lon,n_tracks,"
                    + "near_id,near_dist_m,near_closing_ms,near_ttc_s,thermal\n");
            alerts.write("t_ms,type,reason,ego_ms,lat,lon\n");
        } catch (IOException ignored) { }
    }

    public void frame(long t, float fps, long inferMs, long totalMs, float egoMs,
                      double lat, double lon, int nTracks,
                      int nearId, float nearDist, float nearClosing, float nearTtc,
                      int thermal) {
        io.execute(() -> {
            try {
                if (frames == null) return;
                frames.write(String.format(Locale.US,
                        "%d,%.1f,%d,%d,%.2f,%.6f,%.6f,%d,%d,%.2f,%.2f,%.2f,%d\n",
                        t, fps, inferMs, totalMs, egoMs, lat, lon, nTracks,
                        nearId, nearDist, nearClosing, nearTtc, thermal));
            } catch (IOException ignored) { }
        });
    }

    public void alert(long t, String type, String reason, float egoMs, double lat, double lon) {
        io.execute(() -> {
            try {
                if (alerts == null) return;
                alerts.write(String.format(Locale.US, "%d,%s,\"%s\",%.2f,%.6f,%.6f\n",
                        t, type, reason, egoMs, lat, lon));
                alerts.flush();
            } catch (IOException ignored) { }
        });
    }

    public String path() { return dir == null ? "?" : dir.getAbsolutePath(); }

    public void close() {
        io.execute(() -> {
            try {
                if (frames != null) { frames.flush(); frames.close(); }
                if (alerts != null) { alerts.flush(); alerts.close(); }
            } catch (IOException ignored) { }
        });
        io.shutdown();
    }
}
