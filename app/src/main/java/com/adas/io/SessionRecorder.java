package com.adas.io;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Records a drive so it can be replayed through ReplayRunner at a desk.
 *
 * WHY THIS EXISTS
 *     You cannot tune alert thresholds while driving. Record once, then iterate
 *     offline against the identical frames through the identical shipping code,
 *     as many times as it takes. That loop is the whole reason ReplayRunner is
 *     in the project.
 *
 * OUTPUT — exactly the layout ReplayRunner expects:
 *     <externalFiles>/replay/<session>/frames/000000.jpg
 *     <externalFiles>/replay/<session>/ego.csv     frame,t_ms,ego_ms,lat,lon,limit_ms
 *
 * TWO DESIGN CONSTRAINTS, both non-obvious:
 *
 *   1. EVERY frame is recorded, never every Nth. The tracker's IoU association
 *      and its closing-speed estimate both depend on the inter-frame interval,
 *      so a decimated recording replays with different dynamics and stops being
 *      a faithful reproduction of the drive. Decimation would quietly invalidate
 *      the thing this exists to enable.
 *
 *   2. The analysis thread is NEVER blocked. MainActivity hands us a bitmap it
 *      reuses next frame, so we memcpy into a small pool (cheap) and let a
 *      background thread do the JPEG encode (expensive). If the pool is empty or
 *      the queue is full we DROP the frame and count it, rather than stalling the
 *      pipeline — a recording with holes is recoverable, a collision warning that
 *      arrived late is not.
 *
 * COST: ~120 KB per frame at 1280x720 q75, so roughly 2.4 MB/s and 145 MB/min.
 * Ten minutes is a good tuning session; MAX_FRAMES stops a forgotten recording
 * from filling the phone.
 */
public class SessionRecorder {

    private static final int JPEG_QUALITY = 75;
    private static final int POOL_SIZE = 3;
    private static final int QUEUE_SIZE = 3;
    /** ~10 minutes at 20 fps. Guards against a recording left running. */
    private static final int MAX_FRAMES = 12000;

    private final Context ctx;

    private ThreadPoolExecutor io;
    private File sessionDir, framesDir;
    private FileWriter egoCsv;

    /** Reusable bitmaps, so recording does not churn 74 MB/s of garbage. */
    private final ArrayDeque<Bitmap> pool = new ArrayDeque<>();
    private int poolAllocated = 0;

    private volatile boolean recording = false;
    private int written = 0;
    private int dropped = 0;
    private String sessionName = "";

    public SessionRecorder(Context ctx) {
        this.ctx = ctx;
    }

    public boolean isRecording() { return recording; }

    public String sessionName() { return sessionName; }

    /** @return the new recording state. */
    public synchronized boolean toggle() {
        if (recording) stop(); else start();
        return recording;
    }

    private void start() {
        sessionName = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(new Date());
        File base = ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        sessionDir = new File(base, "replay/" + sessionName);
        framesDir = new File(sessionDir, "frames");
        if (!framesDir.mkdirs() && !framesDir.isDirectory()) return;

        try {
            egoCsv = new FileWriter(new File(sessionDir, "ego.csv"));
            egoCsv.write("frame,t_ms,ego_ms,lat,lon,limit_ms\n");
        } catch (IOException e) {
            egoCsv = null;
            return;
        }

        // Bounded queue + DiscardPolicy: when the writer falls behind, new frames
        // are thrown away instead of backing up onto the analysis thread.
        io = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(QUEUE_SIZE),
                new ThreadPoolExecutor.DiscardPolicy());

        written = 0;
        dropped = 0;
        recording = true;
    }

    private void stop() {
        recording = false;
        final FileWriter csv = egoCsv;
        final ThreadPoolExecutor exec = io;
        egoCsv = null;
        if (exec != null) {
            exec.execute(() -> {
                try {
                    if (csv != null) { csv.flush(); csv.close(); }
                } catch (IOException ignored) { }
            });
            exec.shutdown();
        }
        synchronized (pool) {
            for (Bitmap b : pool) b.recycle();
            pool.clear();
        }
        io = null;
    }

    /**
     * Offer one analysed frame. Returns immediately; may drop.
     *
     * @param bmp the analysed frame. NOT retained — copied before returning.
     */
    public void offer(Bitmap bmp, long tMs, float egoMs, double lat, double lon,
                      float limitMs) {
        if (!recording || bmp == null) return;
        if (written + dropped >= MAX_FRAMES) { stop(); return; }

        Bitmap copy = borrow(bmp);
        if (copy == null) { dropped++; return; }

        final int index = written++;
        final ThreadPoolExecutor exec = io;
        final FileWriter csv = egoCsv;
        if (exec == null) { release(copy); return; }

        final int before = exec.getQueue().size();
        exec.execute(() -> {
            try {
                File out = new File(framesDir,
                        String.format(Locale.US, "%06d.jpg", index));
                try (FileOutputStream fos = new FileOutputStream(out)) {
                    copy.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, fos);
                }
                if (csv != null) {
                    csv.write(String.format(Locale.US, "%d,%d,%.3f,%.6f,%.6f,%.3f%n",
                            index, tMs, egoMs, lat, lon, limitMs));
                }
            } catch (IOException ignored) {
            } finally {
                release(copy);
            }
        });
        // DiscardPolicy silently dropped it if the queue was already full.
        if (before >= QUEUE_SIZE) dropped++;
    }

    /** Copy into a pooled bitmap, or null when the pool is exhausted. */
    private Bitmap borrow(Bitmap src) {
        Bitmap dst;
        synchronized (pool) {
            dst = pool.poll();
            if (dst != null
                    && (dst.getWidth() != src.getWidth()
                        || dst.getHeight() != src.getHeight())) {
                dst.recycle();
                dst = null;
            }
            if (dst == null && poolAllocated < POOL_SIZE) {
                poolAllocated++;
                dst = Bitmap.createBitmap(src.getWidth(), src.getHeight(),
                        Bitmap.Config.ARGB_8888);
            }
        }
        if (dst == null) return null;

        // Straight pixel copy — far cheaper than the JPEG encode we are deferring.
        android.graphics.Canvas c = new android.graphics.Canvas(dst);
        c.drawBitmap(src, 0f, 0f, null);
        return dst;
    }

    private void release(Bitmap b) {
        synchronized (pool) {
            if (recording && b != null && !b.isRecycled()) pool.offer(b);
            else if (b != null && !b.isRecycled()) b.recycle();
        }
    }

    /** Short status for the HUD, or "" when idle. */
    public String status() {
        if (!recording) return "";
        return dropped == 0
                ? String.format(Locale.US, "REC %d", written)
                : String.format(Locale.US, "REC %d (-%d)", written, dropped);
    }

    public void close() {
        if (recording) stop();
    }
}
