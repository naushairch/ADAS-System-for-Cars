package com.adas.detect;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * YOLOv8n TFLite wrapper.
 *
 * Expected model: yolov8n_float32.tflite
 *   input  : [1, 640, 640, 3]  float32, 0..1, RGB
 *   output : [1, 84, 8400]     float32  (4 box params + 80 COCO class scores, transposed)
 *
 * If you export with a different imgsz, change INPUT_SIZE and NUM_ANCHORS to match.
 */
public class Detector {

    public static final int INPUT_SIZE = 640;
    private static final int NUM_CLASSES = 80;
    private static final int NUM_ANCHORS = 8400;
    private static final int ROWS = 4 + NUM_CLASSES; // 84

    // COCO ids we care about. NOTE: no rickshaw / Qingqi class exists in COCO.
    public static final int CLS_PERSON = 0;
    public static final int CLS_BICYCLE = 1;
    public static final int CLS_CAR = 2;
    public static final int CLS_MOTORCYCLE = 3;
    public static final int CLS_BUS = 5;
    public static final int CLS_TRUCK = 7;

    private static final int[] KEEP = {
            CLS_PERSON, CLS_BICYCLE, CLS_CAR, CLS_MOTORCYCLE, CLS_BUS, CLS_TRUCK
    };

    private final Interpreter interpreter;
    private NnApiDelegate nnapi;
    private GpuDelegate gpu;
    private final String backend;

    private final ByteBuffer inputBuf;
    /**
     * FloatBuffer view over inputBuf, plus a staging array.
     *
     * The old loop made one putFloat() call per colour channel — 640*640*3 =
     * 1,228,800 of them per frame, each with its own bounds check and position
     * update. Filling a plain float[] and handing it over in ONE bulk put is the
     * same bytes by a path the JIT can actually vectorise.
     *
     * The staging array costs 4.9 MB, allocated once. That is a real price, but
     * it buys back time on every frame forever, and the alternative was paying
     * for it in per-call overhead 1.2 million times a frame at 9 fps.
     */
    private final FloatBuffer inputFloats;
    private final float[] staging = new float[INPUT_SIZE * INPUT_SIZE * 3];
    private final int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
    private final float[][][] output = new float[1][ROWS][NUM_ANCHORS];

    // Letterbox scratch, allocated once. The old code built two bitmaps per frame
    // (a scaled copy plus the canvas target) and recycled both; at 6 fps that
    // churn was a measurable slice of the budget, and all of it avoidable —
    // Canvas.drawBitmap scales into a dest rect without an intermediate.
    private Bitmap lbBitmap;
    private Canvas lbCanvas;
    private final Rect lbDst = new Rect();
    private final Paint lbPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

    private float confThreshold = 0.35f;
    private float iouThreshold = 0.45f;

    // Letterbox parameters from the most recent preprocess() call.
    private float lbScale = 1f, lbPadX = 0f, lbPadY = 0f;

    /**
     * @param useAccel try GPU, then NNAPI, before falling back to CPU. Pass FALSE
     *                 for replay: a desk replay must reproduce a drive bit-for-bit,
     *                 and GPU inference is fp16 on most Adreno parts, so an
     *                 accelerated replay would not match the CPU run it is compared
     *                 against. Live capture wants the speed; replay wants the
     *                 determinism. They are not the same requirement.
     *
     * IMPORTANT: with useAccel the GPU delegate binds an EGL context to the calling
     * thread, and every later detect() must come from that same thread. Construct
     * this on the analysis thread, not on the main thread.
     */
    public Detector(Context ctx, String modelAsset, int numThreads, boolean useAccel) throws Exception {
        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(numThreads);

        String chosen = "cpu" + numThreads;
        if (useAccel) {
            try {
                if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
                    // fp16 is roughly a 2x win on Adreno and costs a fraction of a
                    // pixel on box corners — irrelevant next to letterbox rounding.
                    gpu = new GpuDelegate(new GpuDelegate.Options()
                            .setPrecisionLossAllowed(true)
                            .setInferencePreference(
                                    GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED));
                    opts.addDelegate(gpu);
                    chosen = "gpu";
                }
            } catch (Throwable t) {
                gpu = null;
            }
            if (gpu == null) {
                try {
                    nnapi = new NnApiDelegate();
                    opts.addDelegate(nnapi);
                    chosen = "nnapi";
                } catch (Throwable t) {
                    nnapi = null;
                }
            }
        }

        Interpreter itp;
        try {
            itp = new Interpreter(loadModel(ctx, modelAsset), opts);
        } catch (Throwable t) {
            // A delegate that reports itself supported can still fail to build the
            // graph. Previously that took the whole app down with "Model load
            // failed"; retry once on plain CPU so a bad delegate degrades to slow
            // rather than to nothing.
            closeDelegates();
            Interpreter.Options cpu = new Interpreter.Options();
            cpu.setNumThreads(numThreads);
            itp = new Interpreter(loadModel(ctx, modelAsset), cpu);
            chosen = "cpu" + numThreads;
        }
        interpreter = itp;
        backend = chosen;

        inputBuf = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuf.order(ByteOrder.nativeOrder());
        // Taken AFTER the byte order is set — a view created before it would
        // keep the buffer's original big-endian order and feed the interpreter
        // byte-swapped floats.
        inputFloats = inputBuf.asFloatBuffer();
    }

    /** Which path inference actually took — shown on the HUD. */
    public String backend() { return backend; }

    private MappedByteBuffer loadModel(Context ctx, String asset) throws Exception {
        AssetFileDescriptor fd = ctx.getAssets().openFd(asset);
        FileInputStream in = new FileInputStream(fd.getFileDescriptor());
        FileChannel ch = in.getChannel();
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    public void setConfThreshold(float t) { this.confThreshold = t; }

    /**
     * Runs detection. Returns boxes mapped back to the coordinate space of the
     * supplied bitmap (i.e. letterbox padding already removed).
     */
    public List<Detection> detect(Bitmap frame) {
        preprocess(frame);
        interpreter.run(inputBuf, output);
        return postprocess();
    }

    private void preprocess(Bitmap src) {
        int w = src.getWidth(), h = src.getHeight();
        lbScale = Math.min((float) INPUT_SIZE / w, (float) INPUT_SIZE / h);
        int nw = Math.round(w * lbScale), nh = Math.round(h * lbScale);
        lbPadX = (INPUT_SIZE - nw) / 2f;
        lbPadY = (INPUT_SIZE - nh) / 2f;

        if (lbBitmap == null) {
            lbBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
            lbCanvas = new Canvas(lbBitmap);
        }
        lbCanvas.drawColor(Color.rgb(114, 114, 114)); // YOLO's grey pad
        lbDst.set(Math.round(lbPadX), Math.round(lbPadY),
                Math.round(lbPadX) + nw, Math.round(lbPadY) + nh);
        lbCanvas.drawBitmap(src, null, lbDst, lbPaint);

        lbBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        int j = 0;
        for (int p : pixels) {
            staging[j++] = ((p >> 16) & 0xFF) / 255f;
            staging[j++] = ((p >> 8) & 0xFF) / 255f;
            staging[j++] = (p & 0xFF) / 255f;
        }
        inputFloats.rewind();
        inputFloats.put(staging);
        inputBuf.rewind();
    }

    private List<Detection> postprocess() {
        List<Detection> raw = new ArrayList<>();
        float[][] o = output[0];

        for (int a = 0; a < NUM_ANCHORS; a++) {
            int bestId = -1;
            float bestScore = confThreshold;
            for (int k : KEEP) {
                float s = o[4 + k][a];
                if (s > bestScore) { bestScore = s; bestId = k; }
            }
            if (bestId < 0) continue;

            float cx = o[0][a], cy = o[1][a], bw = o[2][a], bh = o[3][a];
            // undo letterbox
            float x1 = (cx - bw / 2f - lbPadX) / lbScale;
            float y1 = (cy - bh / 2f - lbPadY) / lbScale;
            float x2 = (cx + bw / 2f - lbPadX) / lbScale;
            float y2 = (cy + bh / 2f - lbPadY) / lbScale;
            raw.add(new Detection(new RectF(x1, y1, x2, y2), bestId, bestScore));
        }
        return nms(raw);
    }

    private List<Detection> nms(List<Detection> in) {
        Collections.sort(in, (a, b) -> Float.compare(b.score, a.score));
        List<Detection> out = new ArrayList<>();
        boolean[] dead = new boolean[in.size()];
        for (int i = 0; i < in.size(); i++) {
            if (dead[i]) continue;
            Detection di = in.get(i);
            out.add(di);
            for (int j = i + 1; j < in.size(); j++) {
                if (dead[j]) continue;
                Detection dj = in.get(j);
                if (dj.classId == di.classId && iou(di.box, dj.box) > iouThreshold) dead[j] = true;
            }
        }
        return out;
    }

    private static float iou(RectF a, RectF b) {
        float ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        float iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float inter = ix * iy;
        float uni = a.width() * a.height() + b.width() * b.height() - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    private void closeDelegates() {
        if (gpu != null) { gpu.close(); gpu = null; }
        if (nnapi != null) { nnapi.close(); nnapi = null; }
    }

    public void close() {
        interpreter.close();
        closeDelegates();
        if (lbBitmap != null) { lbBitmap.recycle(); lbBitmap = null; lbCanvas = null; }
    }
}
