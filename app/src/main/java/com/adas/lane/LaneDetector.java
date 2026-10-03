package com.adas.lane;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

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
import java.util.HashMap;
import java.util.Map;

/**
 * UFLD lane detector, trained on BDD100K by tools/train_lane_ufld.py.
 *
 * MODEL CONTRACT
 *   input   [1, 288, 800, 3] float32, RGB, ImageNet-normalised (NOT 0..1 -
 *           the trunk is a pretrained ResNet and expects mean/std applied)
 *   cls     [1, 196, 101]    float32   196 = 49 anchors * 4 slots
 *   style   [1, 4, 3]        float32   per slot: absent / dashed / solid
 *
 * WHY THE OUTPUTS ARE 3-D AND NOT THE NATURAL 4-D
 *   The training graph produces cls as (1, 101, 49, 4). It is flattened to
 *   (1, 196, 101) before export on purpose. onnx2tf rewrites 4-D tensors into
 *   NHWC order, and export_tflite.py exists because that same class of silent
 *   transposition already cost this project a road test once: a permuted
 *   tensor does not crash and does not warn, it just yields noise. 3-D tensors
 *   pass through unpermuted. The shape is asserted in the constructor so a
 *   mismatched export fails loudly here rather than on a motorway.
 *
 * WHY EVERY OUTPUT CAN BE "NOTHING"
 *   Class index 100 of 101 is ABSENT. It is a trained class, not padding, and
 *   on unmarked or worn road it is the correct answer. On top of that,
 *   MIN_CELL_CONFIDENCE rejects a winning cell that only won weakly. Both
 *   gates matter: AlertEngine fires LANE_SOLID with no multi-frame debounce
 *   (see AlertEngine.java:254), so one confident-but-wrong frame is an audible
 *   warning to the driver. This class is the only thing standing there.
 */
public class LaneDetector {

    public static final int INPUT_H = 288;
    public static final int INPUT_W = 800;

    public static final int GRIDING = 100;      // cells across
    public static final int ABSENT = GRIDING;   // class index 100
    public static final int NUM_CELLS = GRIDING + 1;
    public static final int NUM_ANCHORS = 49;
    public static final int NUM_LANES = 4;
    public static final int NUM_STYLES = 3;

    /** Ego lane is bounded by these two slots. 0 and 3 are the lanes beyond. */
    public static final int SLOT_EGO_LEFT = 1;
    public static final int SLOT_EGO_RIGHT = 2;

    public static final int STYLE_ABSENT = 0, STYLE_DASHED = 1, STYLE_SOLID = 2;

    // Row anchors, expressed as a FRACTION of frame height so they survive a
    // change of capture resolution. Source geometry was 1280x720 with anchors
    // at y = 380..620 step 5 (see bdd_to_ufld.py).
    private static final float ROW_TOP_FRAC = 380f / 720f;
    private static final float ROW_STEP_FRAC = 5f / 720f;

    /**
     * A winning cell below this probability is treated as no lane at all.
     *
     * MEASURED, not guessed. tools/tune_lane_threshold.py over all 10000 BDD
     * val images, at the LINE level this class actually gates on:
     *
     *     thresh   claims a line that isn't there   misses a line that is
     *      0.20              3.81%                        27.8%
     *      0.25              2.66%                        35.0%     <- here
     *      0.30              1.81%                        42.1%
     *      0.35              1.19%                        48.5%
     *      0.40              0.75%                        54.1%
     *      0.50              0.30%                        66.5%
     *
     * THIS TABLE BELONGS TO THE MODEL, NOT TO THE APP. It was re-measured for
     * the hidden=512 export (45.5 MB, 22.7M params). The previous hidden=2048
     * model produced a DIFFERENT table - at 0.35 it gave 2.59% false / 36.4%
     * missed - and 0.35 was the right answer for that model, not this one. The
     * smaller model is uniformly more conservative, so the same threshold buys
     * far more silence than intended: left at 0.35 it would miss 48.5% of lane
     * lines rather than 36.4%.
     *
     * 0.25 is chosen to REPRODUCE the old operating point rather than to pick a
     * new one: 2.66% false here against 2.59% before, i.e. the driver-visible
     * false-alarm behaviour is unchanged, while missed lines actually improve
     * slightly (35.0% against 36.4%). If you retrain the lane model, re-run
     * tune_lane_threshold.py and replace this table. Carrying a stale one over
     * is silently choosing an operating point nobody measured.
     *
     * There is no knee in that curve - every 0.05 buys about one point of
     * false lines and costs about five points of missed ones. So this is a
     * judgement about consequences, not an optimum:
     *
     *   a MISSED line costs a warning that never comes, and the driver is
     *   still driving;
     *   a FALSE line costs a warning on empty road, and those are what teach
     *   a driver to ignore the device - including the collision alert, which
     *   is the part that actually prevents a crash.
     *
     * Hence the bias upward. Note also that these figures come from BDD's
     * well-painted American roads; worn Pakistani markings will push the
     * false-line rate HIGHER than the table says, never lower, so an operating
     * point that looks marginal here will be worse in the car. Re-measure on
     * your own recorded sessions and expect to raise this, not lower it.
     */
    public static final float MIN_CELL_CONFIDENCE = 0.25f;

    /** Below this many located anchors a slot is not a line, it is noise. */
    public static final int MIN_ANCHORS_FOR_LINE = 8;

    // ImageNet normalisation - must match ufld_lane/dataset.py exactly.
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    /** One lane boundary as the model sees it. */
    public static final class Lane {
        /** x per anchor as a fraction of frame width; NaN where abstained. */
        public final float[] x = new float[NUM_ANCHORS];
        public final float[] conf = new float[NUM_ANCHORS];
        public int present;          // how many anchors are not NaN
        public int style = STYLE_ABSENT;
        public float styleConf;

        public boolean isLine() { return present >= MIN_ANCHORS_FOR_LINE; }
        public boolean isSolid() { return isLine() && style == STYLE_SOLID; }
    }

    public static final class Result {
        public final Lane[] lanes = new Lane[NUM_LANES];
        public Result() { for (int i = 0; i < NUM_LANES; i++) lanes[i] = new Lane(); }
    }

    private final Interpreter interpreter;
    private NnApiDelegate nnapi;
    private GpuDelegate gpu;
    private final String backend;
    private final ByteBuffer inputBuf;
    /** Bulk-write path — 800*288*3 = 691,200 putFloat calls a frame otherwise.
     *  See the fuller note in Detector. Staging costs 2.8 MB, allocated once. */
    private final FloatBuffer inputFloats;
    private final float[] staging = new float[INPUT_W * INPUT_H * 3];
    private final int[] pixels = new int[INPUT_W * INPUT_H];
    private final float[][][] clsOut = new float[1][NUM_ANCHORS * NUM_LANES][NUM_CELLS];
    private final float[][][] styleOut = new float[1][NUM_LANES][NUM_STYLES];
    private final Result result = new Result();

    private Bitmap scaled;
    private Canvas scaledCanvas;
    private final Rect dstRect = new Rect(0, 0, INPUT_W, INPUT_H);
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);

    /**
     * @param useAccel GPU, then NNAPI, then CPU - the same ladder Detector
     *                 climbs, and for the same reason.
     *
     * This class originally forced CPU, arguing that fp16 rounding would jitter
     * the confidence values that MIN_CELL_CONFIDENCE compares against. That was
     * the wrong call. This model costs roughly what YOLOv8n costs per frame
     * (~8 GFLOPs at 288x800 against YOLO's ~8.7 at 640x640), so running it on
     * CPU alongside an accelerated YOLO would roughly halve the frame rate -
     * and MainActivity's own diagnostics call anything under 8 fps "too slow to
     * track". Protecting a threshold's third decimal place is not worth
     * degrading the collision warning, which is the alert that actually
     * prevents a crash. The threshold has margin for fp16; the frame budget
     * does not have margin for a second full-size CPU model.
     *
     * MUST be constructed on the analysis thread: the GPU delegate binds an EGL
     * context to the calling thread and every later detect() has to come from
     * that same thread.
     */
    public LaneDetector(Context ctx, String modelAsset, int numThreads,
                        boolean useAccel) throws Exception {
        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(numThreads);

        String chosen = "cpu" + numThreads;
        if (useAccel) {
            try {
                if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
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
            // A delegate can report itself supported and still fail to build
            // the graph. Degrade to slow rather than to nothing.
            closeDelegates();
            Interpreter.Options cpu = new Interpreter.Options();
            cpu.setNumThreads(numThreads);
            itp = new Interpreter(loadModel(ctx, modelAsset), cpu);
            chosen = "cpu" + numThreads;
        }
        interpreter = itp;
        backend = chosen;

        int[] in = interpreter.getInputTensor(0).shape();
        int[] c = interpreter.getOutputTensor(0).shape();
        int[] s = interpreter.getOutputTensor(1).shape();
        require(in, new int[]{1, INPUT_H, INPUT_W, 3}, "input");
        require(c, new int[]{1, NUM_ANCHORS * NUM_LANES, NUM_CELLS}, "cls output");
        require(s, new int[]{1, NUM_LANES, NUM_STYLES}, "style output");

        inputBuf = ByteBuffer.allocateDirect(INPUT_W * INPUT_H * 3 * 4)
                .order(ByteOrder.nativeOrder());
        // View taken after the byte order is fixed; see Detector for why.
        inputFloats = inputBuf.asFloatBuffer();
    }

    private static void require(int[] got, int[] want, String what) {
        if (got.length != want.length) throw new IllegalStateException(
                what + " rank " + got.length + ", expected " + want.length);
        for (int i = 0; i < want.length; i++) {
            if (got[i] != want[i]) {
                StringBuilder sb = new StringBuilder(what + " shape [");
                for (int j = 0; j < got.length; j++) sb.append(got[j]).append(j < got.length - 1 ? "," : "");
                sb.append("], expected [");
                for (int j = 0; j < want.length; j++) sb.append(want[j]).append(j < want.length - 1 ? "," : "");
                sb.append("]. Re-export with tools/export_lane_tflite.py.");
                throw new IllegalStateException(sb.toString());
            }
        }
    }

    private static MappedByteBuffer loadModel(Context ctx, String asset) throws Exception {
        try (AssetFileDescriptor afd = ctx.getAssets().openFd(asset);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            return fis.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    afd.getStartOffset(), afd.getDeclaredLength());
        }
    }

    /** Height fraction of anchor i, for mapping model output back to the frame. */
    public static float anchorYFraction(int i) {
        return ROW_TOP_FRAC + i * ROW_STEP_FRAC;
    }

    /**
     * @param bonnetTopFrac height fraction where our own bonnet starts, from
     *                      Calibration. 1.0 means nothing is masked.
     *
     * WHY THE BONNET MATTERS HERE AND NOT JUST IN THE DETECTOR
     *   The row anchors span 52.8%..86.1% of frame height. On a bonnet mount a
     *   real slice of that is the car's own bodywork, and the model has never
     *   been shown a bonnet - BDD's camera sits behind the windscreen looking
     *   over one. Asked about those rows it does not answer "not road", it
     *   answers with whatever the bonnet's edge or a reflection most resembles.
     *   Those rows are the CLOSEST ones, so they carry the most weight in the
     *   straight-line fit that decides a departure. Masking them is not a
     *   refinement; without it the nearest third of the evidence can be a
     *   picture of your own car.
     */
    public Result detect(Bitmap frame, float bonnetTopFrac) {
        preprocess(frame);
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, clsOut);
        outputs.put(1, styleOut);
        interpreter.runForMultipleInputsOutputs(new Object[]{inputBuf}, outputs);
        return decode(bonnetTopFrac);
    }

    /** Anchors not hidden behind our own bonnet. Fewer than MIN_ANCHORS_FOR_LINE
     *  means the mount is too low for this feature to work at all. */
    public int usableAnchors(float bonnetTopFrac) {
        int n = 0;
        for (int a = 0; a < NUM_ANCHORS; a++) {
            if (anchorYFraction(a) < bonnetTopFrac) n++;
        }
        return n;
    }

    public String backend() { return backend; }

    private void preprocess(Bitmap frame) {
        if (scaled == null) {
            scaled = Bitmap.createBitmap(INPUT_W, INPUT_H, Bitmap.Config.ARGB_8888);
            scaledCanvas = new Canvas(scaled);
        }
        // Straight resize, NOT letterboxed. Training resized the full frame the
        // same way (ufld_lane/dataset.py), so aspect distortion is baked into
        // what the model learned. Letterboxing here would be a mismatch.
        scaledCanvas.drawBitmap(frame, null, dstRect, paint);
        scaled.getPixels(pixels, 0, INPUT_W, 0, 0, INPUT_W, INPUT_H);

        int j = 0;
        for (int p : pixels) {
            float r = ((p >> 16) & 0xFF) / 255f;
            float g = ((p >> 8) & 0xFF) / 255f;
            float b = (p & 0xFF) / 255f;
            staging[j++] = (r - MEAN[0]) / STD[0];
            staging[j++] = (g - MEAN[1]) / STD[1];
            staging[j++] = (b - MEAN[2]) / STD[2];
        }
        inputFloats.rewind();
        inputFloats.put(staging);
        inputBuf.rewind();
    }

    private Result decode(float bonnetTopFrac) {
        for (int slot = 0; slot < NUM_LANES; slot++) {
            Lane lane = result.lanes[slot];
            lane.present = 0;

            for (int a = 0; a < NUM_ANCHORS; a++) {
                // Behind our own bonnet: no answer is possible, so do not take
                // one. This is a hard mask, applied before any confidence test,
                // because the problem is not that the model is unsure here -
                // it is that it is looking at the wrong object entirely.
                if (anchorYFraction(a) >= bonnetTopFrac) {
                    lane.x[a] = Float.NaN;
                    lane.conf[a] = 0f;
                    continue;
                }

                float[] row = clsOut[0][a * NUM_LANES + slot];

                // Softmax in one pass, max-subtracted for stability.
                float max = row[0];
                for (int k = 1; k < NUM_CELLS; k++) if (row[k] > max) max = row[k];
                float sum = 0f;
                for (int k = 0; k < NUM_CELLS; k++) sum += (float) Math.exp(row[k] - max);

                float pAbsent = (float) Math.exp(row[ABSENT] - max) / sum;

                // Expected position over the real cells only, never over
                // ABSENT - averaging a categorical "not here" into a
                // coordinate is meaningless. This recovers sub-cell precision.
                float mass = 0f, weighted = 0f, peak = 0f;
                for (int k = 0; k < GRIDING; k++) {
                    float p = (float) Math.exp(row[k] - max) / sum;
                    mass += p;
                    weighted += p * k;
                    if (p > peak) peak = p;
                }

                if (mass > pAbsent && peak >= MIN_CELL_CONFIDENCE) {
                    lane.x[a] = (weighted / mass + 0.5f) / GRIDING;   // 0..1 of width
                    lane.conf[a] = peak;
                    lane.present++;
                } else {
                    lane.x[a] = Float.NaN;
                    lane.conf[a] = 0f;
                }
            }

            float[] st = styleOut[0][slot];
            float m = Math.max(st[0], Math.max(st[1], st[2]));
            float sm = 0f;
            for (int k = 0; k < NUM_STYLES; k++) sm += (float) Math.exp(st[k] - m);
            int best = 0;
            for (int k = 1; k < NUM_STYLES; k++) if (st[k] > st[best]) best = k;
            lane.style = best;
            lane.styleConf = (float) Math.exp(st[best] - m) / sm;
        }
        return result;
    }

    private void closeDelegates() {
        if (gpu != null) { gpu.close(); gpu = null; }
        if (nnapi != null) { nnapi.close(); nnapi = null; }
    }

    public void close() {
        interpreter.close();
        closeDelegates();
        if (scaled != null) { scaled.recycle(); scaled = null; }
    }
}
