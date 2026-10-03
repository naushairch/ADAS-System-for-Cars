package com.adas;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.adas.audio.AlertPlayer;
import com.adas.calib.Calibration;
import com.adas.calib.FocalSampler;
import com.adas.calib.LensIntrinsics;
import com.adas.detect.Detection;
import com.adas.detect.Detector;
import com.adas.io.SessionLogger;
import com.adas.io.SessionRecorder;
import com.adas.lane.LaneDeparture;
import com.adas.lane.LaneDetector;
import com.adas.road.SpeedLimitProvider;
import com.adas.logic.AlertEngine;
import com.adas.logic.AlertType;
import com.adas.logic.DistanceEstimator;
import com.adas.logic.TripStats;
import com.adas.track.IouTracker;
import com.adas.track.Track;
import com.adas.ui.OverlayView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQ_PERMS = 42;
    /** ~2 seconds at 20 fps — enough for the median to settle out box jitter. */
    private static final int FOCAL_SAMPLES = 40;
    private static final String[] PERMS = {
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION
    };

    private PreviewView previewView;
    private OverlayView overlay;
    private TextView status;

    // Held so their target rotation can be refreshed — see onConfigurationChanged.
    private Preview previewUseCase;
    private ImageAnalysis imageAnalysis;

    private Detector detector;
    private volatile boolean detectorFailed = false;

    // Lane module. Absent-model is a SUPPORTED state, not an error: until
    // tools/export_lane_tflite.py has run there is no ufld_lane asset, and the
    // app must behave exactly as it did before - laneCross stays null and the
    // engine stays silent about lanes. So this fails quietly, once, and never
    // retries, unlike the YOLO detector whose absence is fatal to the app's
    // whole purpose and therefore does deserve a toast.
    private LaneDetector laneDetector;
    private volatile boolean laneFailed = false;
    private final LaneDeparture laneDeparture = new LaneDeparture();
    private volatile long laneMs = 0;
    private volatile String laneInfo = "off";

    // Speed limit source. Entry 0 is AUTO (from the offline road index); every
    // other entry pins that number and ignores the map.
    static final int[] LIMITS = {0, 30, 50, 60, 80, 100, 120};
    private static final String ROAD_ASSET = "roads_pk.bin";
    // New preference key on purpose: the old one stored indices into a list
    // that had no AUTO entry, so reusing it would silently turn a saved 50 into
    // a 30 on first launch after the update.
    private static final String K_LIMIT_IDX = "limit_idx_v2";
    private Button limitBtn;
    private SpeedLimitProvider speedLimits;
    private volatile boolean autoLimit = false;
    private LensIntrinsics lens;
    private volatile boolean diagnosticsVisible = false;
    /** Non-null only while a trip is being scored. Written on the UI thread,
     *  read on the analysis thread — hence volatile. */
    private volatile TripStats trip = null;
    /** 0 until the first GPS fix arrives. Without one, no vehicle alert can fire. */
    private volatile long lastFixMs = 0;
    private volatile boolean testMode = false;
    private final IouTracker tracker = new IouTracker();
    private final AlertEngine engine = new AlertEngine();
    private AlertPlayer player;
    private SessionLogger logger;
    private SessionRecorder recorder;
    private Calibration calibration;

    /** Non-null only while a focal measurement is in progress. */
    private volatile FocalSampler focalSampler;
    /** Stored focal is applied on the first frame, when the real width is known. */
    private boolean calibApplied = false;
    private volatile int analysisWidth = 0;

    private ExecutorService analysisExec;
    private PowerManager.WakeLock wakeLock;

    // ego state
    private volatile float egoSpeedMs = 0f;
    private volatile double lat = 0, lon = 0;
    private volatile float speedLimitMs = 13.9f;   // default 50 km/h; set via UI

    // Filled in by the depth-obstacle module (Phase 4). NaN = nothing credible
    // in the corridor this frame, which is what the engine expects when the
    // module is absent entirely.
    private volatile float obstacleDistM = Float.NaN;
    // Filled in by the lane module (Phase 5). "left"/"right" on the single frame
    // a solid line is crossed, null otherwise.
    private volatile String laneCross = null;

    // fps bookkeeping
    private long lastFrameTs = 0;
    private float fps = 0f;
    private int frameCounter = 0;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        previewView = findViewById(R.id.preview);
        overlay = findViewById(R.id.overlay);
        status = findViewById(R.id.status);

        Button langBtn = findViewById(R.id.langBtn);
        limitBtn = findViewById(R.id.limitBtn);
        Button calBtn = findViewById(R.id.calBtn);
        Button tripBtn = findViewById(R.id.tripBtn);

        player = new AlertPlayer(this);
        logger = new SessionLogger(this);
        recorder = new SessionRecorder(this);
        calibration = new Calibration(this);

        calBtn.setOnClickListener(v -> showCalibrationDialog());
        overlay.setBonnetTopFrac(calibration.bonnetTopFrac());
        analysisExec = Executors.newSingleThreadExecutor();

        // Driver view by default: boxes and a voice, no telemetry. Tapping the
        // picture reveals the diagnostics line. It is deliberately a tap on the
        // whole preview rather than a small control — the one time you want this
        // is while sitting parked wondering whether it is really running, and
        // hunting for a button then is worse than a big dumb target.
        status.setVisibility(View.GONE);
        previewView.setOnClickListener(v -> {
            diagnosticsVisible = !diagnosticsVisible;
            status.setVisibility(diagnosticsVisible ? View.VISIBLE : View.GONE);
        });
        // Long-press anywhere on the picture for the full readiness report, and
        // for the test-mode toggle that lives inside it.
        previewView.setOnLongClickListener(v -> {
            showPreflightDialog();
            return true;
        });

        langBtn.setOnClickListener(v -> {
            AlertPlayer.Lang next = player.getLanguage() == AlertPlayer.Lang.EN
                    ? AlertPlayer.Lang.UR : AlertPlayer.Lang.EN;
            player.setLanguage(next);
            langBtn.setText(next == AlertPlayer.Lang.EN ? "EN" : "UR");
        });

        // The limit is remembered between launches. It used to reset to 50 every
        // time, which is the wrong kind of default: a forgotten limit does not
        // silence "slow down", it makes it fire against the wrong reference.
        //
        // Index 0 is AUTO: the limit comes from the offline road index keyed on
        // the GPS fix (SpeedLimitProvider). Every other entry PINS a number and
        // ignores the map, because a driver who has just told the app the limit
        // is 50 should not be argued with by a stale map extract.
        final android.content.SharedPreferences ui =
                getSharedPreferences("adas_ui", MODE_PRIVATE);
        final int[] idx = {Math.max(0, Math.min(LIMITS.length - 1,
                ui.getInt(K_LIMIT_IDX, 2)))};

        // Derive the live value from the array rather than trusting the field's
        // initialiser to agree with it — they are two statements of the same fact.
        applyLimitChoice(idx[0]);
        limitBtn.setOnClickListener(v -> {
            idx[0] = (idx[0] + 1) % LIMITS.length;
            applyLimitChoice(idx[0]);
            ui.edit().putInt(K_LIMIT_IDX, idx[0]).apply();
        });

        // The road index is a few tens of MB mapped read-only. Absent asset is
        // a supported state, exactly like the lane model: the AUTO entry then
        // simply never produces a number and the manual settings still work.
        try {
            speedLimits = new SpeedLimitProvider(this, ROAD_ASSET);
        } catch (Throwable e) {
            speedLimits = null;
        }

        // Long-press the language button to hear an alert on demand.
        //
        // Audio is this system's ONLY output, and until now there was no way to
        // prove it worked without provoking a genuine near-collision. Media
        // volume muted, Bluetooth routed to a disconnected head unit, clips not
        // loaded — every one of those failures is silent and indistinguishable
        // from "nothing has happened yet". This turns that into a two-second
        // check in the driveway.
        langBtn.setOnLongClickListener(v -> {
            player.play(AlertType.COLLISION_URGENT);
            overlay.banner(alertLabel(AlertType.COLLISION_URGENT),
                    SystemClock.elapsedRealtime() + 1500);
            Toast.makeText(this,
                    "Test alert - if you heard nothing, check MEDIA volume",
                    Toast.LENGTH_LONG).show();
            return true;
        });

        tripBtn.setOnClickListener(v -> {
            if (trip == null) {
                // The check runs HERE rather than behind a button nobody presses.
                // A silent drive is indistinguishable from an uneventful one, so
                // the last moment it is still cheap to find out is this one.
                Preflight p = runPreflight();
                if (p.problems > 0) {
                    new AlertDialog.Builder(this)
                            .setTitle("NOT READY - " + p.problems + " problem(s)")
                            .setMessage(p.text + "\nStarting now means the drive may "
                                    + "record nothing at all.")
                            .setPositiveButton("Start anyway", (d, w) -> beginTrip(tripBtn))
                            .setNegativeButton("Cancel", null)
                            .show();
                    return;
                }
                beginTrip(tripBtn);
            } else {
                TripStats finished = trip;
                trip = null;   // stop counting BEFORE the dialog blocks the UI
                tripBtn.setText("START TRIP");
                showTripSummary(finished);
            }
        });

        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "adas:session");

        if (hasPerms()) start(); else ActivityCompat.requestPermissions(this, PERMS, REQ_PERMS);
    }

    private boolean hasPerms() {
        for (String p : PERMS) {
            if (ContextCompat.checkSelfPermission(this, p) != PackageManager.PERMISSION_GRANTED)
                return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int rc, @NonNull String[] p, @NonNull int[] r) {
        super.onRequestPermissionsResult(rc, p, r);
        if (rc == REQ_PERMS && hasPerms()) start();
        else Toast.makeText(this, "Camera + location required", Toast.LENGTH_LONG).show();
    }

    private void start() {
        // The detector is NOT built here. The GPU delegate ties an EGL context to
        // whichever thread creates the interpreter, and every inference must then
        // come from that same thread — so it is built lazily on the first frame,
        // which arrives on analysisExec. Building it here (main thread) is what
        // limited us to the CPU path, and cost about two thirds of the frame rate.
        // Read once. The lens does not change; only the frame size it is applied
        // to does, and that is handled per-frame in analyze().
        lens = LensIntrinsics.readBackCamera(this);

        wakeLock.acquire(3 * 60 * 60 * 1000L);
        startGps();
        startCamera();
    }

    /** Apply one entry of LIMITS. Index 0 is AUTO. */
    private void applyLimitChoice(int i) {
        int kph = LIMITS[i];
        autoLimit = (kph == 0);
        if (autoLimit) {
            limitBtn.setText("AUTO");
            // speedLimitMs is deliberately NOT zeroed here. Zero disables the
            // speed check outright (AlertEngine.java:264), so an AUTO that has
            // not yet found a road would silently switch the speed warning off
            // altogether. Keeping the previous number is the lesser evil, and
            // the button text says AUTO with no figure until the map speaks.
        } else {
            speedLimitMs = kph / 3.6f;
            limitBtn.setText(kph + " km/h");
        }
    }

    private void startGps() {
        LocationManager lm = (LocationManager) getSystemService(LOCATION_SERVICE);
        LocationListener l = new LocationListener() {
            @Override public void onLocationChanged(@NonNull Location loc) {
                lat = loc.getLatitude();
                lon = loc.getLongitude();
                lastFixMs = SystemClock.elapsedRealtime();
                // GPS speed is far more stable than differentiating positions ourselves.
                if (loc.hasSpeed()) egoSpeedMs = loc.getSpeed();

                // Map lookup is a 3x3 tile scan over a memory-mapped file -
                // tens of microseconds, so it runs inline on the fix rather
                // than being posted somewhere. Returns 0 for "no opinion",
                // which leaves the current limit exactly as it was.
                SpeedLimitProvider sp = speedLimits;
                if (autoLimit && sp != null) {
                    final int kph = sp.update(lat, lon);
                    if (kph > 0) {
                        speedLimitMs = kph / 3.6f;
                        runOnUiThread(() -> limitBtn.setText("AUTO " + kph));
                    }
                }
            }
            @Override public void onProviderDisabled(@NonNull String p) { }
            @Override public void onProviderEnabled(@NonNull String p) { }
        };
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 500, 0f, l);
        } catch (SecurityException ignored) { }
    }

    private void startCamera() {
        ProcessCameraProvider.getInstance(this).addListener(() -> {
            try {
                ProcessCameraProvider provider = ProcessCameraProvider.getInstance(this).get();

                // setTargetResolution() was a REQUEST, not a guarantee, and it is
                // deprecated for exactly that reason. Observed on this device: it
                // silently returned a 1944x1944 SQUARE frame instead of 1280x720
                // — 2.3x the pixels, which halved the frame rate to 8 fps and put
                // the system below the rate its tracker needs.
                //
                // The focal auto-scaled correctly (949 -> 1922, crop correction
                // working), so nothing was WRONG, only unusably slow. That is the
                // dangerous kind of fault: no error, just quiet degradation.
                //
                // ResolutionSelector states the requirement properly — 16:9, as
                // close to 1280x720 as the camera can manage.
                androidx.camera.core.resolutionselector.ResolutionSelector resSelector =
                        new androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                                .setAspectRatioStrategy(androidx.camera.core.resolutionselector
                                        .AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                                .setResolutionStrategy(
                                        new androidx.camera.core.resolutionselector.ResolutionStrategy(
                                                new android.util.Size(1280, 720),
                                                androidx.camera.core.resolutionselector.ResolutionStrategy
                                                        .FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                                .build();

                // The SAME selector goes to the preview.
                //
                // It used to be a bare Preview.Builder().build(), which lets
                // CameraX pick the preview's aspect ratio independently of the
                // analyser's. When it picks 4:3 against the analyser's 16:9, the
                // two streams are different crops of the sensor — the driver is
                // literally looking at a different field of view from the one
                // YOLO is given. No overlay transform can reconcile that, because
                // the boxes describe pixels that are not on screen at all.
                //
                // Sharing the selector makes the two aspect-identical, which is
                // the precondition OverlayView's mapping assumes.
                Preview preview = new Preview.Builder()
                        .setResolutionSelector(resSelector)
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Stated explicitly rather than left to the default, because
                // OverlayView reproduces this exact transform by hand. If it
                // ever changes here it MUST change there, and a silent default
                // is a poor place to hang that contract.
                previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(resSelector)
                        // CRITICAL: never queue frames. A backlog means alerting on
                        // what happened two seconds ago.
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        // RGBA, not YUV. The old path went YUV -> NV21 -> JPEG
                        // compress -> decode, which cost 15-25ms of pure waste
                        // per frame. RGBA is a straight buffer copy.
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();

                analysis.setAnalyzer(analysisExec, this::analyze);

                previewUseCase = preview;
                imageAnalysis = analysis;

                provider.unbindAll();
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                        preview, analysis);
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this,
                        "Camera bind failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    /**
     * Keeps the camera's idea of "up" in step with the screen's.
     *
     * A use case's target rotation is fixed at BIND time, and the manifest
     * declares configChanges="orientation|screenSize", so a device rotation never
     * recreates this activity and nothing else would ever update it. Rotate the
     * phone after launch — which is exactly what happens while clamping it into a
     * dash mount — and the analyser keeps reporting rotationDegrees against the
     * orientation the app started in, while the preview surface corrects itself.
     * The result is boxes rotated 90 degrees away from the picture: a far more
     * dramatic version of the misalignment this change exists to fix, and one
     * that would have looked like the overlay maths being wrong again.
     */
    @Override
    public void onConfigurationChanged(@NonNull android.content.res.Configuration cfg) {
        super.onConfigurationChanged(cfg);
        android.view.Display d = previewView.getDisplay();
        if (d == null) return;
        int rot = d.getRotation();
        if (previewUseCase != null) previewUseCase.setTargetRotation(rot);
        if (imageAnalysis != null) imageAnalysis.setTargetRotation(rot);
    }

    private void analyze(@NonNull ImageProxy proxy) {
        long t0 = SystemClock.elapsedRealtime();
        try {
            if (detector == null) {
                if (detectorFailed) return;
                try {
                    detector = new Detector(this, "yolov8n_float32.tflite", 4, true);
                } catch (Throwable e) {
                    detectorFailed = true;
                    runOnUiThread(() -> Toast.makeText(this,
                            "Model load failed: " + e.getMessage(), Toast.LENGTH_LONG).show());
                    return;
                }
            }

            Bitmap bmp = toBitmap(proxy);
            if (bmp == null) return;

            // The stored focal is in analysis-frame pixels, and the real frame
            // size is only knowable once a frame arrives — CameraX may not honour
            // the requested resolution. Apply (and rescale if needed) on frame one.
            analysisWidth = bmp.getWidth();
            if (!calibApplied) {
                calibApplied = true;
                // Focal is resolved here rather than at startup because it depends
                // on the REAL analysis size, which is only knowable once a frame
                // has actually arrived — CameraX may not honour the request.
                float lensPx = (lens == null) ? Float.NaN
                        : lens.focalPxFor(bmp.getWidth(), bmp.getHeight());
                calibration.resolve(bmp.getWidth(), lensPx);
            }

            long ti = SystemClock.elapsedRealtime();
            List<Detection> dets = detector.detect(bmp);
            long inferMs = SystemClock.elapsedRealtime() - ti;

            // Drop anything lying ENTIRELY below the bonnet line — that is our own
            // car, not traffic. Measured on real footage, YOLO calls the ego bonnet
            // a car in half of all frames; it is wide, central and motionless, so
            // it reads as a vehicle about a metre ahead and never leaves the
            // corridor. Requiring the WHOLE box to be below the line is what keeps
            // this safe: a genuinely close vehicle always extends above it.
            float bonnetFrac = calibration.bonnetTopFrac();
            if (bonnetFrac < 1f) {
                float bonnetY = bonnetFrac * bmp.getHeight();
                List<Detection> kept = new ArrayList<>(dets.size());
                for (Detection det : dets) {
                    if (det.box.top < bonnetY) kept.add(det);
                }
                dets = kept;
            }

            List<Track> tracks = tracker.update(dets);
            long now = SystemClock.elapsedRealtime();
            for (Track t : tracks) {
                if (t.missed == 0) {
                    t.updateDistance(DistanceEstimator.estimate(t.box, t.classId), now);
                }
            }

            final FocalSampler fs = focalSampler;
            if (fs != null) {
                fs.offer(tracks, bmp.getWidth());
                if (fs.isDone()) {
                    focalSampler = null;
                    final int measuredAtWidth = bmp.getWidth();
                    runOnUiThread(() -> showFocalResult(fs, measuredAtWidth));
                }
            }

            // --- Lane departure.
            //
            // Runs on EVERY frame rather than every Nth. Throttling would be the
            // obvious way to buy back the inference cost, but the lane warning is
            // the one alert AlertEngine fires with no multi-frame confirmation
            // (AlertEngine.java:254) precisely because a lane crossing is over in
            // a few frames. Sampling every third frame would hand back exactly
            // the latency that decision was made to avoid.
            //
            // The cost is real and is reported in the diagnostics panel as a
            // separate figure, so if it drags the frame rate down that shows up
            // as a measurement rather than as a mystery.
            if (laneDetector == null && !laneFailed) {
                try {
                    laneDetector = new LaneDetector(this, "ufld_lane_float16.tflite", 2, true);
                } catch (Throwable e) {
                    laneFailed = true;
                    laneInfo = "off (" + e.getClass().getSimpleName() + ")";
                }
            }
            if (laneDetector != null) {
                long tl = SystemClock.elapsedRealtime();
                // The same bonnet line YOLO uses to discard our own bodywork.
                // Rows at or below it show the car, not the road, and the lane
                // model was never trained on a bonnet.
                LaneDetector.Result lr = laneDetector.detect(bmp, bonnetFrac);
                laneMs = SystemClock.elapsedRealtime() - tl;
                // Non-null only on the frame a solid line is actually crossed;
                // null the rest of the time, including every frame where the
                // model declined to commit to a lane at all.
                laneCross = laneDeparture.evaluate(lr, bonnetFrac);
                laneInfo = laneDeparture.describe(lr);
            } else {
                laneCross = null;
            }

            // obstacleDistM is NaN until Phase 4 lands; the engine treats that as
            // "nothing to report" and stays silent on it.
            AlertEngine.Decision d = engine.evaluate(
                    tracks, bmp.getWidth(), egoSpeedMs, speedLimitMs, now,
                    obstacleDistM, laneCross);

            if (d.fire != null) {
                player.play(d.fire);
                logger.alert(now, d.fire.name(), d.reason, egoSpeedMs, lat, lon);
                overlay.banner(alertLabel(d.fire), now + 1500);
                // Snapshot the reference once: the UI thread can null this out
                // mid-frame when the driver ends the trip.
                TripStats t = trip;
                if (t != null) t.record(d.fire);
            }

            // Record the frame that was actually analysed, so a replay reproduces
            // this run exactly. Returns immediately and drops rather than ever
            // stalling this loop.
            recorder.offer(bmp, now, egoSpeedMs, lat, lon, speedLimitMs);

            if (lastFrameTs > 0) {
                float inst = 1000f / Math.max(1, now - lastFrameTs);
                fps = fps == 0 ? inst : (0.1f * inst + 0.9f * fps);
            }
            lastFrameTs = now;

            Track near = nearestInCorridor(tracks, bmp.getWidth());
            long totalMs = SystemClock.elapsedRealtime() - t0;

            if ((frameCounter++ % 3) == 0) {
                logger.frame(now, fps, inferMs, totalMs, egoSpeedMs, lat, lon, tracks.size(),
                        near == null ? -1 : near.id,
                        near == null ? -1 : near.distanceM,
                        near == null ? 0 : near.closingSpeedMs,
                        near == null || Float.isInfinite(near.ttcS) ? -1 : near.ttcS,
                        thermalStatus());
            }

            // The analysed frame size is REQUESTED, not guaranteed — CameraX picks
            // the nearest supported size. FOCAL_PX is expressed in these pixels, so
            // calibration is meaningless unless you know the number it actually
            // gave us. Surfaced on the HUD for exactly that reason.
            // The focal's PROVENANCE matters more than its value: "lens" means
            // derived from hardware and trustworthy to a few percent, "cal" means
            // tape-measured, "PLACEHOLDER" means the distances are fiction.
            final String focalTag;
            switch (calibration.source()) {
                case MANUAL: focalTag = "cal"; break;
                case LENS:   focalTag = "lens"; break;
                default:     focalTag = "PLACEHOLDER"; break;
            }

            final String hudLine = String.format(Locale.US,
                    "%.0f fps | inf %dms %s | %.0f km/h | lim %.0f | trk %d | %dx%d f=%.0f %s",
                    fps, inferMs, detector.backend(),
                    egoSpeedMs * 3.6f, speedLimitMs * 3.6f, tracks.size(),
                    bmp.getWidth(), bmp.getHeight(), DistanceEstimator.FOCAL_PX, focalTag)
                    + (testMode ? "  TESTMODE" : "")
                    + (trip != null ? "  TRIP" : "")
                    + (recorder.isRecording() ? "  " + recorder.status() : "")
                    + (fs != null ? String.format(Locale.US, "  CAL %d/%d",
                            fs.count(), fs.needed()) : "");
            overlay.submit(tracks, bmp.getWidth(), bmp.getHeight());
            // Skip the UI hop entirely while diagnostics are hidden — on a
            // dashboard that is the normal case, and it is a main-thread post per
            // frame for text nobody can see.
            if (diagnosticsVisible) runOnUiThread(() -> status.setText(hudLine));

        } catch (Throwable e) {
            // never let one bad frame kill the analysis loop
        } finally {
            proxy.close();
        }
    }

    // ------------------------------------------------------------ pre-flight

    /**
     * Everything that can silently stop this system working, checked in one
     * place before the driver commits to a trip.
     *
     * WHY THIS EXISTS
     *     Two real faults shipped past every other check: the phone's location
     *     toggle was off, and media volume was zero. Neither produces an error.
     *     Both produce a completely silent drive that looks exactly like "no
     *     hazards occurred", and you only find out after the trip is wasted.
     *
     *     Taking a car out to test is expensive. Anything discoverable in a
     *     driveway must be discovered in the driveway.
     */
    private static class Preflight {
        final StringBuilder text = new StringBuilder();
        int problems = 0;

        void ok(String label, String value) {
            text.append(String.format(Locale.US, "%-16s %-14s OK%n", label, value));
        }

        void bad(String label, String value, String fix) {
            problems++;
            text.append(String.format(Locale.US, "%-16s %-14s <-- FIX%n     %s%n",
                    label, value, fix));
        }

        void warn(String label, String value, String note) {
            text.append(String.format(Locale.US, "%-16s %-14s ?%n     %s%n",
                    label, value, note));
        }
    }

    private Preflight runPreflight() {
        Preflight p = new Preflight();

        // --- GPS. The single most damaging failure: no fix means egoSpeed stays
        // zero, the moving gate never opens, and NOTHING can ever fire.
        boolean providerOn = false;
        try {
            LocationManager lm = (LocationManager) getSystemService(LOCATION_SERVICE);
            providerOn = lm != null && lm.isProviderEnabled(LocationManager.GPS_PROVIDER);
        } catch (Throwable ignored) { }

        long fixAgeS = lastFixMs == 0 ? -1
                : (SystemClock.elapsedRealtime() - lastFixMs) / 1000;

        if (!providerOn) {
            p.bad("GPS", "OFF", "Settings > Location > turn ON, then wait outside");
        } else if (fixAgeS < 0) {
            p.bad("GPS", "no fix yet", "Stand outside with sky view for 30-60 s");
        } else if (fixAgeS > 15) {
            p.warn("GPS", fixAgeS + "s old", "Fix is stale - move somewhere with sky view");
        } else {
            p.ok("GPS", String.format(Locale.US, "%.0f km/h", egoSpeedMs * 3.6f));
        }

        // --- Audio. Alerts play on the MEDIA stream, so the ringer volume is
        // irrelevant and misleading.
        try {
            android.media.AudioManager am =
                    (android.media.AudioManager) getSystemService(AUDIO_SERVICE);
            int vol = am.getStreamVolume(android.media.AudioManager.STREAM_MUSIC);
            int max = Math.max(1, am.getStreamMaxVolume(android.media.AudioManager.STREAM_MUSIC));
            int pct = vol * 100 / max;
            if (vol == 0) {
                p.bad("Media volume", vol + "/" + max,
                        "Press volume UP while nothing is playing");
            } else if (pct < 30) {
                p.warn("Media volume", pct + "%", "Low - you may not hear it over road noise");
            } else {
                p.ok("Media volume", pct + "%");
            }
        } catch (Throwable ignored) { }

        // --- Distance scale.
        switch (calibration.source()) {
            case MANUAL:
                p.ok("Focal", String.format(Locale.US, "%.0f cal", DistanceEstimator.FOCAL_PX));
                break;
            case LENS:
                p.ok("Focal", String.format(Locale.US, "%.0f lens", DistanceEstimator.FOCAL_PX));
                break;
            default:
                p.bad("Focal", "PLACEHOLDER", "Lens data unavailable - run CAL");
        }

        // --- Throughput.
        if (detector == null) {
            p.bad("Detector", "not started", "Camera has not delivered a frame yet");
        } else if (fps < 8f) {
            p.bad("Frame rate", String.format(Locale.US, "%.0f fps", fps),
                    "Too slow to track. Let the phone cool down");
        } else if (fps < 12f) {
            p.warn("Frame rate", String.format(Locale.US, "%.0f fps", fps),
                    "Below 12 - keep the phone out of direct sun");
        } else {
            p.ok("Frame rate", String.format(Locale.US, "%.0f fps %s", fps, detector.backend()));
        }

        // --- Lane module. Its absence is normal (no model exported yet), so it
        // is never counted as a fault. What IS worth seeing is what the model is
        // reporting per side - "L solid/31  R dash/12" - because that is the only
        // way to tell "correctly silent on an unmarked road" apart from "broken".
        // A trailing ! means the located points did not form a straight line.
        if (laneDetector == null) {
            p.warn("Lane", laneInfo, "No lane model - run tools/export_lane_tflite.py");
        } else {
            // The model only reports on rows 52.8%..86.1% down the frame. If the
            // bonnet line sits high in that band, almost none of those rows show
            // road and the feature cannot work however good the model is. Better
            // to say so here than to leave the driver wondering why it is mute.
            int usable = laneDetector.usableAnchors(calibration.bonnetTopFrac());
            if (usable < LaneDetector.MIN_ANCHORS_FOR_LINE) {
                p.bad("Lane", usable + " road rows",
                        "Bonnet hides the road - raise the phone or re-measure the bonnet line");
            } else {
                p.ok("Lane", String.format(Locale.US, "%s  %d ms %s  %d rows",
                        laneInfo, laneMs, laneDetector.backend(), usable));
            }
        }

        // --- Speed limit source. The match distance and the hit rate are the
        // two numbers that say whether AUTO can be trusted on THIS route: a
        // limit snapped from 24 m away, or a hit rate of 40%, means the index
        // is guessing at which road you are on.
        if (speedLimits == null) {
            p.warn("Limit", "manual only",
                    "No road index - run tools/build_road_index.py");
        } else if (!autoLimit) {
            p.ok("Limit", String.format(Locale.US, "pinned %.0f km/h", speedLimitMs * 3.6f));
        } else {
            p.ok("Limit", String.format(Locale.US, "%s  %.0f%% matched",
                    speedLimits.describe(), 100f * speedLimits.hitRate()));
        }

        // --- Bonnet mask. Too low and it eats the part of the frame where a
        // close vehicle lives; 1.0 means nothing is masked at all.
        float bonnetPct = calibration.bonnetTopFrac() * 100f;
        if (bonnetPct <= 55f) {
            // NOT counted as a problem — warn() does not increment. This is a
            // "confirm", not a "fix". 50% is the clamp floor, so a deliberate 50
            // and an accidental 30 are indistinguishable once stored; only the
            // driver looking at the orange line can tell them apart.
            p.warn("Bonnet line", String.format(Locale.US, "%.0f%%", bonnetPct),
                    "Masks the lower half. Fine IF the orange line sits just above "
                    + "your bonnet - check once, then ignore this");
        } else {
            p.ok("Bonnet line", String.format(Locale.US, "%.0f%%", bonnetPct));
        }

        if (testMode) {
            p.warn("TEST MODE", "ON", "Speed gate lowered - turn OFF for a real drive");
        }

        return p;
    }

    private void showPreflightDialog() {
        Preflight p = runPreflight();
        String head = p.problems == 0
                ? "READY\n\n" : ("NOT READY - " + p.problems + " problem(s)\n\n");

        new AlertDialog.Builder(this)
                .setTitle("Pre-flight check")
                .setMessage(head + p.text)
                .setPositiveButton("Close", null)
                .setNeutralButton(testMode ? "Test mode: ON" : "Test mode: OFF", (d, w) -> {
                    setTestMode(!testMode);
                    showPreflightDialog();
                })
                .show();
    }

    /**
     * Drops the speed gate to walking pace so the alert chain can be proved in a
     * car park instead of on a road.
     *
     * It mutates AlertEngine.MIN_SPEED_MS rather than shipping a lower threshold,
     * so the real value is never edited and the parity test still describes the
     * real system. It is never persisted: a process restart puts the static back
     * to 4.2 on its own, so test mode cannot survive into a drive you forgot
     * about.
     */
    private void setTestMode(boolean on) {
        testMode = on;
        AlertEngine.MIN_SPEED_MS = on ? 0.5f : 4.2f;
        Toast.makeText(this, on
                        ? "TEST MODE on - alerts fire from walking pace. Restart clears it."
                        : "Test mode off - normal 15 km/h gate",
                Toast.LENGTH_LONG).show();
    }

    // ---------------------------------------------------------------- trip

    /**
     * Human wording for the on-screen banner. AlertType.name() prints the enum
     * constant — COLLISION_URGENT — which is source code, not a message to a
     * driver glancing at a dashboard.
     */
    private static String alertLabel(AlertType t) {
        switch (t) {
            case COLLISION_URGENT: return "BRAKE";
            case COLLISION_WARN:   return "VEHICLE CLOSE";
            case SPEED_LIMIT:      return "SLOW DOWN";
            case OBSTACLE_AHEAD:   return "OBSTACLE";
            case LANE_SOLID:       return "LANE";
            default:               return t.name();
        }
    }

    private void beginTrip(Button tripBtn) {
        trip = new TripStats();
        tripBtn.setText("END TRIP");
        Toast.makeText(this, "Trip started", Toast.LENGTH_SHORT).show();
    }

    private void showTripSummary(TripStats s) {
        final float score = s.score();
        int mins = (int) (s.durationS() / 60f);
        int secs = (int) (s.durationS() % 60f);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "SCORE  %.1f / 10   (%s)%n%n",
                score, s.grade()));
        sb.append(String.format(Locale.US, "Duration            %d min %02d s%n",
                mins, secs));
        sb.append(String.format(Locale.US, "Total warnings      %d%n",
                s.totalWarnings));
        sb.append(String.format(Locale.US, "   Brake (urgent)   %d%n",
                s.urgentCollisionWarnings));
        sb.append(String.format(Locale.US, "   Vehicle close    %d%n",
                s.collisionWarnings));
        sb.append(String.format(Locale.US, "   Over speed limit %d%n",
                s.speedLimitWarnings));
        // Lane crossings are deliberately not listed. That detector does not
        // exist, so the count is always zero, and printing a permanent zero
        // implies a feature that is watching for something. It is not.
        sb.append("\nHOW TO DRIVE BETTER\n");
        for (String tip : s.recommendations()) {
            sb.append("\n• ").append(tip).append('\n');
        }

        new AlertDialog.Builder(this)
                .setTitle("Trip summary")
                .setMessage(sb.toString())
                .setPositiveButton("Done", null)
                .show();

        saveTripHistory(s, score);
    }

    /**
     * Appends the trip to trip_history.json, matching the schema written by
     * tools/trip_summary.py so both sets of trips can be read by one script.
     * Failure here must never cost the driver their summary, so it is silent.
     */
    private void saveTripHistory(TripStats s, float score) {
        try {
            File dir = getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS);
            if (dir == null) return;
            File f = new File(dir, "trip_history.json");

            JSONArray history = new JSONArray();
            if (f.exists()) {
                StringBuilder raw = new StringBuilder();
                try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                    String line;
                    while ((line = r.readLine()) != null) raw.append(line);
                }
                try {
                    history = new JSONArray(raw.toString());
                } catch (Throwable corrupt) {
                    history = new JSONArray();  // start over rather than lose this trip
                }
            }

            JSONObject o = new JSONObject();
            o.put("timestamp", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                    .format(new Date(s.startedWallMs())));
            o.put("duration_s", Math.round(s.durationS() * 10f) / 10.0);
            o.put("total_warnings", s.totalWarnings);
            o.put("solid_line_warnings", s.solidLineWarnings);
            o.put("collision_warnings", s.collisionWarnings);
            o.put("urgent_collision_warnings", s.urgentCollisionWarnings);
            o.put("speed_limit_warnings", s.speedLimitWarnings);
            o.put("score", score);
            JSONArray recs = new JSONArray();
            for (String tip : s.recommendations()) recs.put(tip);
            o.put("recommendations", recs);

            history.put(o);
            try (FileWriter w = new FileWriter(f, false)) {
                w.write(history.toString(2));
            }
        } catch (Throwable ignored) {
        }
    }

    // ---------------------------------------------------------- calibration

    private void showCalibrationDialog() {
        View view = getLayoutInflater().inflate(R.layout.dialog_calibration, null);
        TextView status = view.findViewById(R.id.calStatus);
        EditText distance = view.findViewById(R.id.calDistance);
        EditText width = view.findViewById(R.id.calWidth);
        EditText height = view.findViewById(R.id.calHeight);
        EditText pitch = view.findViewById(R.id.calPitch);
        EditText bonnet = view.findViewById(R.id.calBonnet);

        status.setText(calibration.summary(analysisWidth));
        distance.setText("10.0");
        width.setText("1.75");
        height.setText(String.format(Locale.US, "%.2f", calibration.camHeightM()));
        pitch.setText(String.format(Locale.US, "%.1f", calibration.camPitchDeg()));
        bonnet.setText(String.format(Locale.US, "%.0f",
                calibration.bonnetTopFrac() * 100f));

        final AlertDialog dlg = new AlertDialog.Builder(this)
                .setTitle("Calibration")
                .setView(view)
                .setPositiveButton("Measure focal", null)
                .setNeutralButton("Save geometry", null)
                .setNegativeButton("Close", null)
                .create();

        // Buttons are wired AFTER show() so they can decline to dismiss. The
        // default behaviour closes the dialog on every tap, which would throw
        // away typed values on a typo and force a restart — done one-handed in a
        // car park, that matters. Only "Measure" and "Close" end the dialog.
        dlg.setOnShowListener(d -> {
            dlg.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                Float distM = parse(distance);
                Float widthM = parse(width);
                if (distM == null || widthM == null || distM <= 0 || widthM <= 0) {
                    toast("Enter a positive distance and target width");
                    return;
                }
                focalSampler = new FocalSampler(distM, widthM, FOCAL_SAMPLES);
                toast("Hold steady - sampling " + FOCAL_SAMPLES + " frames");
                dlg.dismiss();
            });
            dlg.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener(v -> {
                Float h = parse(height);
                Float p = parse(pitch);
                if (h == null || p == null || h <= 0) {
                    toast("Enter a positive height and a pitch");
                    return;
                }
                calibration.saveGeometry(h, p);

                Float b = parse(bonnet);
                if (b != null) {
                    calibration.saveBonnetTopFrac(b / 100f);
                    overlay.setBonnetTopFrac(calibration.bonnetTopFrac());
                }
                status.setText(calibration.summary(analysisWidth));
                toast(String.format(Locale.US,
                        "Saved height %.2f m, pitch %.1f deg, bonnet %.0f%%",
                        h, p, calibration.bonnetTopFrac() * 100f));
            });
        });
        dlg.show();
    }

    private void showFocalResult(FocalSampler fs, int frameWidth) {
        final float focal = fs.focalPx();
        if (Float.isNaN(focal)) {
            toast("No steady target found. Park square behind the vehicle and retry.");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Measured focal length")
                .setMessage(fs.describe()
                        + String.format(Locale.US,
                            "%n%nframe width %d px%ncurrently using %.1f px",
                            frameWidth, DistanceEstimator.FOCAL_PX))
                .setPositiveButton("Save", (d, w) -> {
                    calibration.saveFocal(focal, frameWidth);
                    calibration.applyForFrameWidth(frameWidth);
                    toast(String.format(Locale.US, "Calibrated: focal %.1f px", focal));
                })
                .setNegativeButton("Discard", null)
                .show();
    }

    private Float parse(EditText field) {
        try {
            return Float.parseFloat(field.getText().toString().trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
    }

    private Track nearestInCorridor(List<Track> tracks, int w) {
        Track best = null;
        for (Track t : tracks) {
            if (!t.isConfirmed() || Float.isNaN(t.distanceM)) continue;
            if (!DistanceEstimator.inEgoCorridor(t.box, w, t.distanceM)) continue;
            if (best == null || t.distanceM < best.distanceM) best = t;
        }
        return best;
    }

    private int thermalStatus() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            return pm.getCurrentThermalStatus();
        }
        return -1;
    }

    /**
     * RGBA_8888 ImageProxy -> Bitmap, by direct buffer copy.
     *
     * The analyser is configured for OUTPUT_IMAGE_FORMAT_RGBA_8888, so there is
     * exactly one plane and its bytes already match Android's ARGB_8888 in-memory
     * layout. The only wrinkle is row padding: rowStride can exceed width*4, so
     * we allocate at the padded width and crop.
     *
     * The padded bitmap is cached and reused across frames — this runs 20x a
     * second and allocating a 1280x720 bitmap each time is how you get GC pauses
     * in the middle of a collision warning.
     */
    private Bitmap rgbaScratch;

    private Bitmap toBitmap(ImageProxy proxy) {
        if (proxy.getPlanes().length < 1) return null;
        ImageProxy.PlaneProxy plane = proxy.getPlanes()[0];
        ByteBuffer buf = plane.getBuffer();

        int w = proxy.getWidth();
        int h = proxy.getHeight();
        int pixelStride = plane.getPixelStride();      // 4 for RGBA
        int rowStride = plane.getRowStride();
        int paddedW = rowStride / pixelStride;

        if (rgbaScratch == null
                || rgbaScratch.getWidth() != paddedW
                || rgbaScratch.getHeight() != h) {
            rgbaScratch = Bitmap.createBitmap(paddedW, h, Bitmap.Config.ARGB_8888);
        }
        buf.rewind();
        rgbaScratch.copyPixelsFromBuffer(buf);

        Bitmap bmp = (paddedW == w)
                ? rgbaScratch
                : Bitmap.createBitmap(rgbaScratch, 0, 0, w, h);

        int rot = proxy.getImageInfo().getRotationDegrees();
        if (rot != 0) {
            android.graphics.Matrix m = new android.graphics.Matrix();
            m.postRotate(rot);
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
        }
        return bmp;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analysisExec != null) analysisExec.shutdown();
        if (recorder != null) recorder.close();
        if (detector != null) detector.close();
        if (player != null) player.release();
        if (logger != null) logger.close();
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
    }
}
