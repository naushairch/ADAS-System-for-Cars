# Advisory ADAS — Android / Java / TFLite

Runtime forward-collision and speed-limit voice advisory using a dashboard-mounted
phone camera. Advisory only: it produces **warnings, never permissions**. It never
tells the driver an action is safe.

Target device: OnePlus 8T (Snapdragon 865). Expected ~18–25 fps with YOLOv8n INT8/FP32
via NNAPI.

---

## What is implemented

| Module | Status |
|---|---|
| CameraX capture, keep-only-latest backpressure | done |
| YOLOv8n TFLite detection (car/bus/truck/motorcycle/person/bicycle) | done |
| IoU tracker with track IDs | done |
| Monocular distance from bbox width + ego-corridor gating | done |
| Time-to-collision, smoothed | done |
| GPS speed vs. manual speed limit | done |
| Alert state machine — confirmation, hysteresis, cooldown, priority | done |
| Pre-generated EN/UR audio, SoundPool low-latency playback | done |
| CSV session logging (frames + alerts) | done |
| Lane departure | **not implemented** — `laneOffset` is passed as NaN, engine stays silent |

Lane detection was deliberately cut from the core build. The engine already accepts
a `laneOffset` argument, so adding it later is a drop-in.

---

## Setup — do these in order

### 1. Android Studio
Install Android Studio on Windows 11. Create a new **Empty Views Activity (Java)**
project with package `com.adas`, then replace the generated `app/` contents with
the files from this repo. Sync Gradle.

### 2. Phone
Settings → About phone → tap Build number 7× → Developer options → enable
**USB debugging**. Then on Windows:

```
adb devices
```

If your OnePlus does not appear, install the OnePlus USB drivers. **Sort this out
before anything else** — it is the most common day-one blocker.

### 3. Model
Get `yolov8n_float32.tflite` and place it in `app/src/main/assets/`.

```
pip install ultralytics
yolo export model=yolov8n.pt format=tflite imgsz=640
```

Rename the produced file to `yolov8n_float32.tflite`. If you export INT8 instead,
you must add dequantisation in `Detector.postprocess()` — tell me and I will patch it.

`androidResources { noCompress 'tflite' }` is already set in `build.gradle`. Without
it, memory-mapping the model fails at runtime.

### 4. Audio
```
pip install gTTS pydub numpy
python tools/generate_alerts.py
```
Copy the resulting WAVs from `tools/raw/` into `app/src/main/res/raw/`.
Required filenames: `chime`, `en_brake`, `ur_brake`, `en_vehicle_close`,
`ur_vehicle_close`, `en_lane`, `ur_lane`, `en_slow_down`, `ur_slow_down`.

### 5. Mount
Landscape orientation, roughly centred, clear view over the dashboard. **Mark the
mount position with tape** — if the mount moves between sessions your distance
calibration drifts and your results become uncomparable.

Plug into a car charger. Camera + NNAPI running continuously will drain the 8T fast.

### 6. Calibration — REQUIRED

`DistanceEstimator.FOCAL_PX` is currently a placeholder (750). Until you calibrate,
every distance is plausible but wrong.

In an empty car park:
1. Mount the phone exactly as you will drive with it.
2. Park a car directly ahead at a measured **10 m**. Note the bbox width in px
   (add a temporary log in `analyze()`, or read it off the overlay label).
3. Repeat at **20 m** and **30 m**.
4. For each: `focal = (boxWidthPx × distanceM) / 1.75`
5. Average the three. Put that number in `DistanceEstimator.FOCAL_PX`.
6. Re-run and confirm the overlay now reads within ~1 m at each marker.

Log the three residuals — this is your distance-estimation error metric for the report.

---

## Testing protocol

**Stage A — static.** Car park. Walk toward the phone, drive a car slowly toward it.
Confirm boxes appear, IDs are stable, distances are sane, alerts fire.

**Stage B — passenger seat.** You hold/mount the phone, the driver **ignores the app
entirely and drives normally**. You observe whether alerts fire correctly and when.
Nobody reacts to the system. Do 3–4 sessions across daylight, dusk and night.

**Stage C — tuning.** Pull the logs, compute false alerts per km, adjust thresholds
in `AlertEngine`, repeat. Do not move past this until the false-alarm rate is low.

Pull logs with:
```
adb pull /sdcard/Android/data/com.adas/files/Documents/
```

---

## Evaluation metrics for your report

Compute all of these from `frames_*.csv` and `alerts_*.csv`:

1. **False alerts per km** — the headline number. Manually review each alert against
   your dashcam footage and classify it.
2. **True positive rate** on genuine close calls (you will need to annotate these
   manually from footage).
3. **Distance estimation error** vs. your calibration ground truth at 10/20/30 m.
4. **Alert latency** — clapboard test: trigger a known event, record with a second
   phone at 60fps, count frames from event to audible cue.
5. **Sustained fps over 45 minutes**, plotted, with the `thermal` column overlaid.
   Thermal throttling on a 2020 device in a hot car is a real and reportable finding.
6. **Silence rate** — percentage of driving time the system correctly said nothing.
   Unusual metric, worth including: it reframes the system's job as knowing when
   *not* to speak.

---

## Known limitations — state these explicitly in your report

- **No rickshaw or Qingqi class.** COCO has none. These are misdetected as car,
  motorcycle, or missed entirely. This is the single biggest domain gap for
  Pakistani roads and is worth a paragraph on its own.
- **Monocular distance is unreliable** for partially occluded vehicles, at night,
  and for motorcycles cutting in laterally at close range. Real ADAS uses radar
  for exactly this reason.
- **Ego corridor is a fixed trapezoid**, not real lane geometry. It misjudges on
  curves.
- **No lane detection**, so no lane-departure capability in this build.
- **Advisory only.** Automation complacency is a genuine risk: a system trusted more
  than it deserves is worse than no system.

---

## Tuning cheat sheet

All in `AlertEngine`:

| Symptom | Change |
|---|---|
| Too many collision alerts | raise `CONFIRM_FRAMES`, lower `TTC_WARN_SET` |
| Alerts fire too late | raise `TTC_URGENT_SET` |
| Alerts chatter on/off | widen the gap between `_SET` and `_CLEAR` |
| Constant alerts in traffic | raise `MIN_SPEED_MS` |
| Roadside parked cars trigger it | narrow `corridorHalfM` in `DistanceEstimator` |
| Speed alerts too twitchy | raise `SPEED_CONFIRM_FRAMES` |
