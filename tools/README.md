# ADAS — models and features

A camera-only advisory driver-assistance pipeline, developed and validated against
CARLA ground truth before the numbers are ported to the Android app.

Five features, four different learned models, plus two classical (non-learned)
components. **Each feature uses a different model — they are not one network.**

---

## Which model does what

| # | Feature | Model / method | Type | Weights | Loaded in |
|---|---------|----------------|------|---------|-----------|
| 1 | **Forward-collision warning** (cars, buses, trucks, motorbikes, bicycles, pedestrians) | **YOLOv8n**, stock COCO weights, unmodified | Learned — object detection | `yolov8n.pt` | [carla_live.py:244](carla_live.py#L244) |
| 2 | **Speed-limit sign detection & reading** | **YOLOv8n fine-tuned** on a CARLA auto-labelled sign dataset. 4 classes = `30 / 40 / 60 / 90`; the class name *is* the speed value, so no OCR step exists | Learned — object detection (transfer-learned from #1's COCO backbone) | `sign_yolov8n.pt` | [carla_live.py:251](carla_live.py#L251) |
| 3 | **Lane-line detection + solid/broken classification** | **SCNN — Spatial CNN** (Pan et al. 2018). Truncated **ResNet-18** encoder (stride 8) → spatial message-passing → two heads: 3-class per-pixel segmentation + per-lane existence (`absent/broken/solid`) | Learned — segmentation + classification, trained from scratch on CARLA data | `scnn_lane/weights/scnn_lane.pt` | [lane_detect.py:148](lane_detect.py#L148), [scnn_lane/model.py](scnn_lane/model.py) |
| 4 | **Generic obstacle warning** (trees, poles, walls, debris — anything unnamed) | **Depth Anything V2 – Small** (`depth-anything/Depth-Anything-V2-Small-hf`), monocular **relative** depth, via HuggingFace `transformers` | Learned — monocular depth estimation (pretrained, not fine-tuned) | downloaded to the HF cache | [depth_obstacle.py:84](depth_obstacle.py#L84) |
| 5 | **Looming detector** (comparison baseline for #4, off by default) | **No neural network.** Shi-Tomasi corners + Lucas-Kanade optical flow + RANSAC similarity fit; `TTC = dt / (scale − 1)` | Classical CV | — | [obstacle_flow.py](obstacle_flow.py) |
| 6 | **Tracking, distance, TTC, alert arbitration** | **No model.** IoU tracker + pinhole width-based distance + hysteresis/cooldown state machine | Deterministic logic | — | [adas_core.py](adas_core.py) |
| 7 | **End-of-trip coaching notes** | **Gemini 2.0 Flash** (`gemini-2.0-flash`) over plain REST. Only turns already-computed numbers into bullets — it never computes the score | LLM (optional, falls back offline) | — | [trip_summary.py:40](trip_summary.py#L40) |
| 8 | **Alert voice clips** | **gTTS** (Google Text-to-Speech), English + Urdu, pre-rendered to WAV once | TTS (offline at runtime) | `raw/*.wav` | [generate_alerts.py](generate_alerts.py) |

### Two clarifications worth stating explicitly

- **Lane detection does *not* use YOLO.** YOLO gives axis-aligned boxes, which say
  nothing useful about a thin diagonal line, and single-frame local evidence
  disappears exactly when a tyre covers the paint. SCNN was chosen because its
  message-passing propagates evidence along the whole row/column, so occluded rows
  get filled in from clear ones — see the rationale in
  [scnn_lane/model.py](scnn_lane/model.py#L5-L19).
- **There is no OCR anywhere.** The sign model's *class label* carries the speed
  value (`sign_model.names[cls] -> "60"`), so detection and reading are one step —
  [carla_live.py:471](carla_live.py#L471).

---

## Why two separate YOLO models instead of one

Continuing training of `yolov8n.pt` on a sign-only dataset would replace its
detection head (different class count), and there is no combined vehicle+sign
ground truth here to retrain both together. A second small YOLOv8n is the
lower-risk version: same architecture and same export path as the vehicle model
(the one `Detector.java` already parses on Android), just a separate weights file.
Fusing them into one pass is a follow-up once mobile framerate demands it.
Rationale in full: [train_sign_yolo.py:5-14](train_sign_yolo.py#L5-L14).

---

## Why relative depth rather than metric depth

Metric depth models return metres directly but are heavier and domain-tied.
Relative models are scale-free and far lighter (MiDaS Small is ~16 MB as TFLite).
This pipeline uses a relative model and **recovers the scale itself every frame**
by fitting the prediction against road-plane geometry it already knows from camera
height and pitch ([depth_obstacle.py:168](depth_obstacle.py#L168)).

The consequence is that the algorithm is model-agnostic: swap Depth Anything V2
Small for MiDaS Small on a phone and nothing downstream changes. Pass
`model_name=` to `DepthObstacleDetector` to switch.

---

## Where each model's weights come from

| Weights | Origin |
|---------|--------|
| `yolov8n.pt` | Ultralytics COCO-pretrained, downloaded as-is. Never retrained. |
| `sign_yolov8n.pt` | `carla_collect_signs.py` → `check_dataset.py` → `train_sign_yolo.py` (60 epochs, 640 px, starting from `yolov8n.pt`) |
| `scnn_lane/weights/scnn_lane.pt` | `carla_collect_lanes.py` → `train_lane_scnn.py` (30 epochs). Trained from scratch — the ResNet-18 backbone is instantiated with `weights=None`. |
| Depth Anything V2 Small | HuggingFace Hub, pretrained, used frozen |

Both training datasets are **auto-labelled from CARLA ground truth** — nothing here
was hand-annotated. Lane labels come from `wp.left_lane_marking.type`; sign labels
come from projecting `traffic.speed_limit.<N>` actor geometry into the camera.

> `yolo26n.pt` also sits in this directory but is not referenced by any code path —
> the vehicle detector defaults to `yolov8n.pt`.

---

## Running it

```bash
# everything on
python carla_live.py --traffic 15 --lanes --depth-obstacles

# add the optical-flow baseline for comparison
python carla_live.py --traffic 15 --lanes --depth-obstacles --obstacles

# see exactly why the obstacle path did or did not warn
python carla_live.py --lanes --depth-obstacles --debug-obstacle
```

Each learned feature degrades independently rather than failing the run:

| Missing | Behaviour |
|---------|-----------|
| `sign_yolov8n.pt` | Speed limit stays manual (press `L` to cycle) |
| `scnn_lane.pt` | `--lanes` raises with training instructions |
| `transformers` not installed | Depth obstacles print a hint and switch off |
| `GEMINI_API_KEY` unset / request fails | Trip report uses canned coaching tips |
| Alert WAVs absent | Alerts print to console instead of speaking |

**Controls:** `W` forward · `S` brake, then reverse once stopped · `A`/`D` steer ·
`SPACE` handbrake · `P` autopilot · `R` teleport · `L` cycle speed limit · `ESC` quit.

---

## Training the two models you own

```bash
# --- lanes -------------------------------------------------------------
python carla_collect_lanes.py --out lane_data --towns Town01,Town05 --minutes 20
python train_lane_scnn.py --data lane_data --epochs 30

# --- speed-limit signs -------------------------------------------------
python list_signs.py --towns Town01,Town04,Town05   # confirm SIGN_CLASSES first
python carla_collect_signs.py --out sign_yolo_data --minutes 20 --preview
python check_dataset.py                              # eyeball labels BEFORE training
python train_sign_yolo.py --epochs 60
python train_sign_yolo.py --epochs 60 --export tflite   # mobile export
```

## Keeping the Python and the Android app in step

`adas_core.py` is a line-for-line twin of four Java classes in [../app](../app/).
They drifted apart once, silently, and every threshold tuned after that point
bought nothing for the app that actually ships. That is now enforced rather than
requested:

```bash
python run_parity.py        # regenerate, compile, replay, compare — no phone needed
```

It builds the golden trace from `adas_core.py`, compiles the real
`AlertEngine` / `DistanceEstimator` / `IouTracker` / `Track` against a 20-line
stand-in for `RectF`, replays nine scenarios through both, and asserts the alerts
land on the same frames. Two scenarios are regression tests for bugs that reached
the Java: an unfiltered oncoming vehicle, and the over-wide corridor swallowing
the adjacent lane.

**Run it after changing any constant in `adas_core.py`.** The on-device
`AlertEngineParityTest` asserts the same thing against the real framework class —
run that before a road test.

| Script | Purpose |
|---|---|
| `run_parity.py` | The gate above. One command, no device. |
| `gen_parity_golden.py` | Builds the golden trace alone. |
| `export_tflite.py` | YOLOv8n → TFLite for the app, **and** asserts the tensor layout matches `Detector.java`. Needs Python 3.10+, so a separate venv — not `carlaenv`. |

## Requirements

```
carla==0.9.15   (Python 3.8–3.10 only)
ultralytics  torch  torchvision  transformers
opencv-python  numpy  pygame  requests  python-dotenv
gTTS  pydub          # only for generate_alerts.py
matplotlib           # only for evaluate.py
```

---

For the architecture diagram, per-frame execution order, and a file-by-file walk
through the code, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.
