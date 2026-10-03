"""
export_tflite.py — export YOLOv8n to TFLite for the Android app, then VERIFY the
result actually matches what Detector.java expects.

WHY THE VERIFY STEP EXISTS
    Detector.postprocess() reads the output tensor as [1, 84, 8400] and indexes
    it as o[4 + classId][anchor]. Ultralytics has shipped both that layout and
    the transposed [1, 8400, 84] depending on version and export path. A
    transposed model does not crash and does not warn — it silently produces
    boxes made of noise, and you find out on a road test. Thirty seconds of
    checking here is worth more than any amount of on-device debugging.

WHY IT DOES NOT JUST CALL ultralytics export(format="tflite")
    As of ultralytics 8.4.83 that format was renamed to "litert" and gated:

        assert MACOS or (LINUX and not ARM64),
            "LiteRT export only supported on Linux x86 and macOS"

    So it refuses outright on Windows. The gate is on ultralytics' own wrapper,
    not on the underlying tooling — ONNX export works fine here, and onnx2tf runs
    on Windows perfectly well. This script therefore drives the two stages itself:

        yolov8n.pt --[ultralytics]--> yolov8n.onnx --[onnx2tf]--> *_float32.tflite

    Same result, no Linux box, no cloud export account.

WHICH PYTHON
    NOT carlaenv. That is pinned to Python 3.8 for the CARLA wheel, and the
    export chain (onnx2tf, tensorflow) needs 3.10+. Make a separate venv:

        py -3.11 -m venv D:\\adas\\exportenv
        D:\\adas\\exportenv\\Scripts\\activate
        pip install ultralytics onnx onnxslim onnxruntime tensorflow ^
                    onnx2tf sng4onnx onnx_graphsurgeon tf_keras ai-edge-litert

    tf_keras is a hard requirement of onnx2tf and is NOT pulled in automatically;
    without it the import fails with ModuleNotFoundError.

    Then:

        python tools/export_tflite.py

    It writes app/src/main/assets/yolov8n_float32.tflite and prints a PASS/FAIL
    on the tensor layout.
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "app", "src", "main", "assets")

# What Detector.java is hard-coded to expect.
EXPECT_INPUT = (1, 640, 640, 3)
EXPECT_OUTPUT = (1, 84, 8400)     # 4 box params + 80 COCO classes, transposed


def ensure_calibration_data(workdir):
    """
    Provide onnx2tf's accuracy-check sample data, which it can no longer download.

    After converting, onnx2tf runs the same input through both the ONNX graph and
    the TF graph and compares the outputs. It fetches a 3.9 MB .npy of sample
    images for that, from:

        https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/<name>
        https://s3.us-central-1.wasabisys.com/onnx2tf-en/datas/<name>   (fallback)

    Both now return 404. onnx2tf does not check the HTTP status — it hands the
    404 body straight to np.load, which fails with:

        ValueError: This file contains pickled (object) data...

    which says nothing about the actual problem. That failure aborts the whole
    conversion, so a dead download link takes the export down with it.

    It looks in the working directory before downloading, so putting a valid file
    there bypasses the fetch entirely. The array is only dummy input for a
    numerical equivalence check — the conversion itself does not depend on its
    contents — so uniform random values in [0, 1] serve the purpose. Arguably
    better than photographs, since random input exercises the numeric range more
    evenly and is more likely to surface an op mismatch.

    Seeded, so repeat exports compare against identical input.
    """
    name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    dest = os.path.join(workdir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 3_000_000:
        print(f"      using cached calibration data")
        return dest

    url = ("https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/"
           + name)
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=120) as r, \
                open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh)
        if os.path.getsize(dest) > 3_000_000:
            print(f"      downloaded calibration data")
            return dest
        os.remove(dest)
    except Exception:
        pass   # expected while both mirrors are 404; fall through to synthesis

    import numpy as np
    rng = np.random.default_rng(0)
    data = rng.random((20, 128, 128, 3), dtype=np.float32)
    np.save(dest, data, allow_pickle=False)
    print(f"      calibration data unavailable upstream (404) - "
          f"synthesised {os.path.getsize(dest) / 1e6:.1f} MB")
    return dest


def find_export(root):
    """Locate the float32 .tflite onnx2tf produced, newest first."""
    candidates = []
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".tflite") and "float32" in f:
                candidates.append(os.path.join(base, f))
    if not candidates:
        for base, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".tflite") and "int8" not in f:
                    candidates.append(os.path.join(base, f))
    return sorted(candidates, key=os.path.getmtime)[-1] if candidates else None


def verify(path):
    """Load the model and check its tensor shapes against Detector.java."""
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
    except ImportError:
        print("\n  ! cannot verify: no tflite interpreter available")
        print("    pip install tensorflow   (or ai-edge-litert)")
        return None

    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_shape = tuple(int(v) for v in inp["shape"])
    out_shape = tuple(int(v) for v in out["shape"])

    print("\n  input :", in_shape, inp["dtype"].__name__)
    print("  output:", out_shape, out["dtype"].__name__)

    ok = True

    if in_shape != EXPECT_INPUT:
        ok = False
        print(f"\n  FAIL  input is {in_shape}, Detector.java expects {EXPECT_INPUT}")
        if len(in_shape) == 4 and in_shape[1] == in_shape[2]:
            print(f"        -> set Detector.INPUT_SIZE = {in_shape[1]}")

    if out_shape != EXPECT_OUTPUT:
        ok = False
        print(f"\n  FAIL  output is {out_shape}, Detector.java expects {EXPECT_OUTPUT}")
        if len(out_shape) == 3 and out_shape[1] == EXPECT_OUTPUT[2] \
                and out_shape[2] == EXPECT_OUTPUT[1]:
            print("        -> the tensor is TRANSPOSED. In Detector.java:")
            print("             float[][][] output = new float[1][NUM_ANCHORS][ROWS];")
            print("           and in postprocess() swap the indexing:")
            print("             o[a][4 + k]   instead of   o[4 + k][a]")
            print("             o[a][0..3]    instead of   o[0..3][a]")
        elif len(out_shape) == 3:
            print(f"        -> set NUM_ANCHORS = {out_shape[2]}, ROWS = {out_shape[1]}")

    if "int" in out["dtype"].__name__:
        ok = False
        print("\n  FAIL  quantised output. Detector.postprocess() has no "
              "dequantisation step - export float32, or add one.")

    if ok:
        print("\n  PASS  tensor layout matches Detector.java exactly")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-name", default="yolov8n_float32.tflite")
    ap.add_argument("--verify-only", metavar="PATH",
                    help="skip export, just check an existing .tflite")
    args = ap.parse_args()

    if args.verify_only:
        sys.exit(0 if verify(args.verify_only) else 1)

    if sys.version_info < (3, 10):
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} cannot "
              "run the export chain (onnx2tf and tensorflow need 3.10+).")
        print("Make a separate venv - see the module docstring. Do NOT use carlaenv.")
        sys.exit(2)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        sys.exit(2)

    weights = args.weights
    if not os.path.isabs(weights):
        local = os.path.join(HERE, weights)
        if os.path.exists(local):
            weights = local

    # --- stage 1: PyTorch -> ONNX (this part works on Windows) --------------
    print(f"[1/2] exporting {weights} to ONNX at imgsz={args.imgsz} ...")
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    onnx_path = str(onnx_path)
    if not os.path.exists(onnx_path):
        print("ONNX export produced nothing - check the log above")
        sys.exit(1)
    print(f"      -> {onnx_path}")

    # --- stage 2: ONNX -> TFLite via onnx2tf --------------------------------
    tf_dir = os.path.join(os.path.dirname(os.path.abspath(weights)),
                          "onnx2tf_out")
    if os.path.isdir(tf_dir):
        shutil.rmtree(tf_dir, ignore_errors=True)

    print(f"[2/2] converting to TFLite with onnx2tf ...")
    workdir = os.path.dirname(os.path.abspath(weights))
    ensure_calibration_data(workdir)

    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", tf_dir, "-nuo"]
    proc = subprocess.run(cmd, cwd=workdir,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        print(log[-4000:])
        print("\nonnx2tf failed. If the error is ModuleNotFoundError: tf_keras,")
        print("run:  pip install tf_keras")
        sys.exit(1)

    produced = find_export(tf_dir)
    if not produced:
        print(log[-3000:])
        print("conversion finished but no .tflite was found")
        sys.exit(1)
    print(f"      -> {produced}")

    os.makedirs(ASSETS, exist_ok=True)
    dest = os.path.join(ASSETS, args.out_name)
    shutil.copy2(produced, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"copied -> {os.path.normpath(dest)}  ({size_mb:.1f} MB)")

    ok = verify(dest)
    if ok is False:
        print("\nFix Detector.java as described above before installing the app.")
        sys.exit(1)


if __name__ == "__main__":
    main()
