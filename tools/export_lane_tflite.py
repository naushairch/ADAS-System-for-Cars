"""
export_lane_tflite.py — ufld_lane.pt -> app/src/main/assets/ufld_lane_float32.tflite,
then VERIFY the result matches what LaneDetector.java reads.

    ufld_lane.pt --[torch.onnx]--> ufld_lane.onnx --[onnx2tf]--> *_float32.tflite

WHY THE OUTPUTS ARE RESHAPED TO 3-D BEFORE EXPORT
    The training graph emits cls as (1, 101, 49, 4). onnx2tf rewrites 4-D
    tensors into NHWC, and a silently transposed output is this project's
    known failure mode - export_tflite.py carries the same warning about the
    YOLO head. A transposed lane tensor does not crash and does not warn; it
    produces lane positions made of noise, and a lane warning built on noise
    fires on empty road. 3-D tensors are passed through unpermuted, so cls
    goes out as (1, 196, 101) with index = anchor*4 + slot, which is exactly
    how LaneDetector.decode() indexes it.

WHICH PYTHON
    The training venv (.venv) has torch but NOT tensorflow/onnx2tf. Installing
    them adds about a gigabyte, so do it once, after training:

        d:\\adas\\tools\\.venv\\Scripts\\python -m pip install ^
            onnx onnxslim onnxruntime tensorflow onnx2tf sng4onnx ^
            onnx_graphsurgeon tf_keras ai-edge-litert psutil

    tf_keras AND psutil are hard requirements of onnx2tf that its own metadata
    does not declare; without either, "import onnx2tf" dies with a bare
    ModuleNotFoundError partway through this script rather than at install
    time. Both are already installed in .venv as of this writing.

    Set PIP_CACHE_DIR and TMP to a D: path first - C: does not have the room.

    DO NOT run this while training is running. Importing torch maps around
    2 GB of CUDA DLLs, and on this machine that was enough to push Windows
    past its commit limit and kill a training run's DataLoader worker with
    "[WinError 1455] The paging file is too small". The two jobs each fit
    comfortably alone; they do not fit together.

USAGE
    python export_lane_tflite.py                 # uses ufld_lane/weights/ufld_lane.pt
    python export_lane_tflite.py --skip-verify   # if onnxruntime is unavailable
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

from ufld_lane.model import (UFLDNet, GRIDING_NUM, NUM_LANES, NUM_STYLES,
                             INPUT_H, INPUT_W)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "app", "src", "main", "assets")
DEFAULT_CKPT = os.path.join(HERE, "ufld_lane", "weights", "ufld_lane.pt")

# What LaneDetector.java asserts on load.
EXPECT_INPUT = (1, INPUT_H, INPUT_W, 3)          # NHWC after onnx2tf
NUM_CELLS = GRIDING_NUM + 1


class ExportWrapper(nn.Module):
    """Flattens the head to the 3-D layout the Android side indexes."""

    def __init__(self, net: UFLDNet, num_anchors: int):
        super().__init__()
        self.net = net
        self.num_anchors = num_anchors

    def forward(self, x):
        cls, style = self.net(x)                       # (B,101,A,4), (B,4,3)
        # -> (B, A, 4, 101) -> (B, A*4, 101). Index a*4+slot, matching
        # LaneDetector.decode()'s clsOut[0][a * NUM_LANES + slot].
        cls = cls.permute(0, 2, 3, 1).reshape(1, self.num_anchors * NUM_LANES,
                                              NUM_CELLS)
        return cls, style


def to_onnx(ckpt_path: str, onnx_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_anchors = int(ck["num_anchors"])
    # Checkpoints written before --hidden existed have no such key and are all
    # 2048, so that is the default. Guessing wrong here does not produce a bad
    # model, it produces a load_state_dict shape error, which is the failure
    # mode you want.
    hidden = int(ck.get("hidden", 2048))
    net = UFLDNet(num_anchors=num_anchors, backbone=ck.get("backbone", "resnet18"),
                  pretrained=False, hidden=hidden)
    net.load_state_dict(ck["model"])
    net.eval()

    n_par = sum(p.numel() for p in net.parameters())
    print(f"checkpoint : {ckpt_path}")
    print(f"anchors    : {num_anchors}")
    print(f"hidden     : {hidden}")
    print(f"params     : {n_par/1e6:.1f}M  (~{2*n_par/1e6:.0f} MB at float16)")
    if "metrics" in ck:
        m = ck["metrics"]
        print(f"val        : pos {100*m['pos_acc']:.1f}%  "
              f"false-lane {100*m['false_lane_rate']:.2f}%  "
              f"miss {100*m['miss_rate']:.1f}%  style {100*m['style_acc']:.1f}%")

    wrapper = ExportWrapper(net, num_anchors).eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["image"], output_names=["cls", "style"],
        opset_version=13, do_constant_folding=True,
    )
    print(f"onnx       : {onnx_path}")
    return num_anchors


def to_tflite(onnx_path: str, workdir: str, precision: str):
    """onnx2tf writes several precisions in one pass; we pick one.

    WHY float16 IS THE DEFAULT
        float32 is 223 MB - seventeen times the whole rest of the app, and
        yolov8n_float32.tflite is 12.8 MB for comparison. 72% of the weights
        sit in one Linear(2048 -> 19796) layer, so halving the weight width
        halves the file to 112 MB for no change in what the model computes
        that survives a softmax. It does NOT make inference faster: that
        layer is ~40M multiply-accumulates against the ResNet trunk's ~8
        billion, so the file is mostly weights the arithmetic barely touches.
    """
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", workdir,
           "-nuo", "--non_verbose"]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("onnx2tf failed")
    hits = glob.glob(os.path.join(workdir, f"*{precision}.tflite"))
    if not hits:
        raise SystemExit(f"no {precision} tflite produced in {workdir}")
    return hits[0]


def verify(tflite_path: str, onnx_path: str, num_anchors: int, skip: bool):
    """Shape check always; numerical agreement when onnxruntime is present."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    got_in = tuple(inp["shape"])
    print(f"\ntflite input  : {got_in}")
    ok = got_in == EXPECT_INPUT
    print(f"  expected    : {EXPECT_INPUT}   {'PASS' if ok else 'FAIL'}")

    want_cls = (1, num_anchors * NUM_LANES, NUM_CELLS)
    want_style = (1, NUM_LANES, NUM_STYLES)
    shapes = {tuple(o["shape"]): o for o in outs}
    print(f"tflite outputs: {[tuple(o['shape']) for o in outs]}")
    print(f"  expect cls  : {want_cls}   {'PASS' if want_cls in shapes else 'FAIL'}")
    print(f"  expect style: {want_style}   {'PASS' if want_style in shapes else 'FAIL'}")

    all_ok = ok and want_cls in shapes and want_style in shapes
    if not all_ok:
        raise SystemExit("\nFAIL: layout does not match LaneDetector.java. "
                         "Do NOT ship this model.")

    if skip:
        print("\nnumerical check skipped")
        return

    import onnxruntime as ort
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, INPUT_H, INPUT_W)).astype(np.float32)

    ref = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref_cls, ref_style = ref.run(None, {"image": x})

    interp.set_tensor(inp["index"], np.transpose(x, (0, 2, 3, 1)).copy())
    interp.invoke()
    got = {tuple(o["shape"]): interp.get_tensor(o["index"]) for o in outs}
    tf_cls, tf_style = got[want_cls], got[want_style]

    d_cls = float(np.abs(ref_cls - tf_cls).max())
    d_style = float(np.abs(ref_style - tf_style).max())
    print(f"\nmax |onnx - tflite|  cls {d_cls:.5f}   style {d_style:.5f}")

    # Raw logit distance is the wrong bar for float16 - fp16 carries about
    # three decimal digits, so a logit of magnitude 10 can legitimately move by
    # ~0.01 and nothing downstream notices. What must NOT change is the
    # decision: which cell wins, and whether ABSENT wins. Compare that
    # directly, because that is what LaneDetector.decode() acts on.
    agree = float((ref_cls.argmax(-1) == tf_cls.argmax(-1)).mean())
    agree_style = float((ref_style.argmax(-1) == tf_style.argmax(-1)).mean())
    print(f"decision agreement   cls {100*agree:.2f}%   style {100*agree_style:.2f}%")

    tol = 1e-2 if "float32" in os.path.basename(tflite_path) else 2e-1
    if max(d_cls, d_style) > tol:
        raise SystemExit(f"FAIL: outputs differ by more than {tol}. "
                         "Do NOT ship this model.")
    # A transposition or a broken layer shows up here as agreement collapsing
    # towards chance (1/101), not as a near-miss.
    if agree < 0.98:
        raise SystemExit(f"FAIL: only {100*agree:.1f}% of cell decisions match. "
                         "Do NOT ship this model.")
    print("PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--workdir", default=os.path.join(HERE, "onnx2tf_lane"))
    ap.add_argument("--precision", default="float16",
                    choices=["float16", "float32"])
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"no checkpoint at {args.ckpt} - train first.")
        return 2

    onnx_path = os.path.join(HERE, "ufld_lane.onnx")
    num_anchors = to_onnx(args.ckpt, onnx_path)
    tfl = to_tflite(onnx_path, args.workdir, args.precision)
    verify(tfl, onnx_path, num_anchors, args.skip_verify)

    os.makedirs(ASSETS, exist_ok=True)
    # The asset name carries the precision; LaneDetector names it explicitly so
    # a silent swap between the two is impossible.
    dest = os.path.join(ASSETS, f"ufld_lane_{args.precision}.tflite")
    shutil.copy2(tfl, dest)
    print(f"\ninstalled -> {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
