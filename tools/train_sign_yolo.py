"""
train_sign_yolo.py — fine-tune YOLOv8n into a dedicated speed-limit-sign
detector on the dataset carla_collect_signs.py produced.

WHY A SEPARATE MODEL, NOT MERGED INTO THE VEHICLE DETECTOR
    carla_live.py already runs yolov8n.pt for vehicles/pedestrians, trained
    on stock COCO. Continuing that training run on a sign-only dataset would
    replace its detection head (different class count) and there is no
    combined vehicle+sign ground truth here to retrain both together without
    risking the vehicle detector's accuracy. A second small yolov8n is the
    lower-risk v1: same architecture and export path as the vehicle model
    (and the one the Android app's Detector.java already parses), just a
    separate weights file. Fusing the two into one pass is a reasonable
    follow-up once this works and mobile framerate demands it.

USAGE
    python check_dataset.py                       # look BEFORE you train
    python train_sign_yolo.py --epochs 60
    python train_sign_yolo.py --epochs 60 --export tflite   # also produce
                                                              # a mobile export
"""

import argparse
import os

from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.dirname(__file__), "sign_yolo_data", "data.yaml")
DEFAULT_OUT = "sign_yolov8n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_YAML)
    ap.add_argument("--base", default="yolov8n.pt",
                    help="starting weights - COCO-pretrained backbone, "
                    "transfer-learned onto sign classes")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="e.g. 0 for first GPU, or 'cpu' - default lets "
                    "ultralytics pick")
    ap.add_argument("--name", default=DEFAULT_OUT)
    ap.add_argument("--export", default=None, choices=[None, "tflite", "onnx"],
                    help="also export the best weights to this format when "
                    "training finishes")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"no dataset at {args.data}\n"
            "run carla_collect_signs.py first, then check_dataset.py to "
            "sanity-check the labels before spending time here.")

    model = YOLO(args.base)
    kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, name=args.name)
    if args.device is not None:
        kwargs["device"] = args.device
    results = model.train(**kwargs)

    best = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"\nbest weights: {best}")
    print(f"copy it into tools/ as sign_yolov8n.pt to use with carla_live.py --sign-model")

    if args.export:
        exported = YOLO(best).export(format=args.export, imgsz=args.imgsz)
        print(f"exported ({args.export}): {exported}")


if __name__ == "__main__":
    main()
