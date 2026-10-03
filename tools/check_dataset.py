"""
check_dataset.py — look at the labels before you spend time training on them.

Thirty seconds now beats fifteen minutes of training on a broken dataset. Draws
the generated boxes onto a sample of images and writes a contact sheet you can
open and eyeball.

WHAT TO LOOK FOR
    - Every box should sit tightly on a round speed limit sign.
    - The number in the box label should match the sign visible inside it.
    - Boxes on the BACK of a sign (a plain grey disc) are a problem: the model
      would learn to report a speed limit for a sign facing away from you.
      See the facing_dot note in carla_collect_signs.py if you see these.
    - Boxes on empty road, poles, or buildings mean the projection is wrong.

USAGE
    python check_dataset.py
    python check_dataset.py --n 24 --split val
"""

import argparse
import glob
import os
import random

import cv2

DATA = os.path.join(os.path.dirname(__file__), "sign_yolo_data")


def load_names(yaml_path):
    names = {}
    if not os.path.exists(yaml_path):
        return names
    in_names = False
    for line in open(yaml_path):
        st = line.strip()
        if st.startswith("names:"):
            in_names = True
            continue
        if in_names and ":" in st:
            k, v = st.split(":", 1)
            try:
                names[int(k.strip())] = v.strip().strip("'\"")
            except ValueError:
                in_names = False
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img_dir = os.path.join(args.data, "images", args.split)
    lbl_dir = os.path.join(args.data, "labels", args.split)
    if not os.path.isdir(img_dir):
        print(f"no dataset at {img_dir}")
        print("run carla_collect_signs.py first")
        return

    names = load_names(os.path.join(args.data, "data.yaml"))
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not paths:
        print("no images found")
        return

    print(f"{len(paths)} images in {args.split}")

    # Prefer frames with a decently sized box - those are the ones worth
    # inspecting. A contact sheet of distant specks tells you nothing.
    scored = []
    for p in random.sample(paths, min(400, len(paths))):
        lp = os.path.join(lbl_dir, os.path.basename(p).replace(".jpg", ".txt"))
        if not os.path.exists(lp):
            continue
        rows = [r.split() for r in open(lp).read().strip().splitlines() if r]
        if not rows:
            continue
        biggest = max(float(r[3]) for r in rows)
        scored.append((biggest, p, rows))
    scored.sort(reverse=True)

    if not scored:
        print("no labelled frames found in the sampled images - the dataset "
              "looks empty (every label file was blank or missing)")
        return

    picks = scored[:args.n // 2] + random.sample(
        scored[args.n // 2:], min(args.n - args.n // 2,
                                  max(0, len(scored) - args.n // 2)))

    tiles = []
    for _, path, rows in picks:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        for r in rows:
            cls = int(r[0])
            cx, cy, bw, bh = [float(v) for v in r[1:5]]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = names.get(cls, str(cls))
            cv2.putText(img, f"{label} ({x2-x1}px)", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Also save a zoomed crop, since a 30 px box on a 1280 px frame is
            # impossible to judge in a contact sheet.
            pad = int(max(bw * w, bh * h) * 0.6)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size:
                tiles.append((label, cv2.resize(crop, (200, 200))))

    if not tiles:
        print("no boxes found in the sampled labels - the dataset looks empty")
        return

    random.shuffle(tiles)
    tiles = tiles[:24]
    cols = 6
    rows_n = (len(tiles) + cols - 1) // cols
    import numpy as np
    sheet = np.zeros((rows_n * 200, cols * 200, 3), dtype="uint8")
    for i, (label, t) in enumerate(tiles):
        r, c = divmod(i, cols)
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        sheet[r * 200:(r + 1) * 200, c * 200:(c + 1) * 200] = t

    out = args.out or os.path.join(args.data, "label_check.jpg")
    cv2.imwrite(out, sheet)
    print(f"\nwrote {out}")
    print("\nOpen it. Every tile should show a round speed limit sign whose")
    print("number matches the yellow label. If you see the plain grey BACK of")
    print("signs, or boxes on empty road, tell me before training.")


if __name__ == "__main__":
    main()
