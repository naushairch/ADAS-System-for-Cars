"""
ufld_lane/dataset.py — loads what bdd_to_ufld.py wrote.

    <ufld>/train.npz   names (relative paths), grids (N,4,A), styles (N,4)
    <ufld>/val.npz     same
    images live under <data>/bdd100k/bdd100k/images/100k/<split>/...

WHY THE FLIP AUGMENTATION IS NOT A ONE-LINER
    Mirroring a lane image is the single most valuable augmentation here -
    BDD is right-hand-traffic American roads, Pakistan drives on the left,
    so a mirrored BDD frame is closer to the deployment domain than the
    original is. But a naive cv2.flip corrupts the labels twice over:

      * a cell index must become GRIDING_NUM-1-index, and
      * the LANE SLOTS reverse. Slot 1 is "nearest line on my left"; after
        mirroring it is the nearest line on the right, which is slot 2.
        Flip the pixels without reordering the slots and every sample
        teaches the network that left is right.

    Both are handled in _flip below. Absent stays absent under both.
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import GRIDING_NUM, ABSENT, NUM_LANES, INPUT_H, INPUT_W

# ImageNet statistics - the ResNet trunk is pretrained, so its inputs have to
# be normalised the way it was trained or the pretrained features are noise.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _flip(grid: np.ndarray, style: np.ndarray):
    """Mirror targets to match a horizontally flipped image."""
    g = grid.copy()
    present = g != ABSENT
    g[present] = GRIDING_NUM - 1 - g[present]
    return g[::-1].copy(), style[::-1].copy()


class BDDLaneDataset(Dataset):
    def __init__(self, npz_path: str, img_root: str, train: bool = True,
                 flip_p: float = 0.5, jitter: bool = True):
        blob = np.load(npz_path, allow_pickle=False)
        self.names = [str(n) for n in blob["names"]]
        self.grids = blob["grids"].astype(np.int64)
        self.styles = blob["styles"].astype(np.int64)
        self.row_anchors = blob["row_anchors"]
        self.img_root = img_root
        self.train = train
        self.flip_p = flip_p if train else 0.0
        self.jitter = jitter and train

    def __len__(self):
        return len(self.names)

    @property
    def num_anchors(self) -> int:
        return self.grids.shape[2]

    def class_balance(self):
        """Fraction of (slot, anchor) targets that are ABSENT.

        Reported because it drives the loss weighting and because a number
        that drifts far from what the converter printed means the two files
        have gone out of sync."""
        total = self.grids.size
        absent = int((self.grids == ABSENT).sum())
        return absent / total

    def __getitem__(self, i):
        path = os.path.join(self.img_root, self.names[i])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)

        grid = self.grids[i]
        style = self.styles[i]

        if self.flip_p and random.random() < self.flip_p:
            img = cv2.flip(img, 1)
            grid, style = _flip(grid, style)

        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.jitter:
            # Photometric only. Nothing geometric, because any warp would
            # have to be applied to the row anchors too, and an anchor that
            # no longer matches its label is worse than no augmentation.
            img *= random.uniform(0.7, 1.3)                      # brightness
            img = (img - 0.5) * random.uniform(0.8, 1.2) + 0.5    # contrast
            np.clip(img, 0.0, 1.0, out=img)

        img = (img - MEAN) / STD
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())

        return img, torch.from_numpy(grid.copy()), torch.from_numpy(style.copy())
