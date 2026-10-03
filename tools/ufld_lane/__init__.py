"""UFLD lane detection for the phone ADAS, trained on BDD100K.

    bdd_to_ufld.py     BDD polylines -> row-anchor targets (+ solid/dashed)
    ufld_lane/model.py the network: UFLD head + a style head UFLD lacks
    ufld_lane/dataset.py loader and augmentation
    train_lane_ufld.py training loop
"""

from .model import UFLDNet, GRIDING_NUM, NUM_LANES, INPUT_H, INPUT_W  # noqa: F401
