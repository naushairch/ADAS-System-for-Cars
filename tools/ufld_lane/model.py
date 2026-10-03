"""
ufld_lane/model.py — Ultra-Fast Lane Detection, plus the piece it is missing.

THE UFLD IDEA IN ONE PARAGRAPH
    Segmentation networks label every pixel, which is expensive and mostly
    wasted - a lane line occupies well under 1% of the frame. UFLD instead
    asks a small number of multiple-choice questions: for each lane slot, at
    each of a fixed set of image rows, which of GRIDING_NUM cells across is
    the line in? That is a classification over GRIDING_NUM+1 options (the
    +1 being "not here"), answered NUM_LANES * len(row_anchors) times. The
    whole output is one flat vector off a couple of fully-connected layers,
    which is why it runs at phone speed.

WHY THERE IS A SECOND HEAD
    UFLD answers WHERE a line is. It does not answer WHAT KIND of line it
    is. The app's only lane alert is LANE_SOLID - "crossed solid line" -
    because crossing a dashed line is a normal lane change and must stay
    silent. So position alone cannot drive the feature. The style head adds
    a per-slot {absent, dashed, solid} classification off the same trunk
    features, which costs almost nothing and reuses the vocabulary
    scnn_lane/dataset.py already established.

WHY THE ABSENT CLASS IS THE IMPORTANT ONE
    On faded or unmarked road the correct output is "I cannot see a line",
    not a best guess. That is not bolted on afterwards - it is class index
    GRIDING_NUM in every one of these classifications, trained on directly.
    AlertEngine fires the lane warning with NO multi-frame debounce (see the
    comment at AlertEngine.java:254), so a single confident-but-wrong frame
    becomes an audible warning. Everything here is arranged so that
    uncertainty has somewhere to go other than a wrong answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

# Must agree with bdd_to_ufld.py. Imported by the loader and the exporter so
# these numbers are stated once.
GRIDING_NUM = 100
ABSENT = GRIDING_NUM
NUM_LANES = 4
NUM_STYLES = 3            # 0 absent, 1 dashed, 2 solid

# UFLD's standard input. The row anchors were chosen in the ORIGINAL 1280x720
# frame; resizing is uniform so an anchor keeps its relative height, and the
# network never sees the y coordinates anyway - they are just output slots.
INPUT_H, INPUT_W = 288, 800


class UFLDNet(nn.Module):
    def __init__(self, num_anchors: int, backbone: str = "resnet18",
                 pretrained: bool = True, hidden: int = 2048):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_lanes = NUM_LANES
        self.griding_num = GRIDING_NUM
        self.hidden = hidden

        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet18(weights=weights)
            trunk_ch = 512
        elif backbone == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            net = torchvision.models.resnet34(weights=weights)
            trunk_ch = 512
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        # Everything up to and including layer4. At 288x800 in, layer4 is
        # 512 x 9 x 25 - the /32 stride of a standard ResNet.
        self.trunk = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

        # UFLD squeezes 512 channels to 8 with a 1x1 before flattening.
        # Without this the flat vector is 512*9*25 = 115200 and the first
        # linear layer alone would be ~236M parameters.
        self.squeeze = nn.Conv2d(trunk_ch, 8, kernel_size=1)
        flat = 8 * 9 * 25   # 1800

        # WHY `hidden` IS A KNOB AND NOT A CONSTANT
        #     At hidden=2048 the second Linear is 2048*19796 = 40.5M weights,
        #     73% of the whole network, and the first one EXPANDS a 1800-dim
        #     vector to 2048 before feeding it. That is a bottleneck layer that
        #     does not bottleneck.
        #
        #     At batch size 1 every one of those weights is read once and reused
        #     zero times, so the layer is limited by memory bandwidth rather than
        #     arithmetic: ~81 MB crosses the bus per frame and no delegate fixes
        #     that. hidden=512 cuts it to ~20 MB and the model from 55.9M to
        #     22.7M parameters (112 MB to 45 MB at float16).
        #
        #     BOTH of those figures are theory. The on-phone latency saving has
        #     never been measured, and whether 512 dims can still carry 49
        #     anchors x 4 slots of position decisions is answered by the val
        #     metrics, not by this comment. Compare against the baseline
        #     checkpoint's stored metrics before shipping anything trained here.
        out_dim = (GRIDING_NUM + 1) * num_anchors * NUM_LANES
        self.cls_head = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

        # Style is a far easier problem than position, so it gets a much
        # smaller head. Sharing the trunk is deliberate: whatever features
        # locate a line are the same ones that reveal whether it is broken.
        self.style_head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_LANES * NUM_STYLES),
        )

    def forward(self, x):
        """x: (B,3,288,800)
        returns
            cls   (B, GRIDING_NUM+1, num_anchors, NUM_LANES)
            style (B, NUM_LANES, NUM_STYLES)
        """
        f = self.trunk(x)
        f = self.squeeze(f).flatten(1)
        cls = self.cls_head(f).view(-1, GRIDING_NUM + 1, self.num_anchors, NUM_LANES)
        style = self.style_head(f).view(-1, NUM_LANES, NUM_STYLES)
        return cls, style


@torch.no_grad()
def decode(cls_logits: torch.Tensor, min_confidence: float = 0.0):
    """Turn raw logits into lane x-positions in GRID units, abstaining where
    the model is not sure.

    Two independent ways to say "no line here", and both are needed:
      * the ABSENT class simply wins the argmax, or
      * some cell wins, but with probability below min_confidence.
    The second matters on worn Pakistani markings, where the network will
    often produce a weak, spread-out distribution that still has a winner.
    Taking that winner is how a lane warning fires on a road with no visible
    lane at all.

    Returns (positions, confidence), both (B, num_anchors, NUM_LANES);
    positions hold NaN wherever the model abstained.

    NOTE the expected-value trick: rather than take the winning cell index,
    UFLD averages cell indices weighted by probability, which recovers
    sub-cell precision. Done only over the real cells, never over ABSENT.
    """
    prob_all = cls_logits.softmax(dim=1)                 # (B, G+1, A, L)
    prob_cells = prob_all[:, :GRIDING_NUM, :, :]         # drop ABSENT
    p_absent = prob_all[:, GRIDING_NUM, :, :]

    idx = torch.arange(GRIDING_NUM, device=cls_logits.device,
                       dtype=prob_cells.dtype).view(1, -1, 1, 1)
    mass = prob_cells.sum(dim=1)                          # (B, A, L)
    pos = (prob_cells * idx).sum(dim=1) / mass.clamp_min(1e-6)

    peak = prob_cells.max(dim=1).values                   # (B, A, L)
    ok = (mass > p_absent) & (peak >= min_confidence)
    return torch.where(ok, pos, torch.full_like(pos, float("nan"))), peak
