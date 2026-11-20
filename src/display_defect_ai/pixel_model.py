"""Small full-resolution U-Net and common prediction data structures."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, cin, cout, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(4, cout), nn.SiLU(),
        )


class TinyUNet(nn.Module):
    architecture = "tiny_unet8_dilated_v1"

    def __init__(self, width=8):
        super().__init__()
        self.width = width
        widths = [width, width * 2, width * 3, width * 4, width * 6]
        self.stem = Block(3, widths[0])
        self.down = nn.ModuleList(Block(a, b) for a, b in zip(widths, widths[1:]))
        self.context = nn.Sequential(Block(widths[-1], widths[-1], dilation=2),
                                     nn.Conv2d(widths[-1], widths[-1], 3, padding=4, dilation=4), nn.SiLU())
        self.up = nn.ModuleList(Block(a + b, b) for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.head = nn.Conv2d(widths[0], 5, 1)
        nn.init.constant_(self.head.bias, -3.)

    def forward(self, image):
        # Fixed calibration preserves weak brightness/color defects; no per-photo white balance.
        image = (image - 230 / 255) / (16 / 255)
        features = [self.stem(image)]
        for block in self.down:
            features.append(block(F.max_pool2d(features[-1], 2)))
        value = self.context(features[-1])
        for block, skip in zip(self.up, features[-2::-1], strict=True):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((value, skip), dim=1))
        return self.head(value)


def segmentation_loss(logits, target):
    probability = logits.sigmoid()
    positive = ((1 - probability).square() * F.softplus(-logits) * target).sum((0, 2, 3))
    positive = positive / target.sum((0, 2, 3)).clamp_min(1)
    negatives = probability.square() * F.softplus(logits) * (1 - target)
    # Hard negatives in every 16px cell prevent the rare dots being swamped by background.
    cells = F.max_pool2d(negatives, 16, 16)
    # Sparse point errors along a line otherwise vanish in the mean of 4096 cells.
    point_hard = cells[:, :2].permute(1, 0, 2, 3).flatten(1).topk(32, dim=1).values.mean(1)
    negatives = torch.cat((point_hard, cells[:, 2:].mean((0, 2, 3))))
    intersection = (probability * target).sum((0, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probability.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
    return (positive + 2 * negatives + .5 * dice).mean()


def load_unet(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != TinyUNet.architecture:
        raise ValueError("Unknown segmentation architecture")
    model = TinyUNet(checkpoint["width"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def predict_maps(model, images, device="cpu"):
    array = np.stack([np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)) for image in images])
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32) / 255
    with torch.inference_mode():
        scores = model(tensor).sigmoid().mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return scores
