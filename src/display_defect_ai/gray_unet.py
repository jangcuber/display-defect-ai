"""Shared scalar brightness normalization for gray OLED U-Net training and inference."""

import numpy as np
import torch

from .native_inference import NativeEngine, padded
from .pixel_model import TinyUNet


ARCHITECTURE = "gray_normalized_unet_v1"


def normalize_rgb_chw(image):
    """Map the median intensity to 230 DN with one gain for all RGB channels."""
    sample = image[:, 8::16, 8::16].mean(axis=0)
    center = max(32., float(np.median(sample)))
    return np.ascontiguousarray(image, dtype=np.float32) * (230. / center)


class GrayNativeEngine(NativeEngine):
    def __init__(self, kind, weights, device):
        self.kind, self.device = kind, device
        torch.set_num_threads(4)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if kind != "unet" or checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Expected a grayscale U-Net checkpoint")
        self.model = TinyUNet(checkpoint["width"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()

    def predict_tiles(self, images):
        batch = [normalize_rgb_chw(padded(image)[:, :, ::-1].transpose(2, 0, 1))
                 for image in images]
        tensor = torch.from_numpy(np.stack(batch)).to(self.device) / 255
        with torch.inference_mode():
            maps = self.model(tensor).sigmoid().mul(255).round().to(torch.uint8).cpu().numpy()
        maps[maps < 51] = 0
        return [value[:, :image.shape[0], :image.shape[1]]
                for value, image in zip(maps, images, strict=True)]
