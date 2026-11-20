"""Versioned coordinate-stable synthesis of paired OLED scenes and native crops.

This is an uncalibrated digital RGB model, not measured OLED optics or luminance.
Noise and convolution are independent of crop size, so a crop can be reproduced
exactly from a full-scene export without storing thousands of large backgrounds.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

VERSION = "paired-oled-renderer-1"
PANEL_W, PANEL_H, SCALE, BORDER, TILE = 1920, 1080, 3, 64, 1024
PHOTO_W, PHOTO_H = PANEL_W * SCALE + 2 * BORDER, PANEL_H * SCALE + 2 * BORDER
KINDS = ("dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region")
CHANNELS = {"B": 0, "G": 1, "R": 2}
KERNEL = np.exp(-np.arange(-3, 4, dtype=np.float64) ** 2 / (2 * .65 ** 2))
KERNEL /= KERNEL.sum()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mix64(values):
    with np.errstate(over="ignore"):
        values = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def noise_window(source, box):
    x1, y1, x2, y2 = box
    indexes = ((np.arange(y1, y2, dtype=np.uint64)[:, None] * PHOTO_W
                + np.arange(x1, x2, dtype=np.uint64)[None, :])[:, :, None] * 3
               + np.arange(3, dtype=np.uint64))
    key = np.uint64(source["noise_key"])
    a = mix64(indexes ^ key)
    b = mix64(indexes ^ key ^ np.uint64(0xD2B74407B1CE6E93))
    u = ((a >> np.uint64(11)).astype(np.float64) + .5) / 2 ** 53
    v = ((b >> np.uint64(11)).astype(np.float64) + .5) / 2 ** 53
    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v) * source["noise_std"]


def coordinates(box, halo=0):
    x1, y1, x2, y2 = box
    x = np.floor_divide(np.arange(x1 - halo, x2 + halo) - BORDER, SCALE)[None, :]
    y = np.floor_divide(np.arange(y1 - halo, y2 + halo) - BORDER, SCALE)[:, None]
    return x, y


def background(source, x, y):
    settings = source["panel"]
    xx, yy = x / PANEL_W - .5, y / PANEL_H - .5
    shading = settings["shading_amplitude"] * np.exp(
        -((xx - settings["shading_x"]) / settings["shading_width"]) ** 2
        -((yy - settings["shading_y"]) / settings["shading_height"]) ** 2)
    field = 1 + settings["slope_x"] * xx + settings["slope_y"] * yy - shading
    values = settings["level"] * field[:, :, None] * np.asarray(settings["tint_bgr"])
    valid = (x >= 0) & (x < PANEL_W) & (y >= 0) & (y < PANEL_H)
    return np.where(valid[:, :, None], values, 24.)


def blur(signal):
    """Fixed summation order gives identical pixels across crop/full render sizes."""
    h, w = signal.shape[0] - 6, signal.shape[1] - 6
    horizontal = np.zeros((h + 6, w, 3), np.float64)
    for index, weight in enumerate(KERNEL):
        horizontal += signal[:, index:index + w] * weight
    result = np.zeros((h, w, 3), np.float64)
    for index, weight in enumerate(KERNEL):
        result += horizontal[index:index + h] * weight
    return result


def base_window(source, box):
    x, y = coordinates(box, 3)
    return blur(background(source, x, y)) + noise_window(source, box)


def quantize(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def native_bounds(spec):
    if spec["kind"] != "dark_region":
        return [spec["x"], spec["y"], spec["x"] + spec["native_width"],
                spec["y"] + spec["native_height"]]
    angle = np.deg2rad(spec["angle"])
    rx, ry = spec["radius_x"] * 1.16, spec["radius_y"] * 1.02
    ex = abs(np.cos(angle)) * rx + abs(np.sin(angle)) * ry
    ey = abs(np.sin(angle)) * rx + abs(np.cos(angle)) * ry
    return [int(np.floor(spec["x"] - ex)), int(np.floor(spec["y"] - ey)),
            int(np.ceil(spec["x"] + ex)) + 1, int(np.ceil(spec["y"] + ey)) + 1]


def raw_region(spec, x, y):
    angle = np.deg2rad(spec["angle"])
    xx, yy = x - spec["x"], y - spec["y"]
    u, v = xx * np.cos(angle) + yy * np.sin(angle), -xx * np.sin(angle) + yy * np.cos(angle)
    rx, ry = spec["radius_x"], spec["radius_y"]
    main = np.maximum(0, 1 - (u / rx) ** 2 - (v / ry) ** 2) ** 3
    shoulder = np.maximum(0, 1 - ((u + rx * .35) / (rx * .8)) ** 2
                          - ((v - ry * .1) / (ry * .9)) ** 2) ** 3
    return .8 * main + .2 * shoulder


def set_region_normalizer(spec):
    x1, y1, x2, y2 = native_bounds(spec)
    x, y = np.arange(x1, x2)[None, :], np.arange(y1, y2)[:, None]
    spec["field_normalizer"] = float(raw_region(spec, x, y).max())


def field_for(spec, x, y):
    if spec["kind"] == "dark_region":
        return raw_region(spec, x, y) / spec["field_normalizer"]
    field = ((x >= spec["x"]) & (x < spec["x"] + spec["native_width"])
             & (y >= spec["y"]) & (y < spec["y"] + spec["native_height"]))
    if spec.get("line_form") == "broken":
        along = x - spec["x"] if spec["kind"] == "horizontal_line" else y - spec["y"]
        field &= along % (spec["segment"] + spec["gap"]) < spec["segment"]
    return field.astype(np.float64)


def attenuation(spec, channel, field, x, y):
    profile = spec["channel_profiles"][channel]
    kind = profile["profile"]
    if kind == "power":
        return field ** profile["exponent"] * spec["channel_strengths"][channel]
    if kind == "constant":
        return field * spec["channel_strengths"][channel]
    horizontal = spec["kind"] == "horizontal_line"
    coordinate, length = (x, spec["native_width"]) if horizontal else (y, spec["native_height"])
    start = spec["x"] if horizontal else spec["y"]
    t = np.clip((coordinate - start) / max(1, length - 1), 0, 1)
    smooth = t * t * (3 - 2 * t)
    if kind == "rising":
        response = smooth
    elif kind == "falling":
        response = 1 - smooth
    elif kind == "center":
        response = 16 * t * t * (1 - t) * (1 - t)
    else:
        raise ValueError(f"Unknown spatial profile: {kind}")
    return field * (.05 + .95 * response) * spec["channel_strengths"][channel]


def effect_patch(source, specs, window):
    bounds = [native_bounds(spec) for spec in specs]
    x1 = max(window[0], BORDER + min(b[0] for b in bounds) * SCALE - 3)
    y1 = max(window[1], BORDER + min(b[1] for b in bounds) * SCALE - 3)
    x2 = min(window[2], BORDER + max(b[2] for b in bounds) * SCALE + 3)
    y2 = min(window[3], BORDER + max(b[3] for b in bounds) * SCALE + 3)
    if x1 >= x2 or y1 >= y2:
        return None, None
    box = [x1, y1, x2, y2]
    x, y = coordinates(box, 3)
    transmission = np.ones((y.shape[0], x.shape[1], 3), np.float64)
    for spec in specs:
        field = field_for(spec, x, y)
        for channel in spec["channels"]:
            transmission[:, :, CHANNELS[channel]] *= 1 - attenuation(spec, channel, field, x, y)
    loss = background(source, x, y) * (1 - transmission)
    return box, blur(loss)


def render_from_base(source, specs, window, base):
    if not specs:
        return quantize(base)
    box, loss = effect_patch(source, specs, window)
    values = base.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        values[y1 - window[1]:y2 - window[1], x1 - window[0]:x2 - window[0]] -= loss
    return quantize(values)


def render_window(source, specs, window):
    return render_from_base(source, specs, window, base_window(source, window))


def render_full(source, specs, chunk=256):
    image = np.empty((PHOTO_H, PHOTO_W, 3), np.uint8)
    for y in range(0, PHOTO_H, chunk):
        window = [0, y, PHOTO_W, min(PHOTO_H, y + chunk)]
        image[y:window[3]] = render_window(source, specs, window)
    return image


def pixel_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def write_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()
