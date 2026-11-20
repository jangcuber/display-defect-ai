"""Shared native-resolution inference and postprocessing for experiments and app.

No dataset labels, normal counterfactuals or synthesis recipes are used here.
"""

from pathlib import Path

import cv2
import numpy as np

from .line_postprocess import Detection, box_iou

KINDS = ("dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region")
LINES = frozenset(KINDS[2:4])
TILE, OVERLAP = 1024, 128
VERSION = "native-inference-1"


def tile_windows(width, height, roi=None):
    left, top, right, bottom = roi or (0, 0, width, height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Invalid inspection region")

    def starts(a, b):
        if b - a <= TILE:
            return [a]
        result = list(range(a, b - TILE + 1, TILE - OVERLAP))
        if result[-1] != b - TILE:
            result.append(b - TILE)
        return result

    return [(x, y, min(x + TILE, right), min(y + TILE, bottom))
            for y in starts(top, bottom) for x in starts(left, right)]


def padded(image):
    h, w = image.shape[:2]
    if h > TILE or w > TILE:
        raise ValueError("Expected a native tile, not an implicitly resized image")
    return cv2.copyMakeBorder(image, 0, TILE - h, 0, TILE - w, cv2.BORDER_REFLECT_101)


def geometry_valid(detection):
    x1, y1, x2, y2 = detection.box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    if detection.kind in KINDS[:2]:
        return max(w, h) <= 32
    if detection.kind in LINES:
        length, cross = (w, h) if detection.kind == "horizontal_line" else (h, w)
        return cross <= 24 and length >= 18 and length >= 4 * cross
    return min(w, h) >= 24


def merge_detections(detections, thresholds, gap=128, region_iou=.35, merge_regions=False):
    candidates = sorted((item for item in detections
                         if item.score >= thresholds[item.kind] and geometry_valid(item)),
                        key=lambda item: item.score, reverse=True)
    kept = []
    for item in candidates:
        if item.kind == "dark_region" and merge_regions:
            merged = item
            changed = True
            while changed:
                changed = False
                for index, other in enumerate(kept):
                    if other.kind == item.kind and box_iou(merged.box, other.box) >= region_iou:
                        a, b = merged.box, other.box
                        merged = Detection(max(merged.score, other.score), item.kind,
                                           (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))
                        kept.pop(index)
                        changed = True
                        break
            kept.append(merged)
            continue
        if item.kind not in LINES:
            if any((other.kind == item.kind or other.kind in KINDS[:2] and item.kind in KINDS[:2])
                   and box_iou(other.box, item.box) >= (region_iou if item.kind == "dark_region" else .35) for other in kept):
                continue
            kept.append(item)
            continue
        axis = 0 if item.kind == "horizontal_line" else 1
        cross = 1 - axis
        merged = item
        changed = True
        while changed:
            changed = False
            for index, other in enumerate(kept):
                if other.kind != item.kind:
                    continue
                a, b = merged.box, other.box
                shift = abs(a[cross] + a[cross + 2] - b[cross] - b[cross + 2]) / 2
                separation = max(a[axis], b[axis]) - min(a[axis + 2], b[axis + 2])
                ta, tb = a[cross + 2] - a[cross], b[cross + 2] - b[cross]
                if shift <= 2 and separation <= gap and max(ta, tb) <= 3 * min(ta, tb):
                    box = list(a if merged.score >= other.score else b)
                    box[axis], box[axis + 2] = min(a[axis], b[axis]), max(a[axis + 2], b[axis + 2])
                    merged = Detection(max(merged.score, other.score), item.kind, tuple(box))
                    kept.pop(index)
                    changed = True
                    break
        kept.append(merged)
    return sorted(kept, key=lambda item: item.score, reverse=True)


def postprocess(detections, selected):
    return merge_detections(detections, selected["thresholds"], selected.get("line_gap", 128),
                            selected.get("region_iou", .35), selected.get("merge_regions", False))


def components(plane, kind, pixel_threshold, min_area):
    cutoff = int(np.ceil(pixel_threshold * 255))
    if plane.max() < cutoff:
        return []
    mask = (plane >= cutoff).astype(np.uint8)
    # Crop only empty background before component analysis; foreground pixels are identical.
    offset_x, offset_y, width, height = cv2.boundingRect(mask)
    mask = mask[offset_y:offset_y + height, offset_x:offset_x + width]
    scores = plane[offset_y:offset_y + height, offset_x:offset_x + width]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return []
    sums = np.bincount(labels.ravel(), weights=scores.ravel(), minlength=count)
    result = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        x, y = x + offset_x, y + offset_y
        if area < min_area:
            continue
        detection = Detection(float(sums[index] / area / 255), kind,
                              (float(x), float(y), float(x + w), float(y + h)))
        if geometry_valid(detection):
            result.append(detection)
    return sorted(result, key=lambda item: item.score, reverse=True)[:300]


def maps_to_detections(maps, selected):
    candidates = maps_to_candidates(maps, selected)
    return postprocess(candidates, selected)


def maps_to_candidates(maps, selected):
    candidates = []
    for index, kind in enumerate(KINDS):
        candidates.extend(components(maps[index], kind, selected["pixel_thresholds"][kind],
                                     selected["min_areas"][kind]))
    return candidates


class NativeEngine:
    def __init__(self, kind, weights: Path, device="cpu"):
        import torch
        self.kind, self.device = kind, device
        torch.set_num_threads(4)
        if kind == "unet":
            from .pixel_model import load_unet
            self.model = load_unet(weights, device)
        elif kind == "yolo":
            from .oled_yolo import load_model
            self.model = load_model(weights)
        else:
            raise ValueError("Unknown inspection model")

    def predict_tiles(self, images):
        shapes = [image.shape[:2] for image in images]
        batch = [padded(image) for image in images]
        if self.kind == "unet":
            from .pixel_model import predict_maps
            maps = predict_maps(self.model, batch, self.device)
            # Below the smallest calibrated pixel threshold, values cannot affect detections.
            maps[maps < 51] = 0
            return [value[:, :h, :w] for value, (h, w) in zip(maps, shapes, strict=True)]
        from .oled_yolo import OledPredictor, detections_from_result
        predictions = self.model.predict(batch, predictor=OledPredictor, imgsz=TILE, rect=False,
                                         conf=.001, iou=.7, device=self.device, quantize=32, verbose=False)
        result = []
        for prediction, (h, w) in zip(predictions, shapes, strict=True):
            detections = []
            for item in detections_from_result(prediction):
                box = (max(0., item.box[0]), max(0., item.box[1]), min(float(w), item.box[2]), min(float(h), item.box[3]))
                value = Detection(item.score, item.kind, box)
                if geometry_valid(value):
                    detections.append(value)
            result.append(detections)
        return result

    def predict_provider(self, provider, width, height, selected, roi=None, progress=None,
                         batch_size=1, raw=False, cache=None):
        """The provider only supplies image pixels. Optional cache keys are actual pixel hashes."""
        import hashlib
        windows = tile_windows(width, height, roi)
        maps = np.zeros((5, height, width), np.uint8) if self.kind == "unet" else None
        candidates = []
        for start in range(0, len(windows), batch_size):
            items = windows[start:start + batch_size]
            crops = [provider(window) for window in items]
            outputs = []
            for crop in crops:
                key = hashlib.sha256(crop.tobytes()).digest() if cache is not None else None
                if cache is not None and key in cache:
                    output = cache[key]
                else:
                    output = self.predict_tiles([crop])[0]
                    if cache is not None:
                        cache[key] = output
                outputs.append(output)
            for (x1, y1, x2, y2), output in zip(items, outputs, strict=True):
                if maps is not None:
                    np.maximum(maps[:, y1:y2, x1:x2], output, out=maps[:, y1:y2, x1:x2])
                else:
                    candidates.extend(Detection(item.score, item.kind, (item.box[0] + x1, item.box[1] + y1,
                                                                         item.box[2] + x1, item.box[3] + y1)) for item in output)
            if progress:
                progress(min(start + len(items), len(windows)), len(windows))
        if maps is not None:
            candidates = maps_to_candidates(maps, selected)
        detections = candidates if raw else postprocess(candidates, selected)
        return detections, maps

    def predict_image(self, image, selected, roi=None, progress=None, batch_size=1, raw=False):
        height, width = image.shape[:2]
        return self.predict_provider(lambda box: image[box[1]:box[3], box[0]:box[2]],
                                     width, height, selected, roi, progress, batch_size, raw)
