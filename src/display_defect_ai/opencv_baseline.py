"""Detect dark point and axis-aligned line defects in a fixed white-screen capture."""

from dataclasses import dataclass

import cv2
import numpy as np

from .line_postprocess import Detection


# The fixed 512px setup leaves a dark frame around the panel. Stay inside the
# transition at its edges so the frame cannot become a detected line.
ROI_START = 31
ROI_END = 481
FRAME_SIZE = 512


@dataclass(frozen=True)
class RuleSettings:
    min_contrast: int
    dot_margin: int
    line_margin: int


def white_screen_contrast(image: np.ndarray) -> np.ndarray:
    """Estimate per-pixel darkening from this image's local BGR background."""
    if image.shape != (FRAME_SIZE, FRAME_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("Expected one 512×512 BGR image with 8-bit channels")

    panel = image[ROI_START:ROI_END, ROI_START:ROI_END]
    radius = 15
    padded = cv2.copyMakeBorder(panel, radius, radius, radius, radius, cv2.BORDER_REFLECT_101)
    background = cv2.medianBlur(padded, radius * 2 + 1)[radius:-radius, radius:-radius]
    blue, green, red = cv2.split(cv2.subtract(background, panel))
    return cv2.max(cv2.max(blue, green), red)


def _join_line_segments(segments: list[Detection], axis: int) -> list[Detection]:
    joined: list[Detection] = []
    cross_axis = 1 - axis
    for segment in sorted(segments, key=lambda detection: detection.box[axis]):
        box = segment.box
        for index, previous in enumerate(joined):
            other = previous.box
            center_shift = abs(box[cross_axis] + box[cross_axis + 2]
                               - other[cross_axis] - other[cross_axis + 2]) / 2
            gap = box[axis] - other[axis + 2]
            if center_shift <= 2 and gap <= 64:
                joined[index] = Detection(max(segment.score, previous.score), "line", (
                    min(box[0], other[0]), min(box[1], other[1]),
                    max(box[2], other[2]), max(box[3], other[3]),
                ))
                break
        else:
            joined.append(segment)
    return joined


def find_defect_regions(contrast: np.ndarray, min_contrast: int, *, include_regions=False) -> list[Detection]:
    """Threshold a contrast image, classify connected regions and join collinear pieces."""
    if not 1 <= min_contrast <= 255:
        raise ValueError("min_contrast must be in [1, 255]")
    mask = cv2.compare(contrast, min_contrast, cv2.CMP_GE)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots: list[Detection] = []
    horizontal: list[Detection] = []
    vertical: list[Detection] = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area < 3:
            continue
        pixels = contrast[y:y + h, x:x + w][labels[y:y + h, x:x + w] == index]
        score = float(pixels.max())
        box = (int(x), int(y), int(x + w), int(y + h))
        if w >= 10 and w >= 3 * h:
            horizontal.append(Detection(score, "line", box))
        elif h >= 10 and h >= 3 * w:
            vertical.append(Detection(score, "line", box))
        elif max(w, h) < 24:
            dots.append(Detection(score, "dot", box))
        elif include_regions:
            dots.append(Detection(score, "dark_region", box))
    return dots + _join_line_segments(horizontal, axis=0) + _join_line_segments(vertical, axis=1)


def put_boxes_in_frame(regions: list[Detection], settings: RuleSettings) -> list[Detection]:
    """Return original-image coordinates with room for the faint edges of the defect."""
    detections = []
    for region in regions:
        margin = settings.dot_margin if region.kind == "dot" else settings.line_margin
        x1, y1, x2, y2 = region.box
        box = (
            max(0, x1 + ROI_START - margin), max(0, y1 + ROI_START - margin),
            min(FRAME_SIZE, x2 + ROI_START + margin), min(FRAME_SIZE, y2 + ROI_START + margin),
        )
        detections.append(Detection(region.score, region.kind, box))
    return sorted(detections, key=lambda detection: detection.score, reverse=True)


def detect_white_screen(image: np.ndarray, settings: RuleSettings) -> list[Detection]:
    """Detect points and lines from a single image, with no paired normal image."""
    contrast = white_screen_contrast(image)
    return put_boxes_in_frame(find_defect_regions(contrast, settings.min_contrast), settings)
