"""Suppress overlapping line detections while preserving point detections."""

from collections.abc import Iterable
from typing import NamedTuple

Box = tuple[float, float, float, float]


class Detection(NamedTuple):
    score: float
    kind: str
    box: Box


def box_iou(a: Box, b: Box) -> float:
    overlap_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = overlap_width * overlap_height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def suppress_duplicate_lines(detections: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    """Keep the highest scoring box among overlapping line boxes, in score order."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    kept = []
    line_boxes = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if detection.kind in ("line", "horizontal_line", "vertical_line"):
            if any(kind == detection.kind and box_iou(detection.box, box) >= iou_threshold
                   for kind, box in line_boxes):
                continue
            line_boxes.append((detection.kind, detection.box))
        kept.append(detection)
    return kept
