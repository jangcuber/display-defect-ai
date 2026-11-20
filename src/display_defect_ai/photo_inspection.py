"""Prepare a white-panel photo and map detections back to the original image."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .line_postprocess import Detection
from .opencv_baseline import FRAME_SIZE, ROI_END, ROI_START

_MARGIN = FRAME_SIZE * .055
_EDGE = FRAME_SIZE - 1 - _MARGIN
_PANEL_CORNERS = np.float32([
    [_MARGIN, _MARGIN], [_EDGE, _MARGIN], [_EDGE, _EDGE], [_MARGIN, _EDGE],
])


class InvalidPhoto(ValueError):
    """The input could not be prepared for this inspector."""


@dataclass(frozen=True)
class PanelPhoto:
    original: np.ndarray
    aligned: np.ndarray
    to_original: np.ndarray


def _load_photo(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG"):
                raise InvalidPhoto("PNG 또는 JPEG 사진을 선택해 주세요.")
            if min(source.size) < 128 or source.width * source.height > 48_000_000:
                raise InvalidPhoto("한 변이 128픽셀 이상이고 4,800만 픽셀 이하인 사진을 선택해 주세요.")
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise InvalidPhoto("사진을 열 수 없습니다. PNG 또는 JPEG 파일을 다시 선택해 주세요.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _order_corners(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    ordered = points[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def _find_panel(photo: np.ndarray) -> np.ndarray:
    height, width = photo.shape[:2]
    scale = min(1.0, 1200 / max(width, height))
    thumb = cv2.resize(photo, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.compare(smooth, max(135, int(threshold)), cv2.CMP_GE)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_floor = gray.size * .12
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < area_floor:
            break
        hull = cv2.convexHull(contour)
        if cv2.contourArea(contour) < .88 * cv2.contourArea(hull):
            continue
        perimeter = cv2.arcLength(hull, True)
        for fraction in (.01, .02, .035):
            quad = cv2.approxPolyDP(hull, perimeter * fraction, True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            if cv2.contourArea(quad) < area_floor:
                continue
            points = _order_corners(quad.reshape(4, 2).astype(np.float32))
            sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
            if sides.min() < max(32, sides.max() * .18):
                continue
            return points * np.float32([width / gray.shape[1], height / gray.shape[0]])
    raise InvalidPhoto("흰 화면의 테두리를 찾지 못했습니다. 화면 전체가 보이는 정면 사진을 선택해 주세요.")


def prepare_panel(path: Path) -> PanelPhoto:
    original = _load_photo(path)
    corners = _find_panel(original)
    if original.shape[:2] == (FRAME_SIZE, FRAME_SIZE) and np.max(np.abs(corners - _PANEL_CORNERS)) <= 3:
        # Preserve the original pixels of captures already using the benchmark framing.
        aligned = original
        to_original = np.eye(3)
    else:
        to_panel = cv2.getPerspectiveTransform(corners, _PANEL_CORNERS)
        if not np.isfinite(to_panel).all() or abs(np.linalg.det(to_panel)) < 1e-10:
            raise InvalidPhoto("패널 모서리를 정렬할 수 없습니다. 정면 사진을 선택해 주세요.")
        aligned = cv2.warpPerspective(
            original, to_panel, (FRAME_SIZE, FRAME_SIZE), borderValue=(24, 24, 24),
        )
        to_original = np.linalg.inv(to_panel)
    inner = aligned[ROI_START:ROI_END, ROI_START:ROI_END]
    if np.median(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)) < 145:
        raise InvalidPhoto("흰 화면이 너무 어둡습니다. 화면 밝기와 촬영 조건을 확인해 주세요.")
    return PanelPhoto(original, aligned, to_original)


def project_detections(photo: PanelPhoto, detections: list[Detection]) -> list[dict]:
    height, width = photo.original.shape[:2]
    projected = []
    for item in detections:
        x1, y1, x2, y2 = item.box
        corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        original = cv2.perspectiveTransform(corners, photo.to_original)[0]
        polygon = np.clip(original / [width, height], 0, 1).tolist()
        projected.append(dict(kind=item.kind, score=item.score, polygon=polygon))
    return projected
