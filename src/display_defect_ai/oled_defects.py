"""Visual defect categories for the white-screen OLED inspection experiment."""

from dataclasses import dataclass

import cv2
import numpy as np

from .line_postprocess import Detection
from .opencv_baseline import FRAME_SIZE, ROI_END, ROI_START, RuleSettings, find_defect_regions, put_boxes_in_frame

KINDS = ("dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region")
TITLES = {
    "dark_dot": "어두운 점",
    "color_dot": "색 이상점",
    "horizontal_line": "가로선",
    "vertical_line": "세로선",
    "dark_region": "어두운 영역",
}
LINE_KINDS = frozenset(("horizontal_line", "vertical_line"))


@dataclass(frozen=True)
class OledRuleSettings:
    min_contrast: int = 9
    margin: int = 1
    color_fraction: float = .35

    def __post_init__(self):
        if not 1 <= self.min_contrast <= 255 or not 0 <= self.margin <= 3 or not 0 < self.color_fraction < 1:
            raise ValueError("Invalid OLED rule settings")


def background_deficit(image: np.ndarray) -> np.ndarray:
    """Fit the nearly planar white background robustly, including around large dark regions."""
    if image.shape != (FRAME_SIZE, FRAME_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("Expected a 512×512 uint8 BGR image")
    panel = image[ROI_START:ROI_END, ROI_START:ROI_END].astype(np.float32)
    height, width = panel.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    design = np.stack((np.ones_like(xx), xx / width, yy / height), axis=-1)
    sample_design = design[4::8, 4::8].reshape(-1, 3)
    sample_pixels = panel[4::8, 4::8].reshape(-1, 3)
    # Start with the brighter majority so a very dark patch cannot tilt the first fit.
    brightness = sample_pixels.min(axis=1)
    selected = brightness >= np.percentile(brightness, 30)
    for _ in range(4):
        fit = np.linalg.lstsq(sample_design[selected], sample_pixels[selected], rcond=None)[0]
        residual = np.abs(sample_pixels - sample_design @ fit).max(axis=1)
        selected = residual < 6
        if np.count_nonzero(selected) < len(selected) // 2:
            raise ValueError("The white background is insufficient for OLED rule inspection")
    background = np.clip(design @ fit, 0, 255)
    deficit = np.zeros_like(image, dtype=np.float32)
    deficit[ROI_START:ROI_END, ROI_START:ROI_END] = np.maximum(background - panel, 0)
    return deficit


def _appearance(deficit, box, fraction):
    x1, y1 = (int(np.floor(value)) for value in box[:2])
    x2, y2 = (int(np.ceil(value)) for value in box[2:])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(FRAME_SIZE, x2), min(FRAME_SIZE, y2)
    patch = deficit[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("Detection has an empty box")
    peak = float(patch.max())
    foreground = patch.max(axis=2) >= max(6, peak * .45)
    loss = np.median(patch[foreground], axis=0) if foreground.any() else np.zeros(3)
    colored = float(loss.max() - loss.min()) >= max(5, float(loss.max()) * fraction)
    # These are observed complementary colors on a white screen, not inferred faulty circuits.
    color = ("노랑 계열", "자홍 계열", "청록 계열")[int(loss.argmax())] if colored else "검정·회색 계열"
    return colored, color, peak


def detect_oled_white_screen(image: np.ndarray, settings: OledRuleSettings) -> list[Detection]:
    deficit = background_deficit(image)
    contrast = np.rint(deficit.max(axis=2)).astype(np.uint8)[ROI_START:ROI_END, ROI_START:ROI_END]
    regions = find_defect_regions(contrast, settings.min_contrast, include_regions=True)
    boxes = put_boxes_in_frame(regions, RuleSettings(settings.min_contrast, settings.margin, settings.margin))
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box.box
        kind = box.kind
        if kind == "dot":
            kind = "color_dot" if _appearance(deficit, box.box, settings.color_fraction)[0] else "dark_dot"
        elif kind == "line":
            kind = "horizontal_line" if x2 - x1 >= y2 - y1 else "vertical_line"
        result.append(Detection(box.score, kind, box.box))
    return result


def describe_detections(image: np.ndarray, detections: list[Detection]) -> list[dict]:
    """Attach observable attributes to model or rule boxes without using dataset metadata."""
    deficit = background_deficit(image)
    descriptions = []
    for item in detections:
        colored, color, contrast = _appearance(deficit, item.box, .35)
        x1, y1 = (int(np.floor(value)) for value in item.box[:2])
        x2, y2 = (int(np.ceil(value)) for value in item.box[2:])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(FRAME_SIZE, x2), min(FRAME_SIZE, y2)
        width, height = x2 - x1, y2 - y1
        attributes = dict(observed_color=color, width_px=width, height_px=height,
                          peak_contrast_8bit=round(contrast, 1), coordinate_space="aligned_512")
        detail = [color]
        if item.kind in ("dark_dot", "color_dot"):
            center = np.array(((x1 + x2) / 2, (y1 + y2) / 2))
            clustered = any(
                other is not item and other.kind in ("dark_dot", "color_dot")
                and np.linalg.norm(center - np.array(((other.box[0] + other.box[2]) / 2,
                                                       (other.box[1] + other.box[3]) / 2))) <= 55
                for other in detections
            )
            attributes["clustered"] = clustered
            detail.append("군집" if clustered else "단독")
        if item.kind in LINE_KINDS:
            patch = deficit[y1:y2, x1:x2].max(axis=2)
            profile = patch.max(axis=0 if item.kind == "horizontal_line" else 1)
            active = profile >= max(7, contrast * .25)
            points = np.flatnonzero(active)
            broken = len(points) > 1 and int(np.diff(points).max()) >= 4
            axis = (x1, x2) if item.kind == "horizontal_line" else (y1, y2)
            full = axis[0] <= ROI_START + 4 and axis[1] >= ROI_END - 4
            form = "broken" if broken else "full" if full else "partial"
            attributes.update(appearance="colored" if colored else "dark", line_form=form,
                              thickness_px=height if item.kind == "horizontal_line" else width)
            detail.append({"broken": "끊긴 선", "full": "전체 길이", "partial": "부분 길이"}[form])
        detail.append(f"{width}×{height}px")
        detail.append(f"밝기 차이 {contrast:.0f}/255")
        descriptions.append(dict(title=TITLES[item.kind], detail=" · ".join(detail), attributes=attributes))
    return descriptions
