"""Shared YOLO inference and validated postprocessing for the OLED app and evaluation."""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms

from .line_postprocess import Detection, box_iou, suppress_duplicate_lines
from .oled_defects import KINDS, LINE_KINDS

CANDIDATE_CONFIDENCE = .001


class OledPredictor(DetectionPredictor):
    def postprocess(self, predictions, image, originals, **kwargs):
        boxes = nms.non_max_suppression(
            predictions, self.args.conf, self.args.iou, multi_label=True,
            max_det=self.args.max_det, end2end=self.model.end2end,
        )
        return self.construct_results(boxes, image, originals, **kwargs)


def load_model(weights: Path):
    model = YOLO(weights)
    if model.names != dict(enumerate(KINDS)):
        raise ValueError("Weights do not match the five OLED defect classes")
    return model


def detections_from_result(result) -> list[Detection]:
    return [Detection(float(score), result.names[int(class_id)], tuple(box))
            for box, score, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                           result.boxes.cls.tolist(), strict=True)]


def select_detections(detections, selected) -> list[Detection]:
    thresholds = selected["thresholds"]
    if set(thresholds) != set(KINDS) or any(not 0 < score <= 1 for score in thresholds.values()):
        raise ValueError("Missing or invalid class confidence thresholds")
    filtered = [item for item in detections if item.score >= thresholds[item.kind]]
    kept = []
    for item in suppress_duplicate_lines(filtered, selected["line_nms_iou"]):
        # A single point can receive both point-class labels under multi-label NMS.
        if item.kind in ("dark_dot", "color_dot") and any(
            other.kind in ("dark_dot", "color_dot") and box_iou(item.box, other.box) >= .5 for other in kept
        ):
            continue
        kept.append(item)
    return kept


def fuse_line_scales(primary, additional) -> list[Detection]:
    """Use collinear primary YOLO candidates to complete higher-resolution line boxes."""
    combined = list(primary)
    for item in additional:
        if item.kind not in LINE_KINDS:
            continue
        axis = 0 if item.kind == "horizontal_line" else 1
        cross = 1 - axis
        box = item.box
        length, thickness = box[axis + 2] - box[axis], box[cross + 2] - box[cross]
        if length <= 0 or thickness <= 0:
            continue
        supports = []
        for other in primary:
            if other.kind != item.kind:
                continue
            reference = other.box
            reference_length = reference[axis + 2] - reference[axis]
            reference_thickness = reference[cross + 2] - reference[cross]
            if reference_length <= 0 or reference_thickness <= 0:
                continue
            shift = abs(box[cross] + box[cross + 2] - reference[cross] - reference[cross + 2]) / 2
            overlap = min(box[axis + 2], reference[axis + 2]) - max(box[axis], reference[axis])
            if (shift <= max(1.5, min(thickness, reference_thickness) / 2)
                    and max(thickness, reference_thickness) <= 2.5 * min(thickness, reference_thickness)
                    and overlap >= .65 * min(length, reference_length)):
                supports.append(other)
        if supports:
            support = max(supports, key=lambda other: box_iou(box, other.box))
            # Keep thickness from the more confident model box and combine its longitudinal extent.
            completed = list(support.box if support.score >= item.score else box)
            completed[axis] = min(box[axis], support.box[axis])
            completed[axis + 2] = max(box[axis + 2], support.box[axis + 2])
            combined.append(Detection(max(item.score, support.score), item.kind, tuple(completed)))
        else:
            combined.append(item)
    return combined


def predict_candidates(model, images, size, line_sizes=(), *, device="cpu"):
    """Share the exact multi-resolution path between cached evaluation and single-image inspection."""
    def forward(resolution):
        return model.predict(
            images, predictor=OledPredictor, device=device, imgsz=resolution, rect=False,
            conf=CANDIDATE_CONFIDENCE, iou=.7, verbose=False,
        )

    results = forward(size)
    primary = [detections_from_result(result) for result in results]
    additional = [[] for _ in results]
    for resolution in line_sizes:
        for index, (original, result) in enumerate(zip(results, forward(resolution), strict=True)):
            if original.path != result.path:
                raise ValueError("Multi-resolution inference reordered the images")
            additional[index].extend(item for item in detections_from_result(result) if item.kind in LINE_KINDS)
    return [(result.path, fuse_line_scales(base, extra)) for result, base, extra
            in zip(results, primary, additional, strict=True)]


def predict_image(model, image, selected) -> list[Detection]:
    _, candidates = predict_candidates(model, image, selected["size"], selected.get("line_sizes", ()))[0]
    return select_detections(candidates, selected)
