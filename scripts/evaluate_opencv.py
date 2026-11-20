"""Calibrate the fixed white-screen rule detector and compare it with YOLO."""

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from display_defect_ai.line_postprocess import Detection, box_iou
from display_defect_ai.opencv_baseline import (
    RuleSettings,
    detect_white_screen,
    find_defect_regions,
    put_boxes_in_frame,
    white_screen_contrast,
)


KINDS = ("dot", "line")
MATCH_IOU = 0.50


def read_rows(dataset: Path, split: str) -> list[dict]:
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as stream:
        rows = [row for row in map(json.loads, stream) if row["split"] == split]
    if Counter(row["kind"] for row in rows) != {"normal": 150, "dot": 150, "line": 150}:
        raise ValueError(f"Unexpected {split} split counts")
    if any(row["pattern"] != "white" or len(row["bbox_xyxy"]) != int(row["kind"] != "normal") for row in rows):
        raise ValueError("Expected white-screen images with zero or one labeled defect")
    return rows


def image_for(dataset: Path, row: dict):
    image = cv2.imread(str(dataset / row["image"]))
    if image is None:
        raise ValueError(f"Cannot decode {row['image']}")
    return image


def summarize(rows: list[dict], predictions: list[list[Detection]]) -> dict:
    counts = {kind: Counter(tp=0, fp=0, fn=0, duplicates=0) for kind in KINDS}
    thin_total = thin_hits = normal_total = normal_false_alarms = 0
    matched_ious = []
    for row, detections in zip(rows, predictions, strict=True):
        kind = row["kind"]
        if kind == "normal":
            normal_total += 1
            normal_false_alarms += bool(detections)
        hit = False
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            overlap = box_iou(detection.box, row["bbox_xyxy"][0]) if detection.kind == kind else 0.0
            if overlap >= MATCH_IOU and not hit:
                counts[kind]["tp"] += 1
                matched_ious.append(overlap)
                hit = True
            else:
                counts[detection.kind]["fp"] += 1
                counts[detection.kind]["duplicates"] += overlap >= MATCH_IOU
        if kind in KINDS:
            counts[kind]["fn"] += not hit
        if kind == "line" and row["defect"]["thickness"] == 1:
            thin_total += 1
            thin_hits += hit

    classes = {}
    for kind, counter in counts.items():
        tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
        classes[kind] = dict(
            counter, precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn),
            f1=2 * tp / (2 * tp + fp + fn),
        )
    return dict(
        classes=classes, one_pixel_line=dict(hit=thin_hits, total=thin_total),
        normal_images=normal_total, normal_false_alarms=normal_false_alarms,
        mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
    )


def calibrate(dataset: Path, output: Path):
    rows = read_rows(dataset, "val")
    thresholds = (6, 7, 8, 9, 10, 12, 14)
    regions = {threshold: [] for threshold in thresholds}
    for row in rows:
        contrast = white_screen_contrast(image_for(dataset, row))
        for threshold in thresholds:
            regions[threshold].append(find_defect_regions(contrast, threshold))

    candidates = []
    for threshold, dot_margin, line_margin in product(thresholds, range(3), range(3)):
        settings = RuleSettings(threshold, dot_margin, line_margin)
        detections = [put_boxes_in_frame(items, settings) for items in regions[threshold]]
        candidates.append(dict(settings=asdict(settings), metrics=summarize(rows, detections)))

    def rank(candidate):
        metrics = candidate["metrics"]
        return (
            -metrics["normal_false_alarms"],
            sum(metrics["classes"][kind]["f1"] for kind in KINDS),
            metrics["one_pixel_line"]["hit"],
            metrics["mean_matched_iou"],
        )

    selected = max(candidates, key=rank)
    result = dict(split="val", minimum_iou=MATCH_IOU, selected=selected, candidates=candidates)
    (output / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))


def yolo_predictions(yolo_run: Path, rows: list[dict]) -> tuple[list[list[Detection]], dict]:
    report = json.loads((yolo_run / "test_comparison.json").read_text())
    if report["split"] != "test" or report["image_count"] != len(rows) or report["minimum_iou"] != MATCH_IOU:
        raise ValueError("The YOLO report uses a different evaluation split")
    thresholds = (report["confidence"], report["selected"]["line_confidence"])
    labels = yolo_run / "test-improved-native" / "labels"
    if not labels.is_dir():
        raise FileNotFoundError(labels)
    test_ids = {row["image_id"] for row in rows}
    if any(path.stem not in test_ids for path in labels.glob("*.txt")):
        raise ValueError("The YOLO predictions include images outside this test split")
    predictions = []
    for row in rows:
        label = labels / f"{row['image_id']}.txt"
        detections = []
        if label.exists():
            for line in label.read_text().splitlines():
                class_id, x, y, w, h, score = map(float, line.split())
                kind = int(class_id)
                if score >= thresholds[kind]:
                    box = tuple(coordinate * 512 for coordinate in (x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    detections.append(Detection(score, KINDS[kind], box))
        predictions.append(detections)
    return predictions, report


def compare(dataset: Path, output: Path, yolo_run: Path):
    calibration = json.loads((output / "validation_selection.json").read_text())
    settings = RuleSettings(**calibration["selected"]["settings"])
    rows = read_rows(dataset, "test")
    detections = [detect_white_screen(image_for(dataset, row), settings) for row in rows]
    previous_predictions, previous_report = yolo_predictions(yolo_run, rows)
    previous = summarize(rows, previous_predictions)
    for kind in KINDS:
        for metric in ("tp", "fp", "fn", "duplicates"):
            if previous["classes"][kind][metric] != previous_report["improved"]["classes"][kind][metric]:
                raise ValueError("YOLO counts do not match its saved test report")
    if previous["normal_false_alarms"] != previous_report["improved"]["normal_false_alarms"]:
        raise ValueError("YOLO normal false alarms do not match its saved test report")

    result = dict(
        split="test", images=len(rows), minimum_iou=MATCH_IOU, opencv_version=cv2.__version__,
        opencv_settings=asdict(settings), opencv=summarize(rows, detections),
        yolo_settings=previous_report["selected"], yolo=previous,
    )
    (output / "test_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, rules, yolo in zip(rows, detections, previous_predictions, strict=True):
            stream.write(json.dumps(dict(image_id=row["image_id"], opencv=[d._asdict() for d in rules],
                                         yolo=[d._asdict() for d in yolo])) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("calibrate", "compare"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-run", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "calibrate":
        calibrate(args.dataset, args.output)
    elif args.yolo_run is None:
        parser.error("--yolo-run is required for compare")
    else:
        compare(args.dataset, args.output, args.yolo_run)
