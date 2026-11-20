"""Select OLED thresholds on validation images and compare runs on a fixed test split."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from display_defect_ai.line_postprocess import Detection, box_iou
from display_defect_ai.oled_defects import KINDS, LINE_KINDS, OledRuleSettings, detect_oled_white_screen

MATCH_IOU = .50
CONFIDENCE_CANDIDATES = (.001, .002, .004, .006, .008, .01, .02, .04, .06, .08, .12, .15, .20, .25, .35, .45)


def rows_for(dataset, split):
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(row["source_case_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Source-panel leakage between splits")
    result = [row for row in rows if row["split"] == split]
    if not result or any(row["pattern"] != "white" for row in result):
        raise ValueError("Expected white-screen images")
    return result


def read_predictions(path, rows, dataset, *, size=None, line_sizes=None):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["image_id"] for row in data] != [row["image_id"] for row in rows]:
        raise ValueError("Predictions and evaluation image IDs differ")
    weights_sha = hashlib.sha256((path.parent / "train/weights/best.pt").read_bytes()).hexdigest()
    dataset_sha = hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest()
    if any(row["weights_sha256"] != weights_sha or row["dataset_manifest_sha256"] != dataset_sha for row in data):
        raise ValueError("Saved predictions use different weights or dataset contents")
    sizes = {row.get("image_size", 640) for row in data}
    if len(sizes) != 1 or (size is not None and sizes != {size}):
        raise ValueError("Saved predictions use a different inference resolution")
    floors = {row.get("minimum_confidence", .01) for row in data}
    if len(floors) != 1 or any(not 0 < value <= 1 for value in floors):
        raise ValueError("Saved predictions have inconsistent confidence floors")
    scales = {tuple(row.get("line_sizes", [])) for row in data}
    if len(scales) != 1 or (line_sizes is not None and scales != {tuple(line_sizes)}):
        raise ValueError("Saved predictions use a different multi-resolution line pipeline")
    return [[Detection(item["score"], item["kind"], tuple(item["box"])) for item in row["detections"]]
            for row in data]


def collect(dataset, output, split, device, size=640, line_sizes=()):
    import torch
    from ultralytics import settings
    from display_defect_ai.oled_yolo import CANDIDATE_CONFIDENCE, load_model, predict_candidates

    path = output / f"raw_{split}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    rows = rows_for(dataset, split)
    settings.update({"sync": False})
    torch.set_num_threads(4)
    weights = output / "train/weights/best.pt"
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    dataset_sha = hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest()
    model = load_model(weights)
    with tempfile.TemporaryDirectory(prefix=f".collect-{split}-", dir=output) as temporary:
        pending = Path(temporary) / path.name
        with pending.open("w", encoding="utf-8") as stream:
            # Lists of PIL images are treated as one batch by Ultralytics, so bound each call explicitly.
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                results = predict_candidates(
                    model, [str((dataset / row["image"]).resolve()) for row in batch], size, line_sizes, device=device,
                )
                for row, (image_path, detections) in zip(batch, results, strict=True):
                    if Path(image_path).stem != row["image_id"]:
                        raise ValueError("YOLO reordered the evaluation images")
                    stream.write(json.dumps(dict(image_id=row["image_id"], weights_sha256=weights_sha,
                                                  dataset_manifest_sha256=dataset_sha, image_size=size,
                                                  line_sizes=list(line_sizes), minimum_confidence=CANDIDATE_CONFIDENCE,
                                                  detections=[item._asdict() for item in detections])) + "\n")
        if (weights_sha != hashlib.sha256(weights.read_bytes()).hexdigest()
                or dataset_sha != hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest()):
            raise ValueError("Model or dataset changed during prediction collection")
        if path.exists():
            raise FileExistsError(path)
        pending.rename(path)
    print(f"Saved {len(rows)} {split} predictions", flush=True)


def summarize(rows, predictions):
    counters = {kind: Counter(tp=0, fp=0, fn=0, duplicates=0) for kind in KINDS}
    normal_total = normal_alarms = defect_images = missed_images = thin_total = thin_hits = 0
    mixed_total = mixed_complete = 0
    matched_ious = []
    localization_confusion = {kind: Counter() for kind in KINDS}
    by_form = {form: Counter(total=0, hit=0) for form in ("full", "partial", "broken")}
    for row, detections in zip(rows, predictions, strict=True):
        objects = row["objects"]
        if objects:
            defect_images += 1
            missed_images += not bool(detections)
        else:
            normal_total += 1
            normal_alarms += bool(detections)
        claimed = set()
        for item in sorted(detections, key=lambda detection: detection.score, reverse=True):
            candidates = [(box_iou(item.box, obj["bbox_xyxy"]), index) for index, obj in enumerate(objects)
                          if obj["kind"] == item.kind]
            available = [(overlap, index) for overlap, index in candidates if index not in claimed]
            overlap, index = max(available, default=(0, -1))
            if overlap >= MATCH_IOU:
                counters[item.kind]["tp"] += 1
                claimed.add(index)
                matched_ious.append(overlap)
            else:
                counters[item.kind]["fp"] += 1
                counters[item.kind]["duplicates"] += any(overlap >= MATCH_IOU for overlap, _ in candidates)
        for index, obj in enumerate(objects):
            hit = index in claimed
            counters[obj["kind"]]["fn"] += not hit
            candidates = [(box_iou(item.box, obj["bbox_xyxy"]), item.kind) for item in detections]
            overlap, predicted_kind = max(candidates, default=(0, "missed"))
            localization_confusion[obj["kind"]][predicted_kind if overlap >= MATCH_IOU else "missed"] += 1
            if obj["kind"] in LINE_KINDS:
                attributes = obj["attributes"]
                form = by_form[attributes["line_form"]]
                form["total"] += 1
                form["hit"] += hit
                if attributes["thickness"] == 1:
                    thin_total += 1
                    thin_hits += hit
        if len(objects) > 1:
            mixed_total += 1
            mixed_complete += len(claimed) == len(objects)
    classes = {}
    for kind, counts in counters.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        classes[kind] = dict(counts, total=tp + fn, precision=tp / (tp + fp) if tp + fp else 0,
                             recall=tp / (tp + fn) if tp + fn else 0,
                             f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return dict(
        classes=classes, normal_images=normal_total, normal_false_alarms=normal_alarms,
        defect_images=defect_images, defect_image_misses=missed_images,
        one_pixel_lines=dict(total=thin_total, hit=thin_hits), line_forms=by_form,
        multiple_defect_images=dict(total=mixed_total, all_objects_found=mixed_complete),
        localization_confusion=localization_confusion,
        mean_matched_iou=float(np.mean(matched_ious)) if matched_ious else 0,
    )


def rank(metrics):
    return (-metrics["normal_false_alarms"], sum(c["f1"] for c in metrics["classes"].values()),
            metrics["one_pixel_lines"]["hit"], metrics["mean_matched_iou"])


def calibrate(dataset, output):
    from display_defect_ai.oled_yolo import select_detections

    rows = rows_for(dataset, "val")
    predictions = read_predictions(output / "raw_val.jsonl", rows, dataset)
    with (output / "raw_val.jsonl").open() as stream:
        context = json.loads(stream.readline())
    size = context.get("image_size", 640)
    minimum_confidence = context.get("minimum_confidence", .01)
    confidence_candidates = [value for value in CONFIDENCE_CANDIDATES if value >= minimum_confidence]
    if not confidence_candidates:
        raise ValueError("Prediction cache does not include the calibration confidence range")
    selected = dict(size=size, thresholds={kind: .25 for kind in KINDS}, line_nms_iou=.25)
    if context.get("line_sizes"):
        selected["line_sizes"] = context["line_sizes"]
    # Class thresholds and suppression are selected using validation data only.
    for _ in range(2):
        for kind in KINDS:
            candidates = []
            for threshold in confidence_candidates:
                config = dict(selected, thresholds={**selected["thresholds"], kind: threshold})
                metrics = summarize(rows, [select_detections(items, config) for items in predictions])
                candidates.append((rank(metrics), threshold))
            selected["thresholds"][kind] = max(candidates)[1]
    candidates = []
    for nms_iou in (.20, .25, .35, .50):
        config = dict(selected, line_nms_iou=nms_iou)
        metrics = summarize(rows, [select_detections(items, config) for items in predictions])
        candidates.append(dict(settings=config, metrics=metrics))
    yolo = max(candidates, key=lambda candidate: rank(candidate["metrics"]))
    rule_candidates = []
    for threshold in (7, 9, 12):
        config = OledRuleSettings(min_contrast=threshold)
        detections = [detect_oled_white_screen(cv2.imread(str(dataset / row["image"])), config) for row in rows]
        rule_candidates.append(dict(settings=asdict(config), metrics=summarize(rows, detections)))
    opencv = max(rule_candidates, key=lambda candidate: rank(candidate["metrics"]))
    result = dict(
        schema_version="oled-white-2", split="val", images=len(rows), minimum_iou=MATCH_IOU,
        dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256((output / "train/weights/best.pt").read_bytes()).hexdigest(),
        classes=list(KINDS), selected=yolo["settings"], opencv=opencv["settings"],
        confidence_candidates=confidence_candidates,
        yolo_metrics=yolo["metrics"], opencv_metrics=opencv["metrics"],
        nms_candidates=candidates, rule_candidates=rule_candidates,
    )
    (output / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected", "opencv", "yolo_metrics", "opencv_metrics")}, indent=2))


def evaluate(dataset, output):
    from display_defect_ai.oled_yolo import select_detections

    rows = rows_for(dataset, "test")
    calibration = json.loads((output / "validation_selection.json").read_text())
    if calibration["dataset_manifest_sha256"] != hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest():
        raise ValueError("Dataset differs from the calibration dataset")
    weights = output / "train/weights/best.pt"
    if calibration["weights_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest():
        raise ValueError("Weights differ from the calibrated model")
    predictions = read_predictions(output / "raw_test.jsonl", rows, dataset,
                                   size=calibration["selected"]["size"],
                                   line_sizes=calibration["selected"].get("line_sizes", []))
    yolo = [select_detections(items, calibration["selected"]) for items in predictions]
    rules = OledRuleSettings(**calibration["opencv"])
    opencv = [detect_oled_white_screen(cv2.imread(str(dataset / row["image"])), rules) for row in rows]
    fused_normal_fp = sum(bool(a or b) for row, a, b in zip(rows, opencv, yolo, strict=True) if not row["objects"])
    fused_misses = sum(not bool(a or b) for row, a, b in zip(rows, opencv, yolo, strict=True) if row["objects"])
    report = dict(
        schema_version="oled-white-2", split="test", images=len(rows), minimum_iou=MATCH_IOU,
        selected=calibration["selected"], opencv_settings=calibration["opencv"],
        weights_sha256=calibration["weights_sha256"], opencv=summarize(rows, opencv), yolo=summarize(rows, yolo),
        dataset_manifest_sha256=calibration["dataset_manifest_sha256"],
        combined_decision=dict(rule="defect if either detector reports an object",
                               normal_false_alarms=fused_normal_fp, defect_image_misses=fused_misses),
    )
    (output / "test_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, a, b in zip(rows, opencv, yolo, strict=True):
            stream.write(json.dumps(dict(image_id=row["image_id"], opencv=[d._asdict() for d in a],
                                         yolo=[d._asdict() for d in b])) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "calibrate", "test"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, choices=(640, 768, 960, 1280), default=640,
                        help="Inference resolution recorded with the prediction cache")
    parser.add_argument("--line-sizes", type=int, nargs="*", choices=(640, 768, 960, 1280), default=[],
                        help="Additional YOLO resolutions used for line candidates only")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.action == "collect":
        collect(args.dataset, args.output, args.split, args.device, args.size, args.line_sizes)
    elif args.action == "calibrate":
        calibrate(args.dataset, args.output)
    else:
        evaluate(args.dataset, args.output)
