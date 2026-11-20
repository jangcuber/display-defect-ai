"""Audit a same-image OLED comparison and render newly recovered one-pixel lines."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from display_defect_ai.line_postprocess import box_iou
from display_defect_ai.oled_defects import KINDS, LINE_KINDS, TITLES
from display_defect_ai.oled_yolo import select_detections
from evaluate_oled import MATCH_IOU, read_predictions, rows_for, summarize


def matched(obj, detections):
    return any(item.kind == obj["kind"] and box_iou(item.box, obj["bbox_xyxy"]) >= MATCH_IOU for item in detections)


def read_run(dataset, run, rows):
    calibration = json.loads((run / "validation_selection.json").read_text())
    report = json.loads((run / "test_comparison.json").read_text())
    if report["split"] != "test" or report["images"] != len(rows) or report["minimum_iou"] != MATCH_IOU:
        raise ValueError("Evaluation protocols differ")
    if report["selected"] != calibration["selected"] or report["weights_sha256"] != calibration["weights_sha256"]:
        raise ValueError("Report and selected model differ")
    predictions = read_predictions(run / "raw_test.jsonl", rows, dataset, size=calibration["selected"]["size"],
                                   line_sizes=calibration["selected"].get("line_sizes", []))
    predictions = [select_detections(items, calibration["selected"]) for items in predictions]
    actual = summarize(rows, predictions)
    if actual != report["yolo"]:
        raise ValueError("Saved metrics do not reproduce from the original prediction cache")
    return predictions, report


def render_examples(dataset, output, examples):
    # Use a local font for the standalone Korean comparison image.
    fonts = (Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    font_path = next((path for path in fonts if path.exists()), None)
    font = ImageFont.truetype(str(font_path), 22) if font_path else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path else ImageFont.load_default()
    chosen = []
    for kind in KINDS:
        if kind not in LINE_KINDS:
            continue
        example = next((item for item in examples if item["object"]["kind"] == kind), None)
        if example is not None:
            chosen.append(example)
    if not chosen:
        return
    width, tile, row_height = 1080, 512, 606
    canvas = Image.new("RGB", (width, 66 + row_height * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "같은 1픽셀 선 이미지의 YOLO 검출 전후", fill="#17263d", font=font)
    for row_index, example in enumerate(chosen):
        row_top = 66 + row_index * row_height
        with Image.open(dataset / example["row"]["image"]) as source:
            original = source.convert("RGB")
        for column, key, label in ((0, "baseline", "기존"), (1, "improved", "개선")):
            x_offset = 20 + column * 540
            image = original.copy()
            painter = ImageDraw.Draw(image)
            for item in example[key]:
                if item.kind not in LINE_KINDS:
                    continue
                painter.rectangle(item.box, outline="#e77713" if item.kind == "horizontal_line" else "#2265ca", width=2)
            detected = matched(example["object"], example[key])
            status = "검출" if detected else "미검출"
            draw.text((x_offset, row_top), f"{label} · {TITLES[example['object']['kind']]} · {status}",
                      fill="#17263d", font=font)
            canvas.paste(image, (x_offset, row_top + 36))
            draw.text((x_offset, row_top + 36 + tile + 9), example["row"]["image_id"], fill="#56616e", font=small)
    canvas.save(output)


def compare(dataset, baseline, improved):
    rows = rows_for(dataset, "test")
    before, baseline_report = read_run(dataset, baseline, rows)
    after, improved_report = read_run(dataset, improved, rows)
    recovered, regressed, examples = [], [], []
    by_orientation = {kind: dict(total=0, baseline_hits=0, improved_hits=0) for kind in KINDS if kind in LINE_KINDS}
    for row, old, new in zip(rows, before, after, strict=True):
        for index, obj in enumerate(row["objects"]):
            if obj["kind"] not in LINE_KINDS or obj["attributes"]["thickness"] != 1:
                continue
            entry = dict(image_id=row["image_id"], object_index=index, kind=obj["kind"],
                         bbox_xyxy=obj["bbox_xyxy"], line_form=obj["attributes"]["line_form"])
            old_hit, new_hit = matched(obj, old), matched(obj, new)
            counts = by_orientation[obj["kind"]]
            counts["total"] += 1
            counts["baseline_hits"] += old_hit
            counts["improved_hits"] += new_hit
            if new_hit and not old_hit:
                recovered.append(entry)
                examples.append(dict(row=row, object=obj, baseline=old, improved=new))
            elif old_hit and not new_hit:
                regressed.append(entry)
    one_pixel_before = baseline_report["yolo"]["one_pixel_lines"]
    one_pixel_after = improved_report["yolo"]["one_pixel_lines"]
    if one_pixel_before["total"] != one_pixel_after["total"]:
        raise ValueError("One-pixel-line target sets differ")
    result = dict(
        split="test", images=len(rows), minimum_iou=MATCH_IOU,
        dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        test_image_ids_sha256=hashlib.sha256("\n".join(row["image_id"] for row in rows).encode()).hexdigest(),
        baseline=dict(run=str(baseline.resolve()), weights_sha256=baseline_report["weights_sha256"],
                      selected=baseline_report["selected"], metrics=baseline_report["yolo"]),
        improved=dict(run=str(improved.resolve()), weights_sha256=improved_report["weights_sha256"],
                      selected=improved_report["selected"], metrics=improved_report["yolo"]),
        one_pixel_recall_gain_pp=100 * (one_pixel_after["hit"] - one_pixel_before["hit"]) / one_pixel_before["total"],
        one_pixel_by_orientation=by_orientation,
        recovered_one_pixel_lines=recovered, regressed_one_pixel_lines=regressed,
        note="Repeated evaluation on the same controlled synthetic test images. Settings are selected on validation data.",
    )
    (improved / "improvement_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    render_examples(dataset, improved / "thin_line_examples.png", examples)
    print(json.dumps(dict(before=one_pixel_before, after=one_pixel_after, gain_pp=result["one_pixel_recall_gain_pp"],
                          recovered=len(recovered), regressed=len(regressed)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    args = parser.parse_args()
    compare(args.dataset, args.baseline, args.improved)
