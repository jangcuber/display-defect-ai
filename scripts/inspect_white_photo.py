"""Run both local detectors on one white-screen photo for the Mac inspector."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from display_defect_ai.oled_defects import OledRuleSettings, describe_detections, detect_oled_white_screen
from display_defect_ai.photo_inspection import InvalidPhoto, prepare_panel, project_detections

YOLO_RUN = ROOT / "outputs/yolo26n-oled-lines-v3"
YOLO_WEIGHTS = YOLO_RUN / "train/weights/best.pt"


def _yolo_detect(image, selected):
    import torch
    from ultralytics import settings
    from display_defect_ai.oled_yolo import load_model, predict_image

    torch.set_num_threads(4)
    settings.update({"sync": False})
    return predict_image(load_model(YOLO_WEIGHTS), image, selected)


def inspect(path: Path, output: Path) -> dict:
    photo = prepare_panel(path)
    calibration = json.loads((YOLO_RUN / "validation_selection.json").read_text())
    selected = calibration["selected"]
    if calibration["schema_version"] != "oled-white-2" or calibration["split"] != "val":
        raise ValueError("Expected five-class OLED validation settings")
    if hashlib.sha256(YOLO_WEIGHTS.read_bytes()).hexdigest() != calibration["weights_sha256"]:
        raise ValueError("Model and selected validation settings do not match")

    opencv = detect_oled_white_screen(photo.aligned, OledRuleSettings(**calibration["opencv"]))
    yolo = _yolo_detect(photo.aligned, selected)
    height, width = photo.original.shape[:2]
    scale = min(1.0, 1400 / max(width, height))
    preview = cv2.resize(photo.original, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(output / "preview.png"), preview):
        raise OSError("Could not save the preview")

    def marks(detections):
        positions = project_detections(photo, detections)
        attributes = describe_detections(photo.aligned, detections)
        return [dict(position, **description) for position, description in zip(positions, attributes, strict=True)]

    return {
        "schema_version": "oled-white-2",
        "decision": "defect" if opencv or yolo else "normal",
        "rule_id": "oled-white-five-either-detector-v1",
        "width": width,
        "height": height,
        "opencv": marks(opencv),
        "yolo": marks(yolo),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    try:
        result = inspect(args.photo, args.output)
    except InvalidPhoto as error:
        result = {"decision": "unavailable", "message": str(error)}
    except (OSError, KeyError, ValueError):
        traceback.print_exc()
        result = {"decision": "unavailable", "message": "검사 설정이나 모델 파일을 읽지 못했습니다."}
    except Exception:
        traceback.print_exc()
        result = {"decision": "unavailable", "message": "사진을 검사하지 못했습니다. 다른 사진으로 다시 시도해 주세요."}
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return 0 if result["decision"] != "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
