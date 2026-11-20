"""Train YOLO26n on the five-category OLED dataset using local Apple MPS."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import torch
from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai.oled_defects import KINDS, LINE_KINDS


def train(dataset, output, epochs, *, weights=None, size=640, thin_focus=False):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable. Run outside the restricted GPU sandbox.")
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    source = weights or ROOT / "outputs/yolo26n-white-lines-v2/finetune/weights/best.pt"
    output.mkdir(parents=True)
    images = []
    for row in rows:
        if row["split"] != "train":
            continue
        thicknesses = [obj["attributes"]["thickness"] for obj in row["objects"] if obj["kind"] in LINE_KINDS]
        thickness = min(thicknesses, default=10)
        repetitions = 3 if thickness == 1 else 2 if thickness == 2 else 1
        if thin_focus:
            repetitions = 6 if thickness == 1 else 3 if thickness == 2 else 1
            if any(obj["kind"] in LINE_KINDS and obj["attributes"]["thickness"] == 1
                   and obj["attributes"]["line_form"] == "full" for obj in row["objects"]):
                repetitions = 12
        images.extend([str((dataset / row["image"]).resolve())] * repetitions)
    train_list = output / "train_images.txt"
    train_list.write_text("\n".join(images) + "\n", encoding="utf-8")
    names = "".join(f"  {index}: {kind}\n" for index, kind in enumerate(KINDS))
    config = output / "weighted_train.yaml"
    config.write_text(
        f"path: {json.dumps(str(dataset.resolve()))}\ntrain: {json.dumps(str(train_list.resolve()))}\n"
        f"val: images/val\ntest: images/test\nnames:\n{names}", encoding="utf-8",
    )
    record = dict(
        dataset=str(dataset.resolve()), dataset_manifest_sha256=hashlib.sha256(
            (dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        initial_weights=str(source), initial_weights_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__, opencv=cv2.__version__, classes=list(KINDS),
        source_train_images=sum(row["split"] == "train" for row in rows),
        weighted_train_samples=len(images), image_counts=dict(Counter(row["kind"] for row in rows)),
        device="mps", epochs=epochs, image_size=size, seed=43, thin_focus=thin_focus,
    )
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    settings.update({"sync": False})
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    YOLO(source).train(
        data=str(config.resolve()), project=str(output.resolve()), name="train",
        epochs=epochs, patience=4, imgsz=size, batch=16, device="mps", workers=0, cache=False,
        seed=43, deterministic=False, amp=False, optimizer="AdamW", lr0=.0003 if thin_focus else .0008,
        warmup_epochs=.5, cos_lr=True, mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=.5, flipud=.5, plots=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--weights", type=Path, help="Local starting weights for continued training")
    parser.add_argument("--size", type=int, choices=(640, 768, 960), default=640)
    parser.add_argument("--thin-focus", action="store_true", help="Oversample 1px full lines 12×, other 1px lines 6×, 2px lines 3×")
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, weights=args.weights, size=args.size, thin_focus=args.thin_focus)
