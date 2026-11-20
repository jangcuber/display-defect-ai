"""Train a fresh COCO-initialized YOLO26n at native 1024px on paired v2 data."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, settings


def train(dataset, output, epochs, batch, resume=False, patience=5):
    if resume:
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        torch.mps.set_per_process_memory_fraction(.5 if batch >= 16 else .28 if batch >= 8 else .22)
        record = json.loads((output / "run.json").read_text())
        record.setdefault("resource_resume_history", []).append(record.get("resource_resume", {}))
        record["resource_resume"] = dict(batch=batch, patience=patience, mps_memory_fraction=.5 if batch >= 16 else .28 if batch >= 8 else .22, reason="Sequential training; memory availability recovered; enlarge batch for stability and throughput")
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        model = YOLO(output / "train/weights/last.pt")
        ticks = [0]
        def release_cache(trainer):
            ticks[0] += 1
            if ticks[0] % 32 == 0:
                torch.mps.empty_cache()
        model.add_callback("on_train_batch_end", release_cache)
        model.add_callback("on_val_end", lambda validator: torch.mps.empty_cache())
        model.train(resume=True, batch=batch, patience=patience, device="mps", workers=0, cache=False)
        return
    if output.exists():
        raise FileExistsError(output)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Run with local Metal GPU access")
    torch.mps.set_per_process_memory_fraction(.5 if batch >= 16 else .28 if batch >= 8 else .22)
    source = Path("artifacts/pretrained/yolo26n.pt")
    output.mkdir(parents=True)
    record = dict(model="YOLO26n", initial_weights=str(source), initialization="generic COCO pretrained",
                  initial_weights_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest(),
                  epochs_max=epochs, batch=batch, image_size=1024, training_images=4900,
                  seed=43, oversampling=False, device="mps", torch=torch.__version__)
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    settings.update({"sync": False})
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    YOLO(source).train(
        data=str((dataset / "dataset.yaml").resolve()), project=str(output.resolve()), name="train",
        epochs=epochs, patience=patience, imgsz=1024, batch=batch, device="mps", workers=0, cache=False,
        seed=43, deterministic=False, amp=False, optimizer="AdamW", lr0=.001,
        warmup_epochs=.5, cos_lr=True, mosaic=0, close_mosaic=0, mixup=0, copy_paste=0,
        degrees=0, translate=0, scale=0, shear=0, perspective=0,
        hsv_h=0, hsv_s=0, hsv_v=0, fliplr=.5, flipud=.5, plots=False, save_period=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-white-paired-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paired-model-comparison-v1/yolo"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.batch, args.resume, args.patience)
