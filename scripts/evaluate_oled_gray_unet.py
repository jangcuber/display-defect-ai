"""Calibrate on gray validation photos, then score a sealed full-scene test split."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from check_runtime_renderer import check as verify_runtime_rgb
from evaluate_paired_models import (
    KINDS, ROOT, calibrate_full, calibrate_tiles, check_frozen, collect_full, collect_tiles,
    digest, from_json, load_full, native_metrics, paths, postprocess, read_rows,
)
from display_defect_ai.gray_unet import GrayNativeEngine
from display_defect_ai.line_postprocess import box_iou


def save(path, value):
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def freeze(dataset, root, run):
    validation = json.loads((run / "validation_selection.json").read_text())
    source_dir = ROOT / "src/display_defect_ai"
    record = dict(selected_model="unet", selection_split="val", evaluation_split="test",
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  dataset_manifest_sha256=digest(dataset / "manifest.jsonl"),
                  mask_manifest_sha256=digest(dataset / "mask_manifest.jsonl"),
                  inference_source_sha256=digest(source_dir / "native_inference.py"),
                  runtime_source_sha256={name: digest(source_dir / name) for name in
                                         ("gray_unet.py", "pixel_model.py", "line_postprocess.py")},
                  evaluator_sha256=digest(ROOT / "scripts/evaluate_paired_models.py"),
                  criterion="Validation: minimize normal photo alarms, then maximize object macro F1",
                  models={"unet": validation})
    save(root / "frozen_selection.json", record)
    print("Frozen validation choices before full test inference", flush=True)


def groups_of_metrics(rows, predictions):
    groups = defaultdict(list)
    for row, items in zip(rows, predictions, strict=True):
        groups[row["gray_level"]].append((row, items))
    by_level = {}
    for level, values in sorted(groups.items()):
        truth, outputs = zip(*values, strict=True)
        by_level[str(level)] = native_metrics(truth, outputs)
        region_tiers = {tier: dict(found=0, total=0) for tier in ("weak", "medium", "strong")}
        for row, detections in values:
            regions = [obj for obj in row["objects"] if obj["kind"] == "dark_region"]
            if not regions:
                continue
            # The generator includes at most one area per scene.
            region, = regions
            tier = region_tiers[region["attributes"]["tier"]]
            tier["total"] += 1
            tier["found"] += any(item.kind == "dark_region" and box_iou(item.box, region["bbox_xyxy"]) >= .5
                                 for item in detections)
        by_level[str(level)]["regions_by_strength"] = region_tiers
    return dict(overall=native_metrics(rows, predictions), by_level=by_level)


def pixel_metrics(dataset, rows, engine, selected):
    totals = {r["gray_level"]: np.zeros((3, len(KINDS)), dtype=np.int64) for r in rows}
    for index, row in enumerate(rows, 1):
        image = cv2.imread(str(dataset / row["image"]))
        mask_path = dataset / "masks/test" / f"{row['image_id']}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        maps = engine.predict_tiles([image])[0]
        counts = totals[row["gray_level"]]
        for i, kind in enumerate(KINDS):
            target = (mask & (1 << i)) != 0
            predicted = maps[i] >= int(np.ceil(selected["pixel_thresholds"][kind] * 255))
            counts[:, i] += [np.count_nonzero(target & predicted), predicted.sum(), target.sum()]
        if index % 100 == 0:
            print(f"Gray test pixel metrics: {index}/{len(rows)}", flush=True)
    return {str(level): {kind: dict(tp=int(tp), predicted=int(pred), actual=int(gt),
                                    dice=float(2 * tp / max(1, gt + pred)),
                                    iou=float(tp / max(1, gt + pred - tp)))
                         for kind, (tp, pred, gt) in zip(KINDS, counts.T, strict=True)}
            for level, counts in sorted(totals.items())}


def evaluate(dataset, root, device):
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(.20)
    run, weights = paths(root, "unet")
    if not (root / "frozen_selection.json").exists():
        if not (root / "runtime_val_render_verification.json").exists():
            verify_runtime_rgb(dataset, root, "val")
        if not (run / "raw_tile_val.jsonl").exists():
            collect_tiles(dataset, root, "unet", device, engine_factory=GrayNativeEngine)
        calibrate_tiles(dataset, root, "unet")
        if not (run / "raw_full_val.jsonl").exists():
            collect_full(dataset, root, "unet", "val", device, engine_factory=GrayNativeEngine)
        calibrate_full(dataset, root, "unet")
        freeze(dataset, root, run)

    frozen = check_frozen(dataset, root, "unet")
    assert frozen["mask_manifest_sha256"] == digest(dataset / "mask_manifest.jsonl")
    assert frozen["evaluator_sha256"] == digest(ROOT / "scripts/evaluate_paired_models.py")
    selected = frozen["models"]["unet"]["selected"]
    if not (root / "runtime_test_render_verification.json").exists():
        verify_runtime_rgb(dataset, root, "test")
    if not (run / "raw_full_test.jsonl").exists():
        collect_full(dataset, root, "unet", "test", device, engine_factory=GrayNativeEngine)
    result = dict(selected=selected, weights_sha256=digest(weights), split="test",
                  full_scene_size=[5888, 3368], object_match="same class, one-to-one box IoU >= .50")
    for split in ("val", "test"):
        scenes = load_full(dataset, root, "unet", split)
        metadata = read_rows(dataset, split)
        for scene, row in zip(scenes, metadata, strict=True):
            scene["gray_level"] = row["gray_level"]
        predictions = [postprocess(from_json(scene["detections"]), selected) for scene in scenes]
        result["full_scene_" + split] = groups_of_metrics(scenes, predictions)
        if split == "test":
            file = run / "test_predictions.jsonl"
            file.write_text("".join(json.dumps(dict(image_id=row["image_id"], gray_level=row["gray_level"],
                                                    detections=[item._asdict() for item in items])) + "\n"
                                    for row, items in zip(scenes, predictions, strict=True)))
    engine = GrayNativeEngine("unet", weights, device)
    result["tile_pixel_metrics_by_level"] = pixel_metrics(dataset, read_rows(dataset, "test"), engine, selected)
    save(root / "test_results.json", result)
    print("Full-scene object evaluation and tile pixel metrics saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-gray-paired-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/oled-gray-unet-v1"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    evaluate(args.dataset, args.output, args.device)
