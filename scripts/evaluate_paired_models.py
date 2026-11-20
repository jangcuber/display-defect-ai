"""Calibrate on validation, freeze a winner, then compare two models on fixed test scenes."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai.line_postprocess import Detection, box_iou
from display_defect_ai.native_inference import (KINDS, NativeEngine, components, maps_to_candidates,
                                               postprocess, tile_windows)
from evaluate_oled import summarize

PIXEL_GRID = (.25, .4, .55, .7, .85, .95)
SCORE_GRID = (.001, .005, .01, .025, .05, .1, .2, .35, .5, .65, .8, .9, .97, .995, 1., 1.01)
MIN_AREAS = dict(zip(KINDS, (3, 3, 6, 6, 24)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths(root, model):
    run = root / model
    weights = run / "best.pt" if model == "unet" else run / "train/weights/best.pt"
    return run, weights


def check_frozen(dataset, root, model):
    frozen = json.loads((root / "frozen_selection.json").read_text())
    _, weights = paths(root, model)
    if frozen["dataset_manifest_sha256"] != digest(dataset / "manifest.jsonl"):
        raise ValueError("Dataset changed after validation freeze")
    if frozen["models"][model]["weights_sha256"] != digest(weights):
        raise ValueError("Weights changed after validation freeze")
    if frozen["inference_source_sha256"] != digest(ROOT / "src/display_defect_ai/native_inference.py"):
        raise ValueError("Inference changed after validation freeze")
    for name, expected in frozen.get("runtime_source_sha256", {}).items():
        if digest(ROOT / "src/display_defect_ai" / name) != expected:
            raise ValueError("Runtime changed after validation freeze: " + name)
    return frozen


def read_rows(dataset, split):
    return [row for row in map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines()) if row["split"] == split]


def from_json(items):
    return [Detection(item["score"], item["kind"], tuple(item["box"])) for item in items]


def native_metrics(rows, predictions):
    result = summarize(rows, predictions)
    lines = fragmented = surplus = 0
    mixed_total = mixed_complete = 0
    for row, items in zip(rows, predictions, strict=True):
        objects = row["objects"]
        if len({obj["kind"] for obj in objects}) > 1:
            mixed_total += 1
            claimed = set()
            for item in sorted(items, key=lambda value: value.score, reverse=True):
                overlap, index = max(((box_iou(item.box, obj["bbox_xyxy"]), index)
                                      for index, obj in enumerate(objects)
                                      if obj["kind"] == item.kind and index not in claimed), default=(0, -1))
                if overlap >= .5:
                    claimed.add(index)
            mixed_complete += len(claimed) == len(objects)
        for obj in row["objects"]:
            if obj["kind"] not in KINDS[2:4]:
                continue
            lines += 1
            axis = 0 if obj["kind"] == "horizontal_line" else 1
            cross = 1 - axis
            truth = obj["bbox_xyxy"]
            supported = 0
            for item in items:
                if item.kind != obj["kind"]:
                    continue
                box = item.box
                along = min(box[axis+2], truth[axis+2]) - max(box[axis], truth[axis])
                across = min(box[cross+2], truth[cross+2]) - max(box[cross], truth[cross])
                if (along >= .5 * min(box[axis+2]-box[axis], truth[axis+2]-truth[axis])
                        and across >= .5 * min(box[cross+2]-box[cross], truth[cross+2]-truth[cross])):
                    supported += 1
            fragmented += supported > 1
            surplus += max(0, supported - 1)
    result["mixed_type_images"] = dict(total=mixed_total, all_objects_found=mixed_complete)
    result["line_fragmentation"] = dict(total_lines=lines, lines_with_multiple_boxes=fragmented,
                                        surplus_aligned_boxes=surplus,
                                        criterion="same class, overlap >= half the smaller longitudinal and cross spans")
    return result


def rank(metrics):
    return (-metrics["normal_false_alarms"], sum(value["f1"] for value in metrics["classes"].values()),
            metrics["one_pixel_lines"]["hit"], metrics["mean_matched_iou"])


def collect_tiles(dataset, root, model, device, *, engine_factory=NativeEngine):
    run, weights = paths(root, model)
    path = run / "raw_tile_val.jsonl"
    if path.exists():
        raise FileExistsError(path)
    engine = engine_factory(model, weights, device)
    rows = read_rows(dataset, "val")
    sha = digest(weights)
    pending = path.with_suffix(".pending")
    with pending.open("w") as file:
        for index, row in enumerate(rows):
            image = cv2.imread(str(dataset / row["image"]))
            value = engine.predict_tiles([image])[0]
            if model == "unet":
                variants = {}
                for pixel in PIXEL_GRID:
                    candidates = []
                    for channel, kind in enumerate(KINDS):
                        candidates.extend(components(value[channel], kind, pixel, MIN_AREAS[kind]))
                    variants[str(pixel)] = [item._asdict() for item in candidates]
            else:
                variants = {"boxes": [item._asdict() for item in value]}
            file.write(json.dumps(dict(image_id=row["image_id"], weights_sha256=sha, variants=variants)) + "\n")
            if (index + 1) % 100 == 0:
                print(f"{model} validation tiles: {index + 1}/{len(rows)}", flush=True)
    if digest(weights) != sha:
        raise ValueError("Checkpoint changed during inference")
    pending.rename(path)


def calibrate_tiles(dataset, root, model):
    run, weights = paths(root, model)
    rows = read_rows(dataset, "val")
    records = list(map(json.loads, (run / "raw_tile_val.jsonl").read_text().splitlines()))
    assert [row["image_id"] for row in rows] == [row["image_id"] for row in records]
    weight_sha = digest(weights)
    assert all(row["weights_sha256"] == weight_sha for row in records)
    selected = dict(size=1024, thresholds={}, line_gap=128)
    if model == "unet":
        selected.update(pixel_thresholds={}, min_areas=MIN_AREAS)
    for kind in KINDS:
        best = None
        for variant in records[0]["variants"]:
            candidates = [[item for item in from_json(record["variants"][variant]) if item.kind == kind]
                          for record in records]
            normal_max = max((item.score for row, items in zip(rows, candidates) if not row["objects"] for item in items), default=0)
            grid = sorted(set((*SCORE_GRID, min(1., max(.001, normal_max + .000001)))))
            for score in grid:
                thresholds = {k: 1.01 for k in KINDS}
                thresholds[kind] = score
                predictions = [postprocess(items, {**selected, "thresholds": thresholds}) for items in candidates]
                metric = native_metrics(rows, predictions)
                values = metric["classes"][kind]
                ranking = (-metric["normal_false_alarms"], values["f1"], values["recall"], metric["mean_matched_iou"])
                if best is None or ranking > best[0]:
                    best = (ranking, variant, score)
        _, variant, score = best
        selected["thresholds"][kind] = score
        if model == "unet":
            selected["pixel_thresholds"][kind] = float(variant)
    predictions = []
    for record in records:
        candidates = []
        for kind in KINDS:
            key = str(selected["pixel_thresholds"][kind]) if model == "unet" else "boxes"
            candidates.extend(item for item in from_json(record["variants"][key]) if item.kind == kind)
        predictions.append(postprocess(candidates, selected))
    result = dict(split="val", mode="native_tile", images=len(rows), selected=selected,
                  weights_sha256=digest(weights), dataset_manifest_sha256=digest(dataset / "manifest.jsonl"),
                  metrics=native_metrics(rows, predictions))
    (run / "tile_validation_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def full_objects(renderer, source, specs):
    result = []
    full = (0, 0, renderer.PHOTO_W, renderer.PHOTO_H)
    for spec in specs:
        box, loss = renderer.effect_patch(source, [spec], full)
        base = renderer.base_window(source, box)
        deficit = cv2.absdiff(renderer.quantize(base), renderer.quantize(base - loss)).max(axis=2)
        y, x = np.nonzero(deficit >= 2)
        if not len(x):
            raise ValueError("Full-scene object is invisible")
        bbox = [box[0] + int(x.min()), box[1] + int(y.min()), box[0] + int(x.max()) + 1, box[1] + int(y.max()) + 1]
        result.append(dict(kind=spec["kind"], bbox_xyxy=bbox, attributes=spec))
    return result


def photo_provider(renderer, source, specs, bases, normal_tiles):
    """Render exact RGB inputs; reuse quantized background only outside all effect support."""
    support = None
    if specs:
        bounds = [renderer.native_bounds(spec) for spec in specs]
        support = (renderer.BORDER + min(box[0] for box in bounds) * renderer.SCALE - 3,
                   renderer.BORDER + min(box[1] for box in bounds) * renderer.SCALE - 3,
                   renderer.BORDER + max(box[2] for box in bounds) * renderer.SCALE + 3,
                   renderer.BORDER + max(box[3] for box in bounds) * renderer.SCALE + 3)

    def provide(window):
        key = tuple(window)
        if support is None or (max(window[0], support[0]) >= min(window[2], support[2])
                               or max(window[1], support[1]) >= min(window[3], support[3])):
            return normal_tiles[key]
        return renderer.render_from_base(source, specs, window, bases[key])
    return provide


def collect_full(dataset, root, model, split, device, *, engine_factory=NativeEngine):
    from display_defect_ai import paired_renderer as renderer
    run, weights = paths(root, model)
    path = run / f"raw_full_{split}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    if split == "test" and not (root / "frozen_selection.json").exists():
        raise ValueError("Freeze validation selection before accessing test predictions")
    if split == "test":
        check_frozen(dataset, root, model)
    calibration = json.loads((run / "tile_validation_selection.json").read_text())
    selected = calibration["selected"]
    sha = digest(weights)
    assert calibration["weights_sha256"] == sha
    rows = read_rows(dataset, split)
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_case_id"]].append(row)
    sources = {row["source_case_id"]: row for row in map(json.loads, (dataset / "source_cases.jsonl").read_text().splitlines())}
    truth_path = root / f"full_{split}_ground_truth.json"
    ground_truth = None
    if truth_path.exists():
        saved_truth = json.loads(truth_path.read_text())
        assert saved_truth["dataset_manifest_sha256"] == digest(dataset / "manifest.jsonl")
        assert saved_truth["renderer_sha256"] == digest(Path(renderer.__file__))
        assert [record["image_id"] for record in saved_truth["rows"]] == [row["image_id"] for row in rows]
        ground_truth = {record["image_id"]: record["objects"] for record in saved_truth["rows"]}
    engine = engine_factory(model, weights, device)
    inset = renderer.BORDER + 3  # exclude the known 7-tap optical border transition
    roi = (inset, inset, renderer.PHOTO_W - inset, renderer.PHOTO_H - inset)
    windows = tile_windows(renderer.PHOTO_W, renderer.PHOTO_H, roi)
    pending = path.with_suffix(".pending")
    source_sha = digest(ROOT / "src/display_defect_ai/native_inference.py")
    data_sha = digest(dataset / "manifest.jsonl")
    config_sha = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    completed = []
    if pending.exists():
        for line in pending.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                break
            if (record.get("weights_sha256") != sha or record.get("inference_source_sha256") != source_sha
                    or record.get("dataset_manifest_sha256") != data_sha or record.get("config_sha256") != config_sha):
                raise ValueError("Pending full-scene predictions use a different experiment")
            completed.append(record)
        # Commit only complete seven-scene source families; replay an interrupted family.
        completed = completed[:len(completed) // 7 * 7]
        assert [record["image_id"] for record in completed] == [row["image_id"] for row in rows[:len(completed)]]
        pending.write_text("".join(json.dumps(record) + "\n" for record in completed))
    start = time.perf_counter()
    with pending.open("a") as file:
        for group_index, (sid, group) in enumerate(groups.items()):
            if group_index < len(completed) // 7:
                continue
            source = sources[sid]
            # Verify this numerical environment reproduces the materialized source before using recipes.
            stored = cv2.imread(str(dataset / group[0]["normal_image"]))
            actual = renderer.render_window(source, [], source["tile_xyxy"])
            if not np.array_equal(stored, actual):
                raise ValueError("Renderer environment differs from the materialized dataset")
            bases = {tuple(window): renderer.base_window(source, window) for window in windows}
            normal_tiles = {key: renderer.quantize(base) for key, base in bases.items()}
            pixel_cache = {}
            for row in group:
                provide = photo_provider(renderer, source, row["scene_objects"], bases, normal_tiles)
                candidates, maps = engine.predict_provider(provide, renderer.PHOTO_W, renderer.PHOTO_H,
                                                           selected, roi=roi, raw=True, cache=pixel_cache)
                objects = ground_truth[row["image_id"]] if ground_truth is not None else full_objects(renderer, source, row["scene_objects"])
                file.write(json.dumps(dict(image_id=row["image_id"], source_case_id=sid, weights_sha256=sha,
                                           objects=objects, detections=[item._asdict() for item in candidates],
                                           inference_source_sha256=source_sha, dataset_manifest_sha256=data_sha,
                                           config_sha256=config_sha)) + "\n")
                file.flush()
                del maps
            print(f"{model} full {split}: {group_index + 1}/{len(groups)} families, {time.perf_counter() - start:.1f}s", flush=True)
            del bases, pixel_cache, normal_tiles, provide
    if digest(weights) != sha:
        raise ValueError("Checkpoint changed during full-scene inference")
    pending.rename(path)


def load_full(dataset, root, model, split):
    run, weights = paths(root, model)
    data = list(map(json.loads, (run / f"raw_full_{split}.jsonl").read_text().splitlines()))
    expected = read_rows(dataset, split)
    assert [row["image_id"] for row in expected] == [row["image_id"] for row in data]
    weight_sha = digest(weights)
    assert all(row["weights_sha256"] == weight_sha for row in data)
    return data


def calibrate_full(dataset, root, model):
    run, weights = paths(root, model)
    data = load_full(dataset, root, model, "val")
    selected = json.loads((run / "tile_validation_selection.json").read_text())["selected"]
    candidates = [from_json(row["detections"]) for row in data]
    for kind in KINDS:
        normal_max = max((item.score for row, items in zip(data, candidates) if not row["objects"]
                          for item in items if item.kind == kind), default=0)
        best = None
        for score in sorted(set((*SCORE_GRID, min(1., max(.001, normal_max + .000001))))):
            thresholds = {key: 1.01 for key in KINDS}
            thresholds[kind] = score
            metric = native_metrics(data, [postprocess(items, {**selected, "thresholds": thresholds}) for items in candidates])
            cls = metric["classes"][kind]
            ranking = (-metric["normal_false_alarms"], cls["f1"], cls["recall"], metric["mean_matched_iou"])
            if best is None or ranking > best[0]:
                best = ranking, score
        selected["thresholds"][kind] = best[1]
    # Joint point suppression couples the class thresholds; refine the combined objective.
    for _ in range(2):
        for kind in KINDS:
            normal_max = max((item.score for row, items in zip(data, candidates) if not row["objects"]
                              for item in items if item.kind == kind), default=0)
            best = None
            for score in sorted(set((*SCORE_GRID, min(1., max(.001, normal_max + .000001)), selected["thresholds"][kind]))):
                thresholds = {**selected["thresholds"], kind: score}
                metric = native_metrics(data, [postprocess(items, {**selected, "thresholds": thresholds}) for items in candidates])
                ranking = rank(metric)
                if best is None or ranking > best[0]:
                    best = ranking, score
            selected["thresholds"][kind] = best[1]
    # Reconstruct or suppress region boxes split across overlapping tile boundaries.
    normal_max = max((item.score for row, items in zip(data, candidates) if not row["objects"]
                      for item in items if item.kind == "dark_region"), default=0)
    best_region = None
    for merge_regions in (False, True):
        for region_iou in (.35, .25, .15):
            for score in sorted(set((*SCORE_GRID, min(1., max(.001, normal_max + .000001)), selected["thresholds"]["dark_region"]))):
                config = {**selected, "merge_regions": merge_regions, "region_iou": region_iou,
                          "thresholds": {**selected["thresholds"], "dark_region": score}}
                metric = native_metrics(data, [postprocess(items, config) for items in candidates])
                ranking = rank(metric)
                if best_region is None or ranking > best_region[0]:
                    best_region = ranking, config
    selected = best_region[1]
    metrics = native_metrics(data, [postprocess(items, selected) for items in candidates])
    result = dict(split="val", mode="full_native_scene", images=len(data), selected=selected,
                  model=model, weights_sha256=digest(weights), dataset_manifest_sha256=digest(dataset / "manifest.jsonl"),
                  inference_source_sha256=digest(ROOT / "src/display_defect_ai/native_inference.py"), metrics=metrics)
    (run / "validation_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def freeze(dataset, root):
    path = root / "frozen_selection.json"
    if path.exists():
        raise FileExistsError(path)
    reports = {model: json.loads((root / model / "validation_selection.json").read_text()) for model in ("unet", "yolo")}
    winner = max(reports, key=lambda model: rank(reports[model]["metrics"]))
    result = dict(selected_model=winner, selection_split="val", evaluation_split="test",
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  criterion="minimize normal image alarms, then maximize macro object F1; IoU>=0.5",
                  input_resolution=1024, full_scene_size=[5888, 3368], overlap=128,
                  dataset_manifest_sha256=digest(dataset / "manifest.jsonl"), models=reports,
                  inference_source_sha256=digest(ROOT / "src/display_defect_ai/native_inference.py"))
    result["runtime_source_sha256"] = {name: digest(ROOT / "src/display_defect_ai" / name)
        for name in ("pixel_model.py", "oled_yolo.py", "line_postprocess.py")}
    result["mask_manifest_sha256"] = digest(dataset / "mask_manifest.jsonl")
    result["evaluation_source_sha256"] = digest(Path(__file__))
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Frozen validation winner:", winner)


def compare(dataset, root):
    frozen = json.loads((root / "frozen_selection.json").read_text())
    result = dict(split="test", minimum_iou=.5, selected_model=frozen["selected_model"], models={})
    for model in ("unet", "yolo"):
        check_frozen(dataset, root, model)
        data = load_full(dataset, root, model, "test")
        selected = frozen["models"][model]["selected"]
        predictions = [postprocess(from_json(row["detections"]), selected) for row in data]
        metrics = native_metrics(data, predictions)
        pooled = {}
        for name, kinds in (("points", KINDS[:2]), ("lines", KINDS[2:4]), ("regions", KINDS[4:])):
            counts = {key: sum(metrics["classes"][kind][key] for kind in kinds) for key in ("tp", "fp", "fn", "total")}
            counts.update(precision=counts["tp"] / max(1, counts["tp"] + counts["fp"]),
                          recall=counts["tp"] / max(1, counts["total"]))
            pooled[name] = counts
        result["models"][model] = dict(metrics=metrics, pooled=pooled, images=len(data),
                                        weights_sha256=frozen["models"][model]["weights_sha256"])
        (root / model / "test_predictions.jsonl").write_text("".join(json.dumps(dict(image_id=row["image_id"],
            detections=[item._asdict() for item in items])) + "\n" for row, items in zip(data, predictions)))
    result.update(dataset_manifest_sha256=frozen["dataset_manifest_sha256"], note="Controlled synthetic native scenes; real capture performance not validated")
    (root / "test_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect-tiles", "calibrate-tiles", "collect-full", "calibrate-full", "freeze", "compare"))
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-white-paired-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paired-model-comparison-v1"))
    parser.add_argument("--model", choices=("unet", "yolo"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.action == "collect-tiles":
        collect_tiles(args.dataset, args.output, args.model, args.device)
    elif args.action == "calibrate-tiles":
        calibrate_tiles(args.dataset, args.output, args.model)
    elif args.action == "collect-full":
        collect_full(args.dataset, args.output, args.model, args.split, args.device)
    elif args.action == "calibrate-full":
        calibrate_full(args.dataset, args.output, args.model)
    elif args.action == "freeze":
        freeze(args.dataset, args.output)
    else:
        compare(args.dataset, args.output)
