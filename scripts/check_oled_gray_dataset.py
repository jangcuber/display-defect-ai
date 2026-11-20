"""Verify gray-level pairing, leakage, mask membership and exact renderer replay."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from generate_oled_gray_dataset import LEVELS, KINDS, KINDS_OF_IMAGE, render, visible_objects


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(dataset):
    cv2.setNumThreads(2)
    metadata = json.loads((dataset / "segmentation.json").read_text())
    assert digest(dataset / "manifest.jsonl") == metadata["dataset_manifest_sha256"]
    assert digest(dataset / "mask_manifest.jsonl") == metadata["mask_manifest_sha256"]
    assert digest(Path(render.__file__)) == metadata["renderer_sha256"]
    rows = list(map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines()))
    masks = {r["image_id"]: r for r in map(json.loads, (dataset / "mask_manifest.jsonl").read_text().splitlines())}
    sources = {r["source_case_id"]: r for r in map(json.loads, (dataset / "source_cases.jsonl").read_text().splitlines())}
    assert len(rows) == len(masks) == metadata["images"]
    assert len({r["pixel_sha256"] for r in rows}) == len(rows)
    by_panel = defaultdict(list)
    counts = Counter()
    overlap = replayed = 0
    for row in rows:
        by_panel[row["panel_id"]].append(row)
    for index, (panel, group) in enumerate(by_panel.items()):
        assert len({row["split"] for row in group}) == 1, panel
        for kind in KINDS_OF_IMAGE:
            matched = [r for r in group if r["kind"] == kind]
            assert tuple(r["gray_level"] for r in matched) == LEVELS, panel
            assert all(r["scene_objects"] == matched[0]["scene_objects"] and r["tile_xyxy"] == matched[0]["tile_xyxy"]
                       for r in matched), panel
        for row in group:
            name = row["image_id"]
            image_path, mask_path = dataset / row["image"], dataset / masks[name]["mask"]
            assert digest(image_path) == row["image_sha256"]
            assert digest(mask_path) == masks[name]["mask_sha256"]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            assert mask.shape == (render.TILE, render.TILE) and mask.max() <= 31, name
            assert {kind for i, kind in enumerate(KINDS) if (mask & (1 << i)).any()} == {
                obj["kind"] for obj in row["objects"]}, name
            assert (row["kind"] == "normal") == (not mask.any()), name
            overlap += int(((mask & (mask - np.uint8(1))) != 0).any())
            counts[f"{row['split']}/g{row['gray_level']:03d}"] += 1
            if index % 20 == 0:
                source = sources[row["source_case_id"]]
                window = source["tile_xyxy"]
                base = render.base_window(source, window)
                normal = render.quantize(base)
                rgb = render.render_from_base(source, row["scene_objects"], window, base)
                objects, expected_mask = visible_objects(source, row["scene_objects"], base, normal)
                assert np.array_equal(rgb, cv2.imread(str(image_path))), name
                assert objects == row["objects"] and np.array_equal(expected_mask, mask), name
                replayed += 1
        if (index + 1) % 30 == 0:
            print(f"Verification: {index + 1}/{len(by_panel)} panels", flush=True)
    record = dict(images=len(rows), matched_panels=len(by_panel), split_counts=counts,
                  duplicate_rgb=0, leaked_panels=0, empty_normal_masks=len(by_panel) * len(LEVELS),
                  overlapping_class_masks=overlap, exact_replayed_images_and_masks=replayed)
    (dataset / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    check(parser.parse_args().dataset)
