"""Create five independent class masks from isolated, paired synthetic effects."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai import paired_renderer as r


def generate(dataset):
    config = json.loads((dataset / "generation.json").read_text())
    if hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest() != config["renderer_sha256"]:
        raise ValueError("Renderer does not match dataset")
    if (dataset / "segmentation.json").exists():
        raise FileExistsError("Masks already completed")
    groups = defaultdict(list)
    for row in map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines()):
        groups[row["source_case_id"]].append(row)
    sources = {row["source_case_id"]: row for row in map(json.loads, (dataset / "source_cases.jsonl").read_text().splitlines())}
    records, counts, overlap_images = [], Counter(), 0
    for index, (sid, rows) in enumerate(groups.items()):
        source = sources[sid]
        window = source["tile_xyxy"]
        base = r.base_window(source, window)
        normal = r.quantize(base)
        normal_row = next(row for row in rows if row["kind"] == "normal")
        if not np.array_equal(normal, cv2.imread(str(dataset / normal_row["image"]))):
            raise ValueError(f"Normal replay mismatch: {sid}")
        for row in rows:
            mask = np.zeros((r.TILE, r.TILE), np.uint8)
            for obj, spec in zip(row["objects"], row["scene_objects"], strict=True):
                box, loss = r.effect_patch(source, [spec], window)
                x1, y1, x2, y2 = box
                px, py = x1 - window[0], y1 - window[1]
                patch = base[py:py + y2 - y1, px:px + x2 - x1]
                delta = cv2.absdiff(r.quantize(patch - loss), normal[py:py + y2 - y1, px:px + x2 - x1]).max(axis=2)
                visible = delta >= 2
                yy, xx = np.nonzero(visible)
                actual = [px + int(xx.min()), py + int(yy.min()), px + int(xx.max()) + 1, py + int(yy.max()) + 1]
                if actual != obj["bbox_xyxy"]:
                    raise ValueError(f"Mask and existing object label differ: {row['image_id']}")
                bit = np.uint8(1 << r.KINDS.index(obj["kind"]))
                mask[py:py + y2 - y1, px:px + x2 - x1] |= visible.astype(np.uint8) * bit
            if row["kind"] == "normal":
                assert not mask.any()
            else:
                assert mask.any()
            pixels = {kind: int(((mask >> i) & 1).sum()) for i, kind in enumerate(r.KINDS)}
            counts.update(pixels)
            overlap_images += bool(np.any((mask & (mask - np.uint8(1))) != 0))
            path = Path("masks") / row["split"] / f"{row['image_id']}.png"
            digest = r.write_png(dataset / path, mask)
            records.append(dict(image_id=row["image_id"], source_case_id=sid, split=row["split"],
                                mask=path.as_posix(), mask_sha256=digest, class_pixels=pixels))
        if (index + 1) % 100 == 0:
            print(f"Masks: {index + 1}/{len(groups)} families, {len(records)} images", flush=True)
    path = dataset / "mask_manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    record = dict(schema_version="oled-bitmasks-1", images=len(records), classes=list(r.KINDS),
                  encoding="uint8 PNG, class i is bit 1<<i; overlaps preserve multiple bits",
                  threshold_dn=2, inputs="image RGB only; normal/reference recipes used only for labels",
                  dataset_manifest_sha256=hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest(),
                  mask_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  renderer_sha256=config["renderer_sha256"], class_pixels=dict(counts),
                  overlapping_class_mask_images=int(overlap_images), numpy=np.__version__, opencv=cv2.__version__)
    (dataset / "segmentation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    generate(args.dataset)
