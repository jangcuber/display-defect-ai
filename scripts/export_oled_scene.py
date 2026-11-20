"""Export a complete, native-resolution scene from a paired OLED dataset recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai import paired_renderer as r


def export(dataset, scene_id, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((dataset / "generation.json").read_text())
    if hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest() != config["renderer_sha256"]:
        raise ValueError("Use the renderer revision matching this dataset's renderer_sha256")
    rows = map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines())
    scene = next((row for row in rows if row["scene_id"] == scene_id), None)
    if scene is None:
        raise ValueError(f"Unknown scene: {scene_id}")
    sources = map(json.loads, (dataset / "source_cases.jsonl").read_text().splitlines())
    source = next(row for row in sources if row["source_case_id"] == scene["source_case_id"])
    image = r.render_full(source, scene["scene_objects"])
    x1, y1, x2, y2 = source["tile_xyxy"]
    stored = cv2.imread(str(dataset / scene["image"]))
    if not np.array_equal(stored, image[y1:y2, x1:x2]):
        raise ValueError("Full export does not reproduce the stored native tile")
    digest = r.write_png(output, image)
    metadata = dict(scene_id=scene_id, source_case_id=source["source_case_id"],
                    full_photo_size=[r.PHOTO_W, r.PHOTO_H], tile_xyxy=source["tile_xyxy"],
                    tile_exactly_matches=True, png_sha256=digest, pixel_sha256=r.pixel_hash(image),
                    is_synthetic=True, calibrated_to_real_hardware=False,
                    source=source, scene_objects=scene["scene_objects"])
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key not in ("source", "scene_objects")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    export(args.dataset, args.scene_id, args.output)
