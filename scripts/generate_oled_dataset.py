"""Generate fixed-framing white OLED images with five visual defect categories."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from display_defect_ai.oled_defects import KINDS, LINE_KINDS
from generate_dataset import draw_defect_mask, make_solid_panel, render_photo, write_image
from generate_training_dataset import apply_drive_fault


def make_mask(kind, rng, *, center=None):
    mask = np.zeros((512, 512), np.uint8)
    if kind in LINE_KINDS:
        form = str(rng.choice(("full", "partial", "broken")))
        detail = draw_defect_mask(mask, "line", rng, horizontal=kind == "horizontal_line", line_form=form)
        return mask, dict(detail, line_form=form)
    if kind == "dark_region":
        width, height = (int(value) for value in rng.integers(36, 103, 2))
        x, y = int(rng.integers(12, 500 - width)), int(rng.integers(12, 500 - height))
        shape = str(rng.choice(("ellipse", "rectangle", "irregular")))
        if shape == "ellipse":
            cv2.ellipse(mask, (x + width // 2, y + height // 2), (width // 2, height // 2), 0, 0, 360, 255, -1)
        elif shape == "rectangle":
            mask[y:y + height, x:x + width] = 255
        else:
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            radius = rng.uniform(.72, 1.0, len(angles))
            points = np.column_stack((x + width / 2 * (1 + radius * np.cos(angles)),
                                      y + height / 2 * (1 + radius * np.sin(angles))))
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        return mask, dict(shape=shape, width=width, height=height)
    radius = int(rng.integers(1, 8))
    x, y = center if center is not None else (int(v) for v in rng.integers(radius + 10, 502 - radius, 2))
    cv2.circle(mask, (x, y), radius, 255, -1)
    return mask, dict(center=[x, y], radius=radius)


def make_object(kind, normal_panel, normal_photo, capture, rng, occupied, *, center=None, group=None):
    for _ in range(120):
        mask, detail = make_mask(kind, rng, center=center)
        # Separate independently labeled objects, including nearby cluster members.
        expanded = cv2.dilate(mask, np.ones((17, 17), np.uint8))
        if np.any((expanded > 0) & (occupied > 0)):
            continue
        channels = "RGB"
        if kind == "color_dot" or (kind in LINE_KINDS and rng.random() < .5):
            channels = str(rng.choice(("R", "G", "B")))
        tier = int(rng.integers(0, 3))
        low, high = ((.08, .22), (.22, .60), (.60, 1.0))[tier]
        strength = float(rng.uniform(low, high))
        if tier == 2 and rng.random() < .3:
            strength = 1.0
        changed = apply_drive_fault(normal_panel, normal_panel, mask > 0, channels, strength, False)
        delta = np.abs(capture(changed).astype(np.int16) - normal_photo).max(axis=2)
        if delta.max() < 12 or np.count_nonzero(delta >= 8) < 2:
            continue
        ys, xs = np.nonzero(delta)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        detail.update(affected_channels=channels, strength=strength, drive_effect="output_loss",
                      peak_delta=int(delta.max()), cluster_id=group)
        return mask, changed, dict(kind=kind, bbox_xyxy=box, attributes=detail)
    raise ValueError(f"Could not place a visible {kind} object")


def defect_image(primary, normal_panel, normal_photo, capture, rng):
    panel = normal_panel.copy()
    occupied = np.zeros((512, 512), np.uint8)
    objects = []

    def place(kind, **kwargs):
        mask, changed, obj = make_object(kind, normal_panel, normal_photo, capture, rng, occupied, **kwargs)
        panel[mask > 0] = changed[mask > 0]
        occupied[mask > 0] = 255
        objects.append(obj)

    clustered = primary in ("dark_dot", "color_dot") and rng.random() < .25
    if clustered:
        x, y = (int(v) for v in rng.integers(65, 400, 2))
        count = int(rng.integers(2, 5))
        offsets = ((0, 0), (32, 0), (0, 32), (32, 32))
        for dx, dy in offsets[:count]:
            place(primary, center=(x + dx, y + dy), group="cluster-0")
    else:
        place(primary)
    if rng.random() < .20:
        place(str(rng.choice(KINDS)))
    return capture(panel), objects


def generate(reference, output, seed):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = json.loads((reference / "generation.json").read_text())
    rows = [json.loads(line) for line in (reference / "manifest.jsonl").read_text().splitlines()]
    if config["size"] != 512 or not config.get("fixed_setup") or not config.get("normal_only"):
        raise ValueError("Reference must contain fixed-framing 512px normal images")
    if len(rows) != config["images"] or len({row["source_case_id"] for row in rows}) != len(rows):
        raise ValueError("Expected exactly one normal image for every reference panel")
    if any(row["kind"] != "normal" for row in rows):
        raise ValueError("Reference contains a defect image")
    rng = np.random.default_rng(seed)
    counts, object_counts = Counter(), Counter()
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        for folder in ("images", "labels", "previews"):
            for split in ("train", "val", "test"):
                (root / folder / split).mkdir(parents=True)
        with (root / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for index, row in enumerate(rows):
                camera = row["camera"]
                matrix = np.asarray(camera["homography"])
                noise = np.random.default_rng(camera["seed"]).normal(0, camera["noise_std"], (512, 512, 3))

                def capture(panel):
                    return render_photo(panel, matrix, camera["blur_sigma"], noise)

                normal_panel = make_solid_panel(255, 512, row["panel"])
                normal = capture(normal_panel)
                if not np.array_equal(normal, cv2.imread(str(reference / row["image"]))):
                    raise ValueError(f"Normal replay differs for {row['image_id']}")
                for kind in ("normal", *KINDS):
                    photo, objects = (normal, []) if kind == "normal" else defect_image(
                        kind, normal_panel, normal, capture, rng,
                    )
                    image_id = f"{row['source_case_id']}_white_{kind}"
                    image = Path("images") / row["split"] / f"{image_id}.png"
                    label = Path("labels") / row["split"] / f"{image_id}.txt"
                    if kind == "normal":
                        shutil.copyfile(reference / row["image"], root / image)
                    else:
                        write_image(root / image, photo)
                    lines = []
                    preview = photo.copy()
                    for obj in objects:
                        x1, y1, x2, y2 = obj["bbox_xyxy"]
                        coordinates = ((x1 + x2) / 1024, (y1 + y2) / 1024, (x2 - x1) / 512, (y2 - y1) / 512)
                        lines.append(f"{KINDS.index(obj['kind'])} " + " ".join(f"{v:.8f}" for v in coordinates))
                        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 120, 255), 1)
                        cv2.putText(preview, obj["kind"], (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .33, (0, 80, 230), 1, cv2.LINE_AA)
                        object_counts[f"{row['split']}/{obj['kind']}"] += 1
                    (root / label).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    write_image(root / "previews" / row["split"] / f"{image_id}.png", preview)
                    record = dict(
                        schema_version="oled-white-2", image_id=image_id, source_case_id=row["source_case_id"],
                        content_id=row["content_id"], split=row["split"], kind=kind, pattern="white",
                        image=image.as_posix(), label=label.as_posix(), objects=objects,
                        bbox_xyxy=[obj["bbox_xyxy"] for obj in objects], panel=row["panel"], camera=camera,
                        is_synthetic=True,
                    )
                    manifest.write(json.dumps(record) + "\n")
                    counts[f"{row['split']}/{kind}"] += 1
                if (index + 1) % 100 == 0:
                    print(f"{index + 1}/{len(rows)} source panels exported", flush=True)
        names = "".join(f"  {index}: {kind}\n" for index, kind in enumerate(KINDS))
        (root / "dataset.yaml").write_text(
            f"path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}",
            encoding="utf-8",
        )
        result = dict(
            schema_version="oled-white-2", generator="oled-five-classes-v1", seed=seed, size=512,
            reference=str(reference.resolve()), reference_manifest_sha256=hashlib.sha256(
                (reference / "manifest.jsonl").read_bytes()).hexdigest(),
            source_cases=len(rows), images=sum(counts.values()), classes=list(KINDS),
            image_counts=dict(sorted(counts.items())), object_counts=dict(sorted(object_counts.items())),
            fixed_setup=True, pattern="white", cluster_probability=.25, secondary_defect_probability=.20,
            cluster_labeling="one box per dot", broken_line_labeling="one box per interrupted line",
            visibility=dict(minimum_peak_delta=12, minimum_pixel_delta=8, minimum_affected_pixels=2),
            numpy=np.__version__, opencv=cv2.__version__, seconds=time.perf_counter() - started,
        )
        (root / "generation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        root.rename(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    generate(args.reference, args.output, args.seed)
