"""Generate matched gray levels, RGB image pairs, and visible-effect masks."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from generate_oled_full_dataset import KINDS, make_plan, render, sample_specs, source_config

LEVELS = (48, 96, 144, 192)
KINDS_OF_IMAGE = ("normal", *KINDS, "mixed")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def visible_objects(source, specs, base, normal):
    x, y, _, _ = window = source["tile_xyxy"]
    mask = np.zeros(normal.shape[:2], np.uint8)
    objects = []
    for spec in specs:
        (x1, y1, x2, y2), loss = render.effect_patch(source, [spec], window)
        slices = np.s_[y1-y:y2-y, x1-x:x2-x]
        delta = cv2.absdiff(normal[slices], render.quantize(base[slices] - loss)).max(axis=2)
        visible = delta >= 2
        yy, xx = np.nonzero(visible)
        if not len(xx):
            return None
        bbox = [x1-x+int(xx.min()), y1-y+int(yy.min()), x1-x+int(xx.max())+1, y1-y+int(yy.max())+1]
        kind = spec["kind"]
        objects.append(dict(kind=kind, bbox_xyxy=bbox,
                            attributes=dict(spec, peak_delta=int(delta.max()), visible_pixels=int(visible.sum()))))
        mask[slices] |= visible.astype(np.uint8) * (1 << KINDS.index(kind))
    return objects, mask


def generate(output, count, seed):
    if count < 100:
        raise ValueError("Use at least 100 base panels for the four-level split.")
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(2)
    stage = output.with_name("." + output.name + ".building")
    stage.mkdir(parents=True)
    plan = make_plan(count, seed)
    rows, sources, masks = [], [], []
    hashes = set()
    rejected = 0
    for index in range(count):
        settings = plan[index]
        split = settings["split"]
        source = source_config(index, split, seed)
        panel = f"graypanel{index:04d}"
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 541]))
        window = source["tile_xyxy"]
        exposure = source["panel"]["level"] / 232.5
        views = []
        for level in LEVELS:
            cfg = dict(source, source_case_id=f"{panel}_g{level:03d}", panel_id=panel, gray_level=level,
                       panel=dict(source["panel"], level=level * exposure))
            cfg.pop("content_id")
            cfg["content_id"] = render.canonical_hash(cfg)
            base = render.base_window(cfg, window)
            views.append((cfg, base, render.quantize(base)))
            sources.append(cfg)
        for kind in KINDS_OF_IMAGE:
            for _ in range(250):
                specs = [] if kind == "normal" else sample_specs(
                    settings["mix"] if kind == "mixed" else [kind], settings["assignments"],
                    source, rng, mixed=kind == "mixed")
                annotations = [visible_objects(cfg, specs, base, normal) for cfg, base, normal in views]
                if all(annotation is not None for annotation in annotations):
                    break
                rejected += 1
            else:
                raise RuntimeError(f"No visible {kind} recipe for {panel}")
            for (cfg, base, normal), (objects, mask) in zip(views, annotations, strict=True):
                sid = cfg["source_case_id"]
                name = f"{sid}_{kind}"
                image = render.render_from_base(cfg, specs, window, base) if specs else normal
                assert np.all(image <= normal) and bool(mask.any()) == bool(specs)
                assert not np.any((cv2.absdiff(normal, image).max(axis=2) < 2) & (mask != 0))
                raw_sha = render.pixel_hash(image)
                if raw_sha in hashes:
                    raise ValueError(f"Duplicate RGB image: {name}")
                hashes.add(raw_sha)
                image_path = f"images/{split}/{name}.png"
                mask_path = f"masks/{split}/{name}.png"
                image_sha = render.write_png(stage / image_path, image)
                mask_sha = render.write_png(stage / mask_path, mask)
                rows.append(dict(image_id=name, source_case_id=sid, panel_id=panel, gray_level=cfg["gray_level"],
                                 split=split, kind=kind, image=image_path, objects=objects, scene_objects=specs,
                                 normal_image=f"images/{split}/{sid}_normal.png", tile_xyxy=window,
                                 image_sha256=image_sha, pixel_sha256=raw_sha))
                masks.append(dict(image_id=name, mask=mask_path, mask_sha256=mask_sha))
        if (index + 1) % 10 == 0:
            print(f"Gray dataset: {index + 1}/{count} panels, {len(rows)} tiles and masks", flush=True)

    # Keep each source's seven images adjacent for reproducible full-scene replay.
    rows.sort(key=lambda row: (row["source_case_id"], KINDS_OF_IMAGE.index(row["kind"])))
    write_rows(stage / "manifest.jsonl", rows)
    write_rows(stage / "source_cases.jsonl", sources)
    write_rows(stage / "mask_manifest.jsonl", masks)
    from hashlib import sha256
    renderer_sha = sha256(Path(render.__file__).read_bytes()).hexdigest()
    for file in (Path(__file__), Path(render.__file__)):
        shutil.copyfile(file, stage / file.name)
    counts = Counter(f"{r['split']}/g{r['gray_level']:03d}/{r['kind']}" for r in rows)
    record = dict(images=len(rows), panels=count, gray_levels=LEVELS, seed=seed, renderer_sha256=renderer_sha,
                  tile_size=render.TILE, full_photo_size=(render.PHOTO_W, render.PHOTO_H),
                  split_unit="panel_id", image_counts=counts, rejected_below_2dn_recipes=rejected,
                  mask="uint8 bits 1,2,4,8,16; isolated max-RGB drop ≥2 DN",
                  visibility_policy="Resample recipes with an invisible object at any level; keep the same final objects across levels.",
                  domain="synthetic camera digital RGB levels; no real hardware calibration",
                  dataset_manifest_sha256=sha256((stage / "manifest.jsonl").read_bytes()).hexdigest(),
                  mask_manifest_sha256=sha256((stage / "mask_manifest.jsonl").read_bytes()).hexdigest(),
                  numpy=np.__version__, opencv=cv2.__version__)
    (stage / "segmentation.json").write_text(json.dumps(record, indent=2) + "\n")
    stage.rename(output)
    print(f"Completed {len(rows)} tiles and masks; rejected {rejected} below-2-DN recipes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/oled-gray-paired-v1"))
    parser.add_argument("--panels", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    generate(args.output, args.panels, args.seed)
