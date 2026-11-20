"""Materialize a balanced 7,000-tile OLED corpus with reproducible full scenes."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai import paired_renderer as render

KINDS = render.KINDS
PATTERNS = ("R", "G", "B", "RG", "RB", "GB", "RGB", "RGB_equal")
PROFILES = ("constant", "rising", "falling", "center")


def balanced(values, count, rng):
    result = []
    while len(result) < count:
        result.extend(values[int(i)] for i in rng.permutation(len(values)))
    return result[:count]


def make_plan(count, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 881]))
    indices = rng.permutation(count)
    boundaries = (int(count * .7), int(count * .85))
    plan = {}
    for split, selection in zip(("train", "val", "test"), np.split(indices, boundaries), strict=True):
        n = len(selection)
        assignments = {}
        for kind in KINDS:
            assignments[kind] = dict(
                pattern=balanced(PATTERNS[:-1] if kind == "color_dot" else
                                 ("RGB_equal",) if kind == "dark_dot" else PATTERNS, n, rng),
                tier=balanced(("weak", "medium", "strong"), n, rng),
                form=balanced(("full", "partial", "broken"), n, rng),
                thickness=balanced((1, 1, 1, 2), n, rng),
            )
        mixes = []
        for size, amount in ((2, n // 2), (3, n - n // 2)):
            mixes.extend(balanced(tuple(itertools.combinations(KINDS, size)), amount, rng))
        rng.shuffle(mixes)
        for local, source_index in enumerate(selection):
            plan[int(source_index)] = dict(split=split, mix=list(mixes[local]), assignments={
                kind: {key: values[local] for key, values in settings.items()}
                for kind, settings in assignments.items()})
    return plan


def source_config(index, split, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 173]))
    tint = rng.uniform(-.0015, .0015, 3)
    tint -= tint.mean()
    x = int(rng.integers(0, render.PHOTO_W - render.TILE + 1))
    y = int(rng.integers(0, render.PHOTO_H - render.TILE + 1))
    if rng.random() < .12:
        x = int(rng.choice((0, render.PHOTO_W - render.TILE)))
    if rng.random() < .12:
        y = int(rng.choice((0, render.PHOTO_H - render.TILE)))
    source = dict(source_case_id=f"panel{index:04d}", split=split,
                  renderer_version=render.VERSION, noise_key=int(rng.bit_generator.random_raw()), noise_std=.6,
                  tile_xyxy=[x, y, x + render.TILE, y + render.TILE],
                  panel=dict(level=float(rng.uniform(225, 240)), tint_bgr=(1 + tint).tolist(),
                             slope_x=float(rng.uniform(-.004, .004)), slope_y=float(rng.uniform(-.004, .004)),
                             shading_amplitude=float(rng.uniform(.0005, .0025)),
                             shading_x=float(rng.uniform(-.2, .2)), shading_y=float(rng.uniform(-.2, .2)),
                             shading_width=float(rng.uniform(.25, .55)), shading_height=float(rng.uniform(.25, .55))))
    source["content_id"] = render.canonical_hash(source)
    return source


def rates(kind, pattern, tier, rng):
    ranges = ((.08, .20), (.20, .55), (.55, 1.0)) if kind.endswith("dot") else (
        ((.01, .025), (.025, .06), (.06, .10)) if kind == "dark_region" else
        ((.06, .12), (.12, .35), (.35, .65)))
    low, high = ranges[("weak", "medium", "strong").index(tier)]
    peak = float(rng.uniform(low, high))
    if kind == "dark_dot" and tier == "strong" and rng.random() < .25:
        peak = 1.
    channels = "RGB" if pattern == "RGB_equal" else pattern
    if pattern == "RGB_equal" or len(channels) == 1:
        values = [peak] * len(channels)
    else:
        values = rng.uniform(.25, .85, len(channels)) * peak
        values[int(rng.integers(len(channels)))] = peak
    return channels, {channel: float(value) for channel, value in zip(channels, values, strict=True)}


def new_spec(kind, assignment, rng, window, mixed=False):
    x1 = max(0, int(np.ceil((window[0] + 15 - render.BORDER) / render.SCALE)))
    y1 = max(0, int(np.ceil((window[1] + 15 - render.BORDER) / render.SCALE)))
    x2 = min(render.PANEL_W, int(np.floor((window[2] - 15 - render.BORDER) / render.SCALE)))
    y2 = min(render.PANEL_H, int(np.floor((window[3] - 15 - render.BORDER) / render.SCALE)))
    channels, strengths = rates(kind, assignment["pattern"], assignment["tier"], rng)
    spec = dict(kind=kind, channels=channels, channel_pattern=assignment["pattern"],
                channel_strengths=strengths, strength=max(strengths.values()), tier=assignment["tier"],
                channel_profiles={}, coordinate_units="synthetic_panel_pixels")
    if kind.endswith("dot"):
        size = int(rng.choice((1, 2, 3), p=(.7, .2, .1)))
        spec.update(x=int(rng.integers(x1, x2 - size)), y=int(rng.integers(y1, y2 - size)),
                    native_width=size, native_height=size, cluster_id=None)
        spec["channel_profiles"] = {channel: dict(profile="constant") for channel in channels}
    elif kind == "dark_region":
        rx = int(rng.integers(42, 85 if mixed else 122))
        ry = int(rng.integers(30, 64 if mixed else 88))
        spec.update(radius_x=rx, radius_y=ry, angle=float(rng.uniform(-65, 65)),
                    x=int(rng.integers(x1, x2)), y=int(rng.integers(y1, y2)))
        shared = float(rng.uniform(.7, 1.8))
        spec["channel_profiles"] = {channel: dict(profile="power", exponent=shared if assignment["pattern"] == "RGB_equal"
                                                   else float(rng.uniform(.65, 2.2))) for channel in channels}
        bounds = render.native_bounds(spec)
        dx1, dy1 = spec["x"] - bounds[0], spec["y"] - bounds[1]
        dx2, dy2 = bounds[2] - spec["x"], bounds[3] - spec["y"]
        if x1 + dx1 >= x2 - dx2 or y1 + dy1 >= y2 - dy2:
            return None
        spec["x"] = int(rng.integers(x1 + dx1, x2 - dx2))
        spec["y"] = int(rng.integers(y1 + dy1, y2 - dy2))
        render.set_region_normalizer(spec)
    else:
        horizontal = kind == "horizontal_line"
        form, thickness = assignment["form"], assignment["thickness"]
        if form == "full":
            length = render.PANEL_W if horizontal else render.PANEL_H
            start = 0
        else:
            limit = (x2 - x1 if horizontal else y2 - y1) - 5
            length = int(rng.integers(45 if form == "partial" else 100, min(limit, 285)))
            start = int(rng.integers(x1 if horizontal else y1,
                                     (x2 if horizontal else y2) - length))
        cross = int(rng.integers(y1 if horizontal else x1, (y2 if horizontal else x2) - thickness))
        spec.update(x=start if horizontal else cross, y=cross if horizontal else start,
                    native_width=length if horizontal else thickness,
                    native_height=thickness if horizontal else length,
                    line_form=form, thickness=thickness, length=length,
                    orientation="horizontal" if horizontal else "vertical", broken=form == "broken")
        if form == "broken":
            spec.update(segment=int(rng.integers(14, 55)), gap=int(rng.integers(5, 18)))
        shared = str(rng.choice(PROFILES))
        spec["channel_profiles"] = {channel: dict(profile=shared if assignment["pattern"] == "RGB_equal"
                                                   else str(rng.choice(PROFILES))) for channel in channels}
    bounds = render.native_bounds(spec)
    spec["native_bbox_xyxy"] = bounds
    return spec


def compatible(spec, occupied):
    box = render.native_bounds(spec)
    for previous in occupied:
        if {spec["kind"], previous["kind"]} == {"horizontal_line", "vertical_line"}:
            continue  # Crossed horizontal/vertical lines remain two labeled objects.
        other = render.native_bounds(previous)
        if min(box[2], other[2]) + 3 > max(box[0], other[0]) and min(box[3], other[3]) + 3 > max(box[1], other[1]):
            return False
    return True


def sample_specs(kinds, assignment, source, rng, mixed=False):
    # Place broad regions first to leave room for small, separate defects.
    ordered = sorted(kinds, key=lambda kind: (kind != "dark_region", kind.endswith("dot")))
    for _ in range(100):
        result, failed = [], False
        for kind in ordered:
            settings = assignment[kind] if not mixed else dict(
                pattern="RGB_equal" if kind == "dark_dot" else str(rng.choice(PATTERNS[:-1] if kind == "color_dot" else PATTERNS)),
                tier=str(rng.choice(("weak", "medium", "strong"))), form=str(rng.choice(("full", "partial", "broken"))),
                thickness=int(rng.choice((1, 1, 1, 2))))
            for _ in range(120):
                candidate = new_spec(kind, settings, rng, source["tile_xyxy"], mixed)
                if candidate is not None and compatible(candidate, result):
                    result.append(candidate)
                    break
            else:
                failed = True
                break
            if kind.endswith("dot") and rng.random() < .25:
                group_id = f"cluster-{len(result)}"
                candidate["cluster_id"] = group_id
                for _ in range(int(rng.integers(1, 4))):
                    for _ in range(30):
                        extra = dict(candidate, x=candidate["x"] + int(rng.integers(-16, 17)),
                                     y=candidate["y"] + int(rng.integers(-16, 17)))
                        bounds = render.native_bounds(extra)
                        camera = [render.BORDER + value * render.SCALE for value in bounds]
                        tile = source["tile_xyxy"]
                        if (bounds[0] >= 0 and bounds[1] >= 0 and bounds[2] <= render.PANEL_W and bounds[3] <= render.PANEL_H
                                and camera[0] >= tile[0] + 10 and camera[1] >= tile[1] + 10
                                and camera[2] <= tile[2] - 10 and camera[3] <= tile[3] - 10 and compatible(extra, result)):
                            extra["native_bbox_xyxy"] = bounds
                            result.append(extra)
                            break
        if not failed:
            return result
    raise RuntimeError(f"Could not place {kinds} for {source['source_case_id']}")


def object_label(source, spec, base, normal):
    window = source["tile_xyxy"]
    box, loss = render.effect_patch(source, [spec], window)
    x1, y1, x2, y2 = box
    px, py = x1 - window[0], y1 - window[1]
    view = base[py:py + y2 - y1, px:px + x2 - x1]
    isolated = render.quantize(view - loss)
    original = normal[py:py + y2 - y1, px:px + x2 - x1]
    delta = cv2.absdiff(original, isolated).max(axis=2)
    ys, xs = np.nonzero(delta >= 2)
    if not len(xs):
        return None
    visible = [px + int(xs.min()), py + int(ys.min()), px + int(xs.max()) + 1, py + int(ys.max()) + 1]
    physical = [render.BORDER + value * render.SCALE + (-3 if i < 2 else 3)
                for i, value in enumerate(render.native_bounds(spec))]
    attributes = dict(spec, peak_delta=int(delta.max()), visible_pixels=int((delta >= 2).sum()),
                      clipped_by_tile=physical[0] < window[0] or physical[1] < window[1]
                      or physical[2] > window[2] or physical[3] > window[3],
                      visibility_threshold_dn=2)
    return dict(kind=spec["kind"], bbox_xyxy=visible, attributes=attributes)


def family(root, index, settings, seed):
    source = source_config(index, settings["split"], seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 357]))
    window = source["tile_xyxy"]
    base = render.base_window(source, window)
    normal = render.quantize(base)
    split, sid = source["split"], source["source_case_id"]
    normal_path = f"images/{split}/{sid}_normal.png"
    rows = []
    for kind in ("normal", *KINDS, "mixed"):
        objects, specs = [], []
        if kind != "normal":
            for attempt in range(80):
                specs = sample_specs(settings["mix"] if kind == "mixed" else [kind], settings["assignments"],
                                     source, rng, mixed=kind == "mixed")
                objects = [object_label(source, spec, base, normal) for spec in specs]
                if all(obj is not None for obj in objects):
                    break
            else:
                raise RuntimeError(f"Could not render visible {kind} for {sid}")
        image = normal if kind == "normal" else render.render_from_base(source, specs, window, base)
        if not np.all(image <= normal) or image.max() >= 255:
            raise ValueError("Unexpected pixel brightening or saturation")
        scene_id = f"{sid}_{kind}"
        image_path, label_path = f"images/{split}/{scene_id}.png", f"labels/{split}/{scene_id}.txt"
        png_hash = render.write_png(root / image_path, image)
        lines = []
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            values = ((x1 + x2) / (2 * render.TILE), (y1 + y2) / (2 * render.TILE),
                      (x2 - x1) / render.TILE, (y2 - y1) / render.TILE)
            # 1024px tiles need 11 fractional decimal places for exact half-pixel centers.
            lines.append(str(KINDS.index(obj["kind"])) + " " + " ".join(f"{value:.12f}" for value in values))
        path = root / label_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(dict(schema_version="oled-white-2", image_id=scene_id, scene_id=scene_id,
                         source_case_id=sid, content_id=source["content_id"], split=split, kind=kind,
                         image=image_path, label=label_path, pattern="white", objects=objects,
                         bbox_xyxy=[obj["bbox_xyxy"] for obj in objects], normal_image=normal_path,
                         scene_objects=specs, tile_xyxy=window, image_width=render.TILE, image_height=render.TILE,
                         full_photo_width=render.PHOTO_W, full_photo_height=render.PHOTO_H,
                         image_sha256=png_hash, pixel_sha256=render.pixel_hash(image),
                         is_synthetic=True, calibrated_to_real_hardware=False))
    return dict(source=source, records=rows)


def generate(output, count, seed):
    if count < 20:
        raise ValueError("At least 20 source panels are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    stage = output.with_name("." + output.name + ".building")
    stage.mkdir(parents=True, exist_ok=True)
    renderer_path = Path(render.__file__)
    requested = dict(sources=count, seed=seed, renderer_sha256=hashlib.sha256(renderer_path.read_bytes()).hexdigest(),
                     generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    request_path = stage / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != requested:
        raise ValueError("Staging data uses a different generator, seed or source count")
    request_path.write_text(json.dumps(requested, indent=2) + "\n")
    shutil.copyfile(Path(__file__), stage / "generator_snapshot.py")
    shutil.copyfile(renderer_path, stage / "renderer_snapshot.py")
    (stage / "families").mkdir(exist_ok=True)
    plan = make_plan(count, seed)
    started = time.perf_counter()
    for index in range(count):
        journal = stage / "families" / f"panel{index:04d}.json"
        if journal.exists():
            previous = json.loads(journal.read_text())
            if any(not (stage / row["image"]).is_file()
                   or hashlib.sha256((stage / row["image"]).read_bytes()).hexdigest() != row["image_sha256"]
                   for row in previous["records"]):
                journal.unlink()  # Regenerate only this unfinished/corrupt generated family.
        if not journal.exists():
            if shutil.disk_usage(stage).free < 3 * 1024 ** 3:
                raise OSError("Less than 3 GiB free; resumable staging data retained")
            data = family(stage, index, plan[index], seed)
            temporary = journal.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False) + "\n")
            temporary.replace(journal)
        if (index + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"{index + 1}/{count} source panels, {(index + 1) * 7} images; {elapsed:.1f}s", flush=True)
        if index == min(4, count - 1):
            sampled = [row for sample in range(index + 1)
                       for row in json.loads((stage / "families" / f"panel{sample:04d}.json").read_text())["records"]]
            used = sum((stage / row["image"]).stat().st_size for row in sampled)
            projected = used * count / (index + 1)
            remaining = count - len(list((stage / "families").glob("panel*.json")))
            if projected * remaining / count > shutil.disk_usage(stage).free - 3 * 1024 ** 3:
                raise OSError(f"Estimated corpus {projected / 1024**3:.2f} GiB exceeds available space")
    image_counts, object_counts, rows, sources = Counter(), Counter(), [], []
    for index in range(count):
        data = json.loads((stage / "families" / f"panel{index:04d}.json").read_text())
        sources.append(data["source"])
        for row in data["records"]:
            rows.append(row)
            image_counts[f"{row['split']}/{row['kind']}"] += 1
            object_counts.update(obj["kind"] for obj in row["objects"])
    (stage / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (stage / "source_cases.jsonl").write_text("".join(json.dumps(source) + "\n" for source in sources))
    names = "".join(f"  {i}: {kind}\n" for i, kind in enumerate(KINDS))
    (stage / "dataset.yaml").write_text(f"path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}")
    (stage / "previews").mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        path = stage / "previews" / split
        if not path.exists():
            path.symlink_to(f"../images/{split}", target_is_directory=True)
    if hashlib.sha256(renderer_path.read_bytes()).hexdigest() != requested["renderer_sha256"]:
        raise ValueError("Renderer source changed during generation; original snapshot retained")
    config = dict(generator="paired-oled-full-v2", renderer_version=render.VERSION,
                  renderer_sha256=requested["renderer_sha256"], seed=seed, sources=count, images=len(rows),
                  generator_sha256=requested["generator_sha256"],
                  classes=list(KINDS), image_counts=dict(sorted(image_counts.items())), object_counts=dict(object_counts),
                  image_storage="1024px native-resolution tiles; complete full-scene recipes retained",
                  full_photo_size=[render.PHOTO_W, render.PHOTO_H], tile_size=render.TILE,
                  synthetic_panel_size=[render.PANEL_W, render.PANEL_H], camera_scale=render.SCALE,
                  fixed_framing=True, paired_noise=True, noise_generator="coordinate-splitmix64-boxmuller-v1",
                  noise_std=.6, blur_sigma=.65, blur_kernel_size=7, visibility_threshold_dn=2,
                  split_unit="source_case_id", calibrated_to_real_hardware=False,
                  region_max_digital_loss=.10, numpy=np.__version__, opencv=cv2.__version__,
                  source_cases_sha256=hashlib.sha256((stage / "source_cases.jsonl").read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((stage / "manifest.jsonl").read_bytes()).hexdigest(),
                  seconds=time.perf_counter() - started)
    (stage / "generation.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    stage.rename(output)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/oled-white-paired-v2"))
    parser.add_argument("--sources", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    generate(args.output, args.sources, args.seed)
