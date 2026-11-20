"""Audit materialized OLED tiles, paired scenes, balance and renderer invariants."""

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


def math_checks(source):
    small = [2100, 1200, 2203, 1311]
    large = [2081, 1172, 2251, 1350]
    spec = dict(kind="color_dot", x=690, y=393, native_width=2, native_height=1,
                channels="RG", channel_strengths={"R": .1, "G": .2},
                channel_profiles={"R": dict(profile="constant"), "G": dict(profile="constant")})
    for specs in ([], [spec]):
        a, b = r.render_window(source, specs, small), r.render_window(source, specs, large)
        assert np.array_equal(a, b[small[1] - large[1]:small[3] - large[1],
                                    small[0] - large[0]:small[2] - large[0]])
    a, b = r.noise_window(source, small), r.noise_window(source, large)
    assert np.array_equal(a, b[small[1] - large[1]:small[3] - large[1],
                                small[0] - large[0]:small[2] - large[0]])
    x = np.asarray([[0., 2., 4.]])
    y = np.asarray([[0.]])
    line = dict(kind="horizontal_line", x=0, y=0, native_width=5, native_height=1,
                channel_strengths={"R": .5}, channel_profiles={})
    for profile, expected in (("rising", [.025, .2625, .5]), ("falling", [.5, .2625, .025]),
                              ("center", [.025, .5, .025])):
        line["channel_profiles"] = {"R": dict(profile=profile)}
        actual = r.attenuation(line, "R", np.ones((1, 3)), x, y)
        np.testing.assert_allclose(actual[0], expected, atol=1e-12)


def check(root, deep_sources=20):
    config = json.loads((root / "generation.json").read_text())
    assert hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest() == config["manifest_sha256"]
    assert hashlib.sha256((root / "source_cases.jsonl").read_bytes()).hexdigest() == config["source_cases_sha256"]
    assert hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest() == config["renderer_sha256"]
    assert hashlib.sha256((root / "renderer_snapshot.py").read_bytes()).hexdigest() == config["renderer_sha256"]
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    sources = {source["source_case_id"]: source for source in map(json.loads, (root / "source_cases.jsonl").read_text().splitlines())}
    assert len(sources) == config["sources"] and len(records) == config["sources"] * 7 == config["images"]
    math_checks(next(iter(sources.values())))
    groups = defaultdict(list)
    for row in records:
        groups[row["source_case_id"]].append(row)
    deep = set(np.asarray(sorted(groups))[np.linspace(0, len(groups) - 1, min(deep_sources, len(groups)), dtype=int)])
    hashes, ids = set(), set()
    image_counts, object_counts, mix_counts = Counter(), Counter(), Counter()
    patterns, tiers, forms, thicknesses = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    normal_summaries, clipped = [], 0
    for group_index, (sid, rows) in enumerate(sorted(groups.items())):
        source = sources[sid]
        assert len(rows) == 7 and {row["kind"] for row in rows} == {"normal", *r.KINDS, "mixed"}
        assert {row["split"] for row in rows} == {source["split"]}
        assert all(row["tile_xyxy"] == source["tile_xyxy"] for row in rows)
        master = next(row for row in rows if row["kind"] == "normal")
        normal = cv2.imread(str(root / master["image"]))
        assert normal is not None
        normal_summaries.append(dict(source_case_id=sid, split=source["split"],
                                     level=source["panel"]["level"], mean_bgr=normal.mean(axis=(0, 1)).tolist()))
        for row in rows:
            assert row["image_id"] not in ids
            ids.add(row["image_id"])
            path = root / row["image"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
            image = cv2.imread(str(path))
            assert image is not None and image.shape == (r.TILE, r.TILE, 3)
            digest = r.pixel_hash(image)
            assert digest == row["pixel_sha256"] and digest not in hashes
            hashes.add(digest)
            assert image.max() < 255 and np.all(image <= normal)
            assert row["normal_image"] == master["image"]
            assert row["is_synthetic"] and not row["calibrated_to_real_hardware"]
            assert (root / "previews" / row["split"] / (row["image_id"] + ".png")).is_file()
            labels = (root / row["label"]).read_text().splitlines()
            assert len(labels) == len(row["objects"]) == len(row["scene_objects"]) == len(row["bbox_xyxy"])
            assert bool(labels) == (row["kind"] != "normal")
            delta = cv2.absdiff(normal, image).max(axis=2)
            allowed = np.zeros((r.TILE, r.TILE), bool)
            channels = set()
            for label, obj, spec in zip(labels, row["objects"], row["scene_objects"], strict=True):
                cls, x, y, w, h = map(float, label.split())
                assert cls == r.KINDS.index(obj["kind"]) and obj["kind"] == spec["kind"]
                box = obj["bbox_xyxy"]
                restored = [(x - w / 2) * r.TILE, (y - h / 2) * r.TILE,
                            (x + w / 2) * r.TILE, (y + h / 2) * r.TILE]
                assert max(abs(a - b) for a, b in zip(restored, box)) < .001
                x1, y1, x2, y2 = box
                assert 0 <= x1 < x2 <= r.TILE and 0 <= y1 < y2 <= r.TILE
                assert delta[y1:y2, x1:x2].max() >= 2
                native = r.native_bounds(spec)
                tx, ty = source["tile_xyxy"][:2]
                physical = [r.BORDER + value * r.SCALE + (-3 if i < 2 else 3)
                            - (tx if i % 2 == 0 else ty) for i, value in enumerate(native)]
                px1, py1, px2, py2 = physical
                allowed[max(0, py1):min(r.TILE, py2), max(0, px1):min(r.TILE, px2)] = True
                channels.update(spec["channels"])
                assert set(spec["channel_strengths"]) == set(spec["channel_profiles"]) == set(spec["channels"])
                assert all(0 < value <= (.10 if spec["kind"] == "dark_region" else 1)
                           for value in spec["channel_strengths"].values())
                if spec["kind"].endswith("dot"):
                    assert 1 <= spec["native_width"] <= 3 and 1 <= spec["native_height"] <= 3
                if spec["kind"] in ("horizontal_line", "vertical_line"):
                    assert spec["thickness"] in (1, 2)
                clipped += bool(obj["attributes"]["clipped_by_tile"])
                object_counts[obj["kind"]] += 1
            assert not delta[~allowed].any()
            for channel, index in r.CHANNELS.items():
                if channel not in channels:
                    assert np.array_equal(normal[:, :, index], image[:, :, index])
            if row["kind"] == "mixed":
                unique = tuple(sorted({obj["kind"] for obj in row["objects"]}))
                assert len(unique) in (2, 3)
                mix_counts["+".join(unique)] += 1
            elif row["scene_objects"]:
                first = row["scene_objects"][0]
                key = row["split"] + "/" + row["kind"]
                patterns[key][first["channel_pattern"]] += 1
                tiers[key][first["tier"]] += 1
                if first["kind"] in ("horizontal_line", "vertical_line"):
                    forms[key][first["line_form"]] += 1
                    thicknesses[key][str(first["thickness"])] += 1
            if sid in deep:
                expected = r.render_window(source, row["scene_objects"], source["tile_xyxy"])
                assert np.array_equal(expected, image), f"Scene reproduction differs: {row['image_id']}"
                for obj, spec in zip(row["objects"], row["scene_objects"], strict=True):
                    isolated = r.render_window(source, [spec], source["tile_xyxy"])
                    deficit = cv2.absdiff(normal, isolated).max(axis=2)
                    yy, xx = np.nonzero(deficit >= 2)
                    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
                    assert box == obj["bbox_xyxy"] and int(deficit.max()) == obj["attributes"]["peak_delta"]
            image_counts[row["split"] + "/" + row["kind"]] += 1
        if (group_index + 1) % 100 == 0:
            print(f"Validated {group_index + 1}/{len(groups)} source families ({(group_index + 1) * 7} images)", flush=True)
    assert dict(sorted(image_counts.items())) == config["image_counts"]
    assert dict(object_counts) == config["object_counts"]
    assert len(list((root / "images").rglob("*.png"))) == len(records)
    assert len(list((root / "labels").rglob("*.txt"))) == len(records)
    for distribution in (patterns, tiers, forms):
        for counts in distribution.values():
            assert max(counts.values()) - min(counts.values()) <= 1
    report = dict(images=len(records), source_cases=len(sources), duplicate_images=0, source_case_leakage=False,
                  hashes_verified=True, labels_verified=True, paired_pixels_verified=True,
                  unaffected_channels_verified=True, outside_support_verified=True, no_saturated_images=True,
                  coordinate_stable_renderer_verified=True, exactly_reproduced_source_families=len(deep),
                  image_counts=dict(sorted(image_counts.items())), object_counts=dict(object_counts),
                  mixed_combinations=dict(sorted(mix_counts.items())), channel_patterns=dict(patterns),
                  strength_tiers=dict(tiers), line_forms=dict(forms), line_thicknesses=dict(thicknesses),
                  truncated_tile_objects=clipped, normal_levels=[min(x["level"] for x in normal_summaries),
                                                               max(x["level"] for x in normal_summaries)],
                  calibrated_to_real_hardware=False)
    (root / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (root / "normal_summary.json").write_text(json.dumps(normal_summaries, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in
                      ("image_counts", "channel_patterns", "strength_tiers", "line_forms", "line_thicknesses")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--deep-sources", type=int, default=20)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    check(args.dataset, args.deep_sources)
