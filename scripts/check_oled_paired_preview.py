"""Check paired preview pixels, unchanged channels, source links and gradient math."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from generate_oled_paired_preview import channel_field, PANEL_H, PANEL_W, BORDER, SCALE


def check(root):
    config = json.loads((root / "generation.json").read_text())
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    pairs = json.loads((root / "pairs.json").read_text())
    by_id = {row["image_id"]: row for row in records}
    assert len(by_id) == len(records)
    assert len(records) == config["normal_images"] + config["defect_images"]
    assert len(pairs) == config["defect_images"]
    digests = set()
    for row in records:
        image_path = root / row["image"]
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        assert digest == row["image_sha256"] and digest not in digests
        digests.add(digest)
        assert row["review_only"] and row["split"] == "preview"
        w, h = row["image_width"], row["image_height"]
        rows = (root / row["label"]).read_text().splitlines()
        assert len(rows) == len(row["objects"])
        for label, obj in zip(rows, row["objects"], strict=True):
            cls, x, y, bw, bh = map(float, label.split())
            assert cls == config["classes"].index(obj["kind"])
            assert 0 <= x - bw / 2 < x + bw / 2 <= 1
            assert 0 <= y - bh / 2 < y + bh / 2 <= 1
            restored = [(x - bw / 2) * w, (y - bh / 2) * h,
                        (x + bw / 2) * w, (y + bh / 2) * h]
            assert max(abs(a - b) for a, b in zip(restored, obj["bbox_xyxy"])) < .001
    summaries = []
    for pair in pairs:
        row = by_id[pair["name"]]
        source = by_id[row["normal_image_id"]]
        assert source["source_case_id"] == row["source_case_id"] == pair["source_case_id"]
        assert source["image"] == pair["normal"] and source["normal"] == row["normal"]
        normal, defect = (cv2.imread(str(root / pair[key])) for key in ("normal", "defect"))
        assert normal.shape == defect.shape == (row["image_height"], row["image_width"], 3)
        assert defect.max() < 255 and normal.max() < 255
        assert np.all(defect <= normal)
        delta = cv2.absdiff(normal, defect).max(axis=2)
        mask = cv2.imread(str(root / "masks" / f"{pair['name']}.png"), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(mask > 0, delta >= 2)
        attributes = pair["attributes"]
        assert int(delta.max()) == attributes["peak_photo_delta"]
        assert int((delta >= 2).sum()) == attributes["visible_pixels"]
        ys, xs = np.nonzero(delta >= 2)
        assert row["bbox_xyxy"] == [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]]
        x1, y1, x2, y2 = attributes["observed_support_bbox"]
        assert not delta[:y1].any() and not delta[y2:].any()
        assert not delta[:, :x1].any() and not delta[:, x2:].any()
        for channel, index in (("B", 0), ("G", 1), ("R", 2)):
            if channel not in attributes["channels"]:
                assert np.array_equal(normal[:, :, index], defect[:, :, index])
        if attributes["kind"].endswith("dot"):
            assert attributes["native_support_pixels"] == 1
        summary = dict(name=pair["name"], peak_delta=int(delta.max()),
                       visible_pixels=int(mask.astype(bool).sum()), channels=attributes["channels"])
        if attributes["kind"] in ("horizontal_line", "vertical_line") and attributes.get("channel_profiles"):
            horizontal = attributes["kind"] == "horizontal_line"
            length = attributes["native_width" if horizontal else "native_height"]
            samples = []
            for fraction in (.1, .5, .9):
                offset = round((length - 1) * fraction)
                x = BORDER + (attributes["x"] + (offset if horizontal else 0)) * SCALE + SCALE // 2
                y = BORDER + (attributes["y"] + (0 if horizontal else offset)) * SCALE + SCALE // 2
                loss = normal[y, x].astype(int) - defect[y, x].astype(int)
                samples.append({channel: int(loss[index]) for channel, index in (("R", 2), ("G", 1), ("B", 0))})
            for channel, settings in attributes["channel_profiles"].items():
                a, b, c = (sample[channel] for sample in samples)
                if settings["profile"] == "rising":
                    assert a <= b + 1 and b <= c + 1
                elif settings["profile"] == "falling":
                    assert a + 1 >= b and b + 1 >= c
                elif settings["profile"] == "center":
                    assert b + 1 >= max(a, c)
            summary["channel_loss_dn_at_10_50_90_percent"] = samples
        summaries.append(summary)
    # Independent endpoint/midpoint checks, including per-channel width profiles.
    field = np.ones((PANEL_H, PANEL_W), np.float32)
    spec = dict(kind="horizontal_line", x=0, y=0, native_width=5, native_height=1)
    for profile, expected in (("rising", [.05, .525, 1]), ("falling", [1, .525, .05]),
                              ("center", [.05, 1, .05])):
        actual = channel_field(field, spec, dict(profile=profile))[0, [0, 2, 4]]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    values = np.asarray([[0, .25, 1]], dtype=np.float32)
    broad = channel_field(values, {}, dict(profile="power", exponent=.7))
    narrow = channel_field(values, {}, dict(profile="power", exponent=1.6))
    assert broad[0, 1] > values[0, 1] > narrow[0, 1]
    assert broad[0, 0] == narrow[0, 0] == 0 and broad[0, 2] == narrow[0, 2] == 1
    for settings in (dict(profile="unknown"), dict(profile="power", exponent=-1)):
        try:
            channel_field(field, spec, settings)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid channel profile accepted")
    report = dict(images=len(records), pairs=len(pairs), source_cases=config["source_cases"],
                  unique_image_hashes=len(digests), labels_verified=True, paired_masks_verified=True,
                  unaffected_channels_identical=True, outside_support_identical=True,
                  gradient_math_verified=True, review_only=True, cases=summaries)
    (root / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    check(parser.parse_args().root)
