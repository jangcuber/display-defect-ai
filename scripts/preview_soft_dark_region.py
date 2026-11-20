"""Compare the old opaque region with smooth, weak synthetic output loss.

Preview only: does not change training data or claim calibrated panel luminance.
Generate with .venv/bin/python, then render with artifacts/yolo-env/bin/python
using --render-only. Replay requires the OpenCV version of the original dataset.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from generate_dataset import make_solid_panel, render_photo, write_image


def attenuation_field(size):
    y, x = np.mgrid[:size, :size].astype(float)
    x, y = x - 270, y - 270
    angle = np.deg2rad(22)
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    # Compact support with smooth boundaries, plus a gentle asymmetric shoulder.
    main = np.maximum(0, 1 - (u / 145) ** 2 - (v / 100) ** 2) ** 3
    shoulder = np.maximum(0, 1 - ((u + 60) / 105) ** 2 - ((v - 10) / 85) ** 2) ** 3
    field = .8 * main + .2 * shoulder
    return field / field.max()


def preview(dataset, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows = map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines())
    row = next(r for r in rows if r["image_id"] == "case0752_white_normal")
    camera = row["camera"]
    noise = np.random.default_rng(camera["seed"]).normal(0, camera["noise_std"], (512, 512, 3))
    matrix = np.asarray(camera["homography"])
    panel = make_solid_panel(255, 512, row["panel"])

    def capture(image):
        return render_photo(image, matrix, camera["blur_sigma"], noise)

    normal = capture(panel)
    assert np.array_equal(normal, cv2.imread(str(dataset / row["image"])))
    field = attenuation_field(512)
    assert field.min() == 0 and np.isclose(field.max(), 1)
    output.mkdir(parents=True)
    old = cv2.imread(str(dataset / "images/test/case0752_white_dark_region.png"))
    photos = [("00_normal", "정상 기준", normal), ("01_old_opaque", "기존 예시 · 검은 영역", old)]
    metadata = []
    for percent in (5, 10, 15):
        strength = percent / 100
        changed = np.rint(panel.astype(float) * (1 - strength * field[:, :, None])).astype(np.uint8)
        assert np.array_equal(changed[field == 0], panel[field == 0])
        assert changed.min() > 0 and np.all(changed <= panel)
        photo = capture(changed)
        delta = normal.astype(int) - photo.astype(int)
        photos.append((f"02_soft_{percent:02d}", f"완만한 출력 저하 · 최대 {percent}%", photo))
        metadata.append(dict(max_digital_output_loss=strength, peak_photo_delta=int(delta.max())))
    # This crop enlarges spatial detail without increasing image contrast.
    crop = photos[3][2][170:370, 170:370]
    enlarged = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_NEAREST)
    photos.append(("03_soft_10_crop", "10% 예시 부분 확대 · 밝기 보정 없음", enlarged))
    for name, _, photo in photos:
        write_image(output / f"{name}.png", photo)
    (output / "preview.json").write_text(json.dumps(dict(
        source_case_id=row["source_case_id"], is_synthetic=True, training_data_changed=False,
        opencv=cv2.__version__,
        note="Preview digital RGB attenuation, not measured OLED luminance or a validated defect standard.",
        variants=metadata,
    ), ensure_ascii=False, indent=2) + "\n")


def render_preview(output):
    from PIL import Image, ImageDraw, ImageFont

    photos = [("00_normal", "정상 기준"), ("01_old_opaque", "기존 예시 · 검은 영역")]
    photos.extend((f"02_soft_{percent:02d}", f"완만한 출력 저하 · 최대 {percent}%") for percent in (5, 10, 15))
    photos.append(("03_soft_10_crop", "10% 예시 부분 확대 · 밝기 보정 없음"))
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 21)
    sheet = Image.new("RGB", (1604, 1270), "#f3f5f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), "어두운 영역 생성 방식 비교 · 같은 구도·노출·노이즈", fill="#14243d", font=font)
    for index, (name, title) in enumerate(photos):
        x, y = 20 + index % 3 * 528, 70 + index // 3 * 590
        draw.text((x, y), title, fill="#14243d", font=small)
        with Image.open(output / f"{name}.png") as photo:
            sheet.paste(photo.convert("RGB"), (x, y + 36))
    draw.text((24, 1233), "합성 미리보기 · %는 생성 영상 RGB 값의 감소율이며 실측 패널 휘도 기준이 아닙니다.", fill="#45536a", font=small)
    sheet.save(output / "comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-white-five-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dark-region-soft-preview-v1"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render_preview(args.output)
    else:
        preview(args.dataset, args.output)
