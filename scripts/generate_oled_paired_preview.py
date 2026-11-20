"""Generate review-only white OLED normal/defect pairs before a full v2 dataset.

Geometry is a declared synthetic assumption, not a calibrated physical OLED model.
Generate with .venv/bin/python; use artifacts/yolo-env/bin/python --render-only
for the contact sheets (Pillow is only required for that separate step).
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from generate_dataset import write_image

PANEL_W, PANEL_H, SCALE, BORDER = 1920, 1080, 3, 64
PHOTO_W, PHOTO_H = PANEL_W * SCALE + 2 * BORDER, PANEL_H * SCALE + 2 * BORDER
KINDS = ("dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region")

# The preview covers the five classes and two weak region strengths.
CASES = (
    dict(index=0, level=232, variants=(
        dict(name="01_dark_dot", title="어두운 점", kind="dark_dot", x=820, y=420,
             native_width=1, native_height=1, strength=.65, channels="RGB"),
        dict(name="02_color_dot", title="색 이상점", kind="color_dot", x=1120, y=610,
             native_width=1, native_height=1, strength=.50, channels="G"),
    )),
    dict(index=1, level=237, variants=(
        dict(name="03_horizontal_line", title="가로선", kind="horizontal_line", x=720, y=450,
             native_width=240, native_height=1, strength=.15, channels="RGB"),
        dict(name="04_vertical_line", title="끊긴 세로선", kind="vertical_line", x=1080, y=300,
             native_width=1, native_height=360, strength=.20, channels="R", segment=48, gap=12),
    )),
    dict(index=2, level=228, variants=(
        dict(name="05_region_03", title="완만한 영역 저하 3%", kind="dark_region", x=1120, y=620,
             radius_x=300, radius_y=185, angle=24, strength=.03, channels="RGB"),
        dict(name="06_region_06", title="완만한 영역 저하 6%", kind="dark_region", x=1120, y=620,
             radius_x=300, radius_y=185, angle=24, strength=.06, channels="RGB"),
    )),
)


def channel_cases():
    """Single/pair/triple channel losses with independently varying spatial profiles."""
    combinations = (
        ("R", {"R": .12}), ("G", {"G": .25}), ("B", {"B": .40}),
        ("RG", {"R": .08, "G": .20}), ("RB", {"R": .18, "B": .07}),
        ("GB", {"G": .12, "B": .30}), ("RGB", {"R": .04, "G": .18, "B": .10}),
    )
    cases = [dict(index=i, level=level, variants=[]) for i, level in enumerate((232, 237, 228))]
    for index, (channels, strengths) in enumerate(combinations):
        common = dict(channels=channels, channel_strengths=strengths, strength=max(strengths.values()))
        cases[0]["variants"].append(dict(
            common, name=f"{index + 1:02d}_dot_{channels}", title=f"작은 점 · {channels} 출력 저하",
            kind="color_dot", x=820, y=420, native_width=1, native_height=1,
        ))
        horizontal = index % 2 == 0
        profiles = {channel: dict(profile=("rising", "falling", "center")[(index + j) % 3])
                    for j, channel in enumerate(channels)}
        line = dict(common, name=f"{index + 8:02d}_line_{channels}", title=f"얇은 선 · {channels} 점진 저하",
                    kind="horizontal_line" if horizontal else "vertical_line", x=720, y=350,
                    native_width=420 if horizontal else 1, native_height=1 if horizontal else 420,
                    channel_profiles=profiles, focus_fraction=.5)
        if index in (3, 5):
            line.update(segment=48, gap=12)
        cases[1]["variants"].append(line)
        region_strengths = {channel: strength * .25 for channel, strength in strengths.items()}
        cases[2]["variants"].append(dict(
            common, name=f"{index + 15:02d}_region_{channels}", title=f"완만한 영역 · {channels} 점진 저하",
            kind="dark_region", x=1120, y=620, radius_x=300, radius_y=185, angle=24,
            channel_strengths=region_strengths, strength=max(region_strengths.values()),
            channel_profiles={channel: dict(profile="power", exponent=(.7, 1.6, 2.2)[j % 3])
                              for j, channel in enumerate(channels)},
        ))
    return cases


def channel_field(field, spec, settings):
    profile = settings.get("profile", "constant")
    if profile == "constant":
        return field
    if profile == "power":
        exponent = settings["exponent"]
        if not np.isfinite(exponent) or exponent <= 0:
            raise ValueError("Channel field exponent must be positive")
        return field ** exponent
    if spec["kind"] not in ("horizontal_line", "vertical_line"):
        raise ValueError("Longitudinal gradients require a line")
    horizontal = spec["kind"] == "horizontal_line"
    length = spec["native_width"] if horizontal else spec["native_height"]
    coordinates = np.arange(PANEL_W if horizontal else PANEL_H, dtype=np.float32)
    t = np.clip((coordinates - spec["x" if horizontal else "y"]) / max(1, length - 1), 0, 1)
    smooth = t * t * (3 - 2 * t)
    if profile == "rising":
        response = smooth
    elif profile == "falling":
        response = 1 - smooth
    elif profile == "center":
        response = 16 * t * t * (1 - t) * (1 - t)
    else:
        raise ValueError(f"Unknown channel profile: {profile}")
    response = .05 + .95 * response
    return field * (response[None, :] if horizontal else response[:, None])


def normal_panel(case, rng):
    y, x = np.mgrid[:PANEL_H, :PANEL_W].astype(np.float32)
    x, y = x / PANEL_W - .5, y / PANEL_H - .5
    slopes = rng.uniform(-.004, .004, 2)
    shading = .0015 * np.exp(-((x - .12) / .45) ** 2 - ((y + .1) / .4) ** 2)
    field = 1 + slopes[0] * x + slopes[1] * y - shading
    tint = rng.uniform(-.0015, .0015, 3)
    tint -= tint.mean()
    panel = case["level"] * field[:, :, None] * (1 + tint)
    return panel.astype(np.float32), dict(
        nominal_digital_level=case["level"], gradient_xy=slopes.tolist(),
        tint_bgr=(1 + tint).tolist(), soft_shading_amplitude=.0015,
    )


def camera_signal(panel):
    signal = np.full((PHOTO_H, PHOTO_W, 3), 24, np.float32)
    signal[BORDER:-BORDER, BORDER:-BORDER] = cv2.resize(
        panel, (PANEL_W * SCALE, PANEL_H * SCALE), interpolation=cv2.INTER_NEAREST,
    )
    return cv2.GaussianBlur(signal, (7, 7), .65)


def capture(signal, noise):
    return np.clip(np.rint(signal + noise), 0, 255).astype(np.uint8)


def apply_defect(panel, spec):
    field = np.zeros((PANEL_H, PANEL_W), np.float32)
    if spec["kind"] == "dark_region":
        y, x = np.mgrid[:PANEL_H, :PANEL_W].astype(np.float32)
        x, y = x - spec["x"], y - spec["y"]
        angle = np.deg2rad(spec["angle"])
        u, v = x * np.cos(angle) + y * np.sin(angle), -x * np.sin(angle) + y * np.cos(angle)
        rx, ry = spec["radius_x"], spec["radius_y"]
        main = np.maximum(0, 1 - (u / rx) ** 2 - (v / ry) ** 2) ** 3
        shoulder = np.maximum(0, 1 - ((u + rx * .35) / (rx * .8)) ** 2
                              - ((v - ry * .1) / (ry * .9)) ** 2) ** 3
        field = .8 * main + .2 * shoulder
        field /= field.max()
    else:
        x, y = spec["x"], spec["y"]
        field[y:y + spec["native_height"], x:x + spec["native_width"]] = 1
        if "segment" in spec:
            for offset in range(spec["segment"], spec["native_height"], spec["segment"] + spec["gap"]):
                field[y + offset:y + min(offset + spec["gap"], spec["native_height"]), x:x + 1] = 0
    altered = panel.copy()
    strengths = spec.get("channel_strengths", {channel: spec["strength"] for channel in spec["channels"]})
    if set(strengths) != set(spec["channels"]) or any(
            not np.isfinite(value) or not 0 < value <= 1 for value in strengths.values()):
        raise ValueError("Expected a valid loss for every affected channel")
    for channel in spec["channels"]:
        index = {"B": 0, "G": 1, "R": 2}[channel]
        spatial = channel_field(field, spec, spec.get("channel_profiles", {}).get(channel, {}))
        altered[:, :, index] *= 1 - spatial * strengths[channel]
    assert np.array_equal(altered[field == 0], panel[field == 0])
    assert np.all(altered <= panel) and altered.min() > 0
    return altered, field


def box_for(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Preview defect is not observable in the rendered image")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_image(root, relative, image):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, image)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_box(spec):
    if spec["kind"] == "dark_region":
        cx, cy, width = spec["x"], spec["y"], 2048
    else:
        cx = spec["x"] + min(spec["native_width"], 24) / 2
        cy = spec["y"] + min(spec["native_height"], 24) / 2
        if "focus_fraction" in spec:
            if spec["kind"] == "horizontal_line":
                cx = spec["x"] + spec["native_width"] * spec["focus_fraction"]
            else:
                cy = spec["y"] + spec["native_height"] * spec["focus_fraction"]
        if "segment" in spec:
            cy = spec["y"] + spec["segment"] + spec["gap"] / 2
        width = 64
    x = max(0, min(PHOTO_W - width, round(BORDER + cx * SCALE - width / 2)))
    y = max(0, min(PHOTO_H - width, round(BORDER + cy * SCALE - width / 2)))
    return [x, y, x + width, y + width]


def generate(output, seed, profile="basic"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs, records, hashes = [], [], set()
    cases = CASES if profile == "basic" else channel_cases()
    with tempfile.TemporaryDirectory(prefix=".oled-paired-preview-", dir=output.parent) as temporary:
        root = Path(temporary)
        for case in cases:
            source = f"paired_preview_{case['index']:04d}"
            rng = np.random.default_rng(np.random.SeedSequence([seed, case["index"]]))
            panel, normal_settings = normal_panel(case, rng)
            # Independent realizations between sources; identical noise within each paired family.
            noise = rng.standard_normal((PHOTO_H, PHOTO_W, 3), dtype=np.float32) * .6
            signal = camera_signal(panel)
            normal = capture(signal, noise)
            normal_path = f"normals/{source}.png"
            normal_hash = save_image(root, normal_path, normal)
            assert normal_hash not in hashes
            hashes.add(normal_hash)
            label = f"labels/{source}_normal.txt"
            (root / label).parent.mkdir(exist_ok=True)
            (root / label).write_text("")
            records.append(dict(image_id=source + "_normal", source_case_id=source, kind="normal",
                                image=normal_path, label=label, bbox_xyxy=[], objects=[],
                                split="preview", image_sha256=normal_hash, normal=normal_settings))
            panel_roi = normal[BORDER:-BORDER, BORDER:-BORDER]
            for spec in case["variants"]:
                altered, field = apply_defect(panel, spec)
                defect = capture(camera_signal(altered), noise)
                delta = cv2.absdiff(normal, defect).max(axis=2)
                visible = delta >= 2
                box, support_box = box_for(visible), box_for(delta > 0)
                name, kind = spec["name"], spec["kind"]
                photo_path = f"defects/{name}.png"
                digest = save_image(root, photo_path, defect)
                assert digest not in hashes
                hashes.add(digest)
                save_image(root, f"masks/{name}.png", visible.astype(np.uint8) * 255)
                x1, y1, x2, y2 = box
                label = f"labels/{name}.txt"
                coordinates = ((x1 + x2) / (2 * PHOTO_W), (y1 + y2) / (2 * PHOTO_H),
                               (x2 - x1) / PHOTO_W, (y2 - y1) / PHOTO_H)
                (root / label).write_text(str(KINDS.index(kind)) + " " + " ".join(f"{v:.10f}" for v in coordinates) + "\n")
                attributes = dict(spec, coordinate_units="synthetic_panel_pixels",
                                  peak_photo_delta=int(delta.max()), visible_pixels=int(visible.sum()),
                                  native_support_pixels=int(np.count_nonzero(field)),
                                  observed_support_bbox=support_box, visibility_threshold_dn=2)
                obj = dict(kind=kind, bbox_xyxy=box, attributes=attributes)
                records.append(dict(image_id=name, source_case_id=source, normal_image_id=source + "_normal",
                                    kind=kind, image=photo_path, label=label, bbox_xyxy=[box], objects=[obj],
                                    split="preview", image_sha256=digest, normal=normal_settings))
                crop = crop_box(spec)
                cx1, cy1, cx2, cy2 = crop
                for suffix, image in (("normal", normal), ("defect", defect)):
                    patch = image[cy1:cy2, cx1:cx2]
                    patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_NEAREST
                                       if cx2 - cx1 < 256 else cv2.INTER_AREA)
                    save_image(root, f"crops/{name}_{suffix}.png", patch)
                pairs.append(dict(name=name, title=spec["title"], source_case_id=source,
                                  normal=normal_path, defect=photo_path, crop_box=crop,
                                  crop_display_scale=256 / (cx2 - cx1), attributes=attributes,
                                  normal_roi_mean_bgr=panel_roi.mean(axis=(0, 1)).tolist()))
                print(f"{name}: source {source}, peak delta {delta.max()}, visible box {box}", flush=True)
            del noise, signal, normal, panel_roi, panel, altered, defect, delta, field
        for record in records:
            record.update(schema_version="oled-white-2", is_synthetic=True,
                          image_width=PHOTO_W, image_height=PHOTO_H, review_only=True)
        (root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        (root / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n")
        config = dict(generator="oled-paired-v2-preview" if profile == "basic" else "oled-channel-gradients-preview",
                      profile=profile, seed=seed, review_only=True,
                      source_cases=len(cases), normal_images=len(cases), defect_images=len(pairs), classes=list(KINDS),
                      geometry=dict(panel_width=PANEL_W, panel_height=PANEL_H, camera_scale=SCALE,
                                    border=BORDER, photo_width=PHOTO_W, photo_height=PHOTO_H),
                      camera=dict(blur_sigma=.65, noise_std=.6, response="assumed linear digital RGB",
                                  fixed_framing=True, paired_noise=True),
                      calibrated_to_real_hardware=False, full_dataset_generated=False,
                      bbox_policy="Observed paired RGB difference >= 2 DN; full support stored separately",
                      numpy=np.__version__, opencv=cv2.__version__)
        (root / "generation.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        root.rename(output)


def render_channel_examples(output, pairs):
    from PIL import Image, ImageDraw, ImageFont

    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    dots = [pair for pair in pairs if pair["attributes"]["kind"] == "color_dot"]
    sheet = Image.new("RGB", (824, 540), "#f3f5f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), "작은 점의 채널 조합 · 촬영 원본 4배 확대", font=font, fill="#14243d")
    tiles = [("정상", dots[0], "normal"), *[(pair["attributes"]["channels"], pair, "defect") for pair in dots]]
    for index, (title, pair, suffix) in enumerate(tiles):
        x, y = 20 + index % 4 * 200, 54 + index // 4 * 224
        draw.text((x, y), title, font=font, fill="#14243d")
        with Image.open(output / f"crops/{pair['name']}_{suffix}.png") as image:
            sheet.paste(image.convert("RGB").crop((48, 48, 208, 208)), (x, y + 30))
        values = pair["attributes"].get("channel_strengths", {}) if suffix == "defect" else {}
        label = " / ".join(f"{channel} {loss:.0%}" for channel, loss in values.items())
        draw.text((x, y + 194), label, font=small, fill="#45536a")
    draw.text((20, 510), "1×1 합성 패널 픽셀 · 결함 크기·밝기·대비 보정 없음", font=small, fill="#45536a")
    sheet.save(output / "channel_dots.png")

    by_name = {pair["name"]: pair for pair in pairs}
    sheet = Image.new("RGB", (1120, 824), "#f3f5f8")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 14), "영역: 채널별 저하 범위와 강도가 다름", font=font, fill="#14243d")
    regions = [("정상", by_name["15_region_R"], "normal"),
               ("R만 넓게 저하", by_name["15_region_R"], "defect"),
               ("R 넓게 · G 좁게", by_name["18_region_RG"], "defect"),
               ("R·G·B 서로 다르게", by_name["21_region_RGB"], "defect")]
    for index, (title, pair, suffix) in enumerate(regions):
        x = 20 + index * 276
        draw.text((x, 55), title, font=font, fill="#14243d")
        with Image.open(output / f"crops/{pair['name']}_{suffix}.png") as image:
            sheet.paste(image.convert("RGB"), (x, 90))
    draw.text((20, 360), "같은 정상 원본의 같은 위치 · 촬영 원본 대비 0.125배 · 밝기 보정 없음", font=small, fill="#45536a")
    draw.text((20, 410), "같은 얇은 선: 길이 방향으로 R↑ · G↓ · B는 중앙 최대", font=font, fill="#14243d")
    pair = by_name["14_line_RGB"]
    spec = pair["attributes"]
    with Image.open(output / pair["normal"]) as source:
        normal = source.convert("RGB")
    with Image.open(output / pair["defect"]) as source:
        defect = source.convert("RGB")
    for index, (title, fraction, source) in enumerate((
            ("정상", .5, normal), ("선의 앞부분", .1, defect),
            ("선의 중간", .5, defect), ("선의 뒷부분", .9, defect))):
        cx = BORDER + (spec["x"] + spec["native_width"] * fraction) * SCALE
        cy = BORDER + (spec["y"] + .5) * SCALE
        x1, y1 = round(cx - 32), round(cy - 32)
        patch = source.crop((x1, y1, x1 + 64, y1 + 64)).resize((256, 256), Image.Resampling.NEAREST)
        x = 20 + index * 276
        draw.text((x, 454), title, font=font, fill="#14243d")
        sheet.paste(patch, (x, 490))
    draw.text((20, 768), "합성 패널 폭 1픽셀 · 촬영 원본 4배 확대 · 생성 사진의 색/대비를 그대로 표시", font=small, fill="#45536a")
    sheet.save(output / "channel_gradient_examples.png")


def render(output):
    from PIL import Image, ImageDraw, ImageFont

    pairs = json.loads((output / "pairs.json").read_text())
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    title_font = ImageFont.truetype(font_path, 26)
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 18)
    if json.loads((output / "generation.json").read_text()).get("profile") == "channel-gradients":
        render_channel_examples(output, pairs)
    for page in range((len(pairs) + 2) // 3):
        sheet = Image.new("RGB", (1616, 1340), "#f3f5f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 14), "정상 원본 → 결함 추가 · 검출 박스·대비 보정 없음", font=title_font, fill="#14243d")
        for column, label in enumerate(("정상 원본 · 전체 사진 축소", "결함 추가 · 전체 사진 축소", "같은 위치 비교 · 정상 / 결함")):
            draw.text((20 + column * 536, 54), label, font=font, fill="#45536a")
        for row, pair in enumerate(pairs[page * 3:(page + 1) * 3]):
            y = 98 + row * 394
            a = pair["attributes"]
            geometry = ("생성 폭 1픽셀" if a["kind"] in ("horizontal_line", "vertical_line") else
                        "생성 크기 1×1픽셀" if a["kind"].endswith("dot") else "경계·내부 강도가 완만하게 변함")
            draw.text((20, y), f"{pair['title']} · {geometry} · 최대 출력 저하 {a['strength']:.0%}", font=font, fill="#14243d")
            for column, key in enumerate(("normal", "defect")):
                with Image.open(output / pair[key]) as original:
                    original = original.convert("RGB")
                    original.thumbnail((512, 300), Image.Resampling.LANCZOS)
                    sheet.paste(original, (20 + column * 536, y + 38))
            for column, suffix in enumerate(("normal", "defect")):
                with Image.open(output / f"crops/{pair['name']}_{suffix}.png") as patch:
                    sheet.paste(patch.convert("RGB"), (1092 + column * 258, y + 38))
            zoom = pair["crop_display_scale"]
            draw.text((1092, y + 304), f"원본 대비 {zoom:g}배 · 밝기 보정 없음", font=small, fill="#45536a")
            draw.text((20, y + 348), f"정상 원본 {pair['source_case_id']} 공유 · 촬영/노이즈 조건 동일", font=small, fill="#45536a")
        draw.text((20, 1292), "합성 가정: 패널 1920×1080, 촬영 대응 3배 · 실제 장비 미보정 · %는 디지털 RGB 감소율", font=small, fill="#45536a")
        sheet.save(output / f"pairs_{page + 1:02d}.png")
    cards = []
    for pair in pairs:
        title = html.escape(pair["title"])
        name = pair["name"]
        normal, defect = pair["normal"], pair["defect"]
        losses = pair["attributes"].get("channel_strengths", {})
        description = " · ".join(f"{channel} 최대 {strength:.1%}" for channel, strength in losses.items())
        cards.append(f'''<section><h2>{title}</h2><p>같은 정상 원본 {pair['source_case_id']} · 구도와 노이즈 공유</p>
<p>{description}</p>
<div class="pair"><figure><figcaption>정상 원본</figcaption><a href="{normal}" target="_blank"><img loading="lazy" src="{normal}" alt="정상 원본"></a></figure>
<figure><figcaption>결함 추가</figcaption><a href="{defect}" target="_blank"><img loading="lazy" src="{defect}" alt="{title}"></a></figure></div>
<p>같은 위치 크롭 · 정상 / 결함 · 촬영 원본 대비 {pair['crop_display_scale']:g}배 · 밝기 보정 없음</p>
<div class="crops"><img src="crops/{name}_normal.png" alt="정상 크롭"><img src="crops/{name}_defect.png" alt="결함 크롭"></div></section>''')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>정상·불량 짝 비교</title>
<style>body{font-family:-apple-system,sans-serif;background:#f3f5f8;color:#14243d;margin:0}main{max-width:1400px;padding:24px;margin:auto}h1{font-size:28px}p{line-height:1.6;color:#45536a}section{background:white;border-radius:12px;padding:22px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{margin-bottom:10px}a img{width:100%;height:auto}.crops{display:flex;gap:4px;flex-wrap:wrap}.crops img{width:256px;height:256px}@media(max-width:700px){.pair{grid-template-columns:1fr}main{padding:12px}}</style>
<main><h1>정상 원본과 결함 추가 사진 · PAIRCOUNT쌍</h1><p>전체 사진을 누르면 촬영 원본 크기로 열립니다. 검출 박스와 대비 강화는 적용하지 않았습니다.</p>
<p>검토용 합성 가정: 패널 1920×1080, 촬영 대응 3배. 실제 장비 미보정이며 전체 데이터 생성·재학습 전 샘플입니다.</p>'''
    if (output / "channel_gradient_examples.png").exists():
        page += '<p><a href="channel_dots.png">작은 점의 채널 조합 확대</a> · <a href="channel_gradient_examples.png">선·영역의 채널별 점진 저하 비교</a></p>'
    (output / "index.html").write_text(page.replace("PAIRCOUNT", str(len(pairs))) + "".join(cards) + "</main></html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=("basic", "channel-gradients"), default="basic")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/oled-white-paired-v2-preview" if args.profile == "basic"
                           else "outputs/oled-white-channel-gradients-preview")
    cv2.setNumThreads(2)
    if args.render_only:
        render(args.output)
    else:
        generate(args.output, args.seed, args.profile)
