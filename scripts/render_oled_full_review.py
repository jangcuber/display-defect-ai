"""Render actual generated training tiles as paired, unmarked review examples."""

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLES = dict(normal="정상", dark_dot="어두운 점", color_dot="색 이상점",
              horizontal_line="가로선", vertical_line="세로선", dark_region="완만한 영역 저하", mixed="복합 불량")


def review(dataset):
    root = dataset / "review"
    root.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    groups = defaultdict(dict)
    for row in rows:
        groups[row["source_case_id"]][row["kind"]] = row

    def score(group):
        value = 4 * (group["dark_region"]["objects"][0]["attributes"]["tier"] == "medium")
        for kind in ("dark_dot", "color_dot"):
            attrs = group[kind]["objects"][0]["attributes"]
            value += 2 * (attrs["native_width"] == 1)
            value += attrs["tier"] == "medium"
        for kind in ("horizontal_line", "vertical_line"):
            value += 2 * (group[kind]["objects"][0]["attributes"]["thickness"] == 1)
        value += len({obj["kind"] for obj in group["mixed"]["objects"]}) == 3
        return value

    sid, chosen = max(groups.items(), key=lambda item: score(item[1]))
    selected = [chosen[kind] for kind in TITLES if kind != "normal"]
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font = ImageFont.truetype(font_path, 24)
    small = ImageFont.truetype(font_path, 18)
    cards = []
    for row in selected:
        with Image.open(dataset / row["normal_image"]) as source:
            normal = source.convert("RGB")
        with Image.open(dataset / row["image"]) as source:
            defect = source.convert("RGB")
        obj = row["objects"][0]
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        delta = np.abs(np.asarray(normal).astype(int) - np.asarray(defect).astype(int)).max(axis=2)
        yy, xx = np.nonzero(delta[y1:y2, x1:x2] >= 2)
        yy, xx = yy + y1, xx + x1
        nearest = np.argmin((xx - (x1 + x2) / 2) ** 2 + (yy - (y1 + y2) / 2) ** 2)
        cx, cy = int(xx[nearest]), int(yy[nearest])
        width = 64 if obj["kind"] != "dark_region" else min(1024, max(256, max(x2 - x1, y2 - y1) + 48))
        left, top = max(0, min(1024 - width, cx - width // 2)), max(0, min(1024 - width, cy - width // 2))
        crop = (left, top, left + width, top + width)
        name = row["image_id"]
        for suffix, photo in (("normal", normal), ("defect", defect)):
            photo.resize((512, 512), Image.Resampling.LANCZOS).save(root / f"{name}_{suffix}_view.png")
            patch = photo.crop(crop).resize((256, 256), Image.Resampling.NEAREST if width < 256 else Image.Resampling.LANCZOS)
            patch.save(root / f"{name}_{suffix}_crop.png")
        row["review_crop_xyxy"] = list(crop)
        row["review_crop_scale"] = 256 / width
        title = TITLES[row["kind"]]
        attributes = obj["attributes"]
        details = f"객체 {len(row['objects'])}개 · " + " / ".join(
            f"{channel} 최대 {loss:.1%}" for channel, loss in attributes["channel_strengths"].items())
        normal_href = html.escape(os.path.relpath(dataset / row["normal_image"], root), quote=True)
        defect_href = html.escape(os.path.relpath(dataset / row["image"], root), quote=True)
        cards.append(f'''<section><h2>{title}</h2><p>{details}</p>
<div class="pair"><figure><figcaption>정상 원본 타일</figcaption><a target="_blank" href="{normal_href}"><img src="{name}_normal_view.png" alt="정상"></a></figure>
<figure><figcaption>결함 추가 타일</figcaption><a target="_blank" href="{defect_href}"><img src="{name}_defect_view.png" alt="{title}"></a></figure></div>
<p>같은 위치 크롭 · 정상 / 결함 · 원본 대비 {256 / width:g}배 · 밝기 보정 없음</p>
<div class="crops"><img src="{name}_normal_crop.png" alt="정상 크롭"><img src="{name}_defect_crop.png" alt="결함 크롭"></div></section>''')
    for page in range(2):
        sheet = Image.new("RGB", (1200, 1340), "#f3f5f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 14), "전체 데이터 실제 생성 예시 · 정상 / 결함 / 같은 위치 비교", font=font, fill="#14243d")
        for x, label in zip((20, 330, 640), ("정상 타일 축소", "결함 타일 축소", "같은 위치 · 정상 / 결함")):
            draw.text((x, 53), label, font=font, fill="#45536a")
        for index, row in enumerate(selected[page * 3:(page + 1) * 3]):
            top = 96 + index * 394
            name = row["image_id"]
            obj = row["objects"][0]["attributes"]
            details = " / ".join(f"{channel} {loss:.0%}" for channel, loss in obj["channel_strengths"].items())
            draw.text((20, top), f"{TITLES[row['kind']]} · 객체 {len(row['objects'])}개 · 첫 객체 {details}", font=font, fill="#14243d")
            for column, suffix in enumerate(("normal", "defect")):
                with Image.open(root / f"{name}_{suffix}_view.png") as photo:
                    photo = photo.resize((288, 288), Image.Resampling.LANCZOS)
                    sheet.paste(photo, (20 + column * 310, top + 40))
                with Image.open(root / f"{name}_{suffix}_crop.png") as patch:
                    sheet.paste(patch, (640 + column * 258, top + 40))
            draw.text((20, top + 343), f"{name} · 같은 정상·카메라·좌표별 노이즈 공유", font=small, fill="#45536a")
            draw.text((640, top + 308), f"타일 원본 대비 {row['review_crop_scale']:.3g}배 · 대비 보정 없음", font=small, fill="#45536a")
        draw.text((20, 1290), "원본 해상도 1024px 타일 · 전체 장면은 5888×3368로 재현 가능 · 합성 / 실제 장비 미보정", font=small, fill="#45536a")
        sheet.save(root / f"examples_{page + 1:02d}.png")
    normal_rows = [row for row in rows if row["kind"] == "normal"]
    normal_rows.sort(key=lambda row: groups[row["source_case_id"]]["dark_region"]["image_id"])
    grid = Image.new("RGB", (972, 1090), "#f3f5f8")
    draw = ImageDraw.Draw(grid)
    draw.text((20, 14), "서로 다른 정상 원본 · 실제 생성 타일", font=font, fill="#14243d")
    draw.text((20, 42), "전체 장면 구도는 고정 · 원본 안의 타일 위치는 다름", font=small, fill="#45536a")
    for index, row in enumerate(normal_rows[:9]):
        x, y = 20 + index % 3 * 316, 62 + index // 3 * 336
        with Image.open(dataset / row["image"]) as photo:
            grid.paste(photo.convert("RGB").resize((296, 296), Image.Resampling.LANCZOS), (x, y))
        draw.text((x, y + 302), row["source_case_id"], font=small, fill="#45536a")
    grid.save(root / "normal_examples.png")
    heading = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전체 OLED 데이터 검토</title>
<style>body{font-family:-apple-system,sans-serif;background:#f3f5f8;color:#14243d;margin:0}main{max-width:1400px;margin:auto;padding:24px}section{background:white;border-radius:12px;padding:22px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{margin-bottom:10px}a img{width:100%;height:auto}.crops{display:flex;gap:4px;flex-wrap:wrap}.crops img{width:256px;height:256px}p{line-height:1.6;color:#45536a}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>
<main><h1>전체 생성 데이터의 정상·불량 짝 비교</h1><p>정상 1,000장과 불량 6,000장의 실제 학습용 PNG에서 고른 예시입니다. 전체 사진을 줄여 작은 결함이 사라지는 것을 피하도록 원본 해상도의 1024px 타일을 저장했습니다. 원본을 누르면 타일 전체 크기로 열립니다.</p>
<p>검출 표시·대비 강화 없음 · 픽셀은 합성 패널 기준 · 실제 장비 미보정</p><p><a href="normal_examples.png">서로 다른 정상 원본 보기</a></p>'''
    full_links = []
    for path in sorted((root / "full").glob("*.png")):
        kind = path.stem.split("_", 1)[1]
        full_links.append(f'<a target="_blank" href="full/{path.name}">{TITLES.get(kind, kind)} 전체 장면</a>')
    if full_links:
        heading += '<section><h2>전체 장면 원본 예시</h2><p>5888×3368의 고정 구도 장면입니다. 같은 위치를 자르면 저장 타일과 픽셀이 일치합니다. 전체 장면 PNG는 이 예시만 저장하고 나머지는 내보내기 도구로 재현합니다.</p><p>' + ' · '.join(full_links) + '</p></section>'
    (root / "index.html").write_text(heading + "".join(cards) + "</main></html>")
    (root / "selection.json").write_text(json.dumps(dict(source_case_id=sid, scene_ids=[row["scene_id"] for row in selected]), indent=2) + "\n")
    print(root.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    review(parser.parse_args().dataset)
