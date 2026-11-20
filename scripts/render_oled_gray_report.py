"""Render comparison figures and the grayscale experiment report."""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from generate_oled_gray_dataset import KINDS, LEVELS, render

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ("어두운 점", "색 이상점", "가로선", "세로선", "밝기·색 영역 저하")
GROUPS = (KINDS[:2], KINDS[2:4], KINDS[4:])


def read(path):
    return json.loads(path.read_text())


def examples(dataset, output):
    train = [r for r in map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines()) if r["split"] == "train"]
    rows = [r for r in train if r["panel_id"] == train[0]["panel_id"]]
    fig, axes = plt.subplots(3, 4, figsize=(10, 6.5), dpi=150)
    for col, level in enumerate(LEVELS):
        by_kind = {r["kind"]: r for r in rows if r["gray_level"] == level}
        defect = by_kind["dark_region"]
        spec = defect["scene_objects"][0]
        left, top, _, _ = defect["tile_xyxy"]
        cx = spec["x"] * render.SCALE + render.BORDER - left
        cy = spec["y"] * render.SCALE + render.BORDER - top
        area = np.s_[max(0, cy-300):cy+300, max(0, cx-300):cx+300]
        for i, kind in enumerate(("normal", "dark_region")):
            photo = cv2.imread(str(dataset / by_kind[kind]["image"]))[:, :, ::-1]
            axes[i, col].imshow(photo[area])
        mask = cv2.imread(str(dataset / "masks/train" / f"{defect['image_id']}.png"), 0)
        axes[2, col].imshow((mask[area] & 16) != 0, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{level} / 255")
    for row, label in zip(axes, ("Normal", "Gradual defect", "2-DN mask"), strict=True):
        row[0].set_ylabel(label)
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Same synthetic panel and defect across four gray levels")
    fig.tight_layout()
    fig.savefig(output / "gray_pairs.png")
    plt.close(fig)


def grouped(metric, kinds):
    return tuple(sum(metric["classes"][kind][key] for kind in kinds) for key in ("tp", "total", "fp"))


def chart(by_level, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=165, sharey=True)
    for ax, title in zip(axes, ("Recall", "Precision"), strict=True):
        x = np.arange(len(LEVELS))
        series = zip(GROUPS, ("Points", "Lines", "Regions"), ("#2566af", "#249570", "#c87829"), strict=True)
        for i, (kinds, label, color) in enumerate(series):
            counts = [grouped(by_level[str(level)], kinds) for level in LEVELS]
            values = [100 * hit / max(1, total if title == "Recall" else hit + fp) for hit, total, fp in counts]
            bars = ax.bar(x + (i-1)*.25, values, width=.23, label=label, color=color)
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
        ax.set_xticks(x, LEVELS)
        ax.set_xlabel("Nominal digital gray level / 255")
        ax.set_ylim(0, 111)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("%")
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=3, loc="upper center", bbox_to_anchor=(.5, .91))
    fig.suptitle("U-Net on held-out grayscale OLED scenes")
    fig.tight_layout(rect=(0, 0, 1, .84))
    fig.savefig(output / "gray_test_chart.png")
    plt.close(fig)


def percent(numerator, denominator):
    return f"{100 * numerator / denominator:.1f}%" if denominator else "—"


def report(dataset, output):
    result = read(output / "test_results.json")
    config = read(output / "unet/run.json")
    checked = read(dataset / "verification.json")
    generated = read(dataset / "segmentation.json")
    initial = read(output / "unet/initial_validation.json")
    epochs = list(csv.DictReader((output / "unet/results.csv").read_text().splitlines()))
    best = max(epochs, key=lambda row: float(row["validation_macro_dice"]))
    test = result["full_scene_test"]
    chart(test["by_level"], output)
    lines = [
        "# OLED 회색 화면 U-Net 시험", "",
        f"회색 48·96·144·192/255에서 정상과 불량을 짝으로 만든 {checked['images']:,}장과 픽셀 마스크를 사용했습니다. "
        f"같은 가상 패널에서 나온 네 밝기와 일곱 유형을 한 분할에 유지했습니다. "
        f"학습 {config['training_images']:,}장, 검증 {config['validation_images']:,}장, 시험 {test['overall']['normal_images'] * 7:,}장입니다.", "",
        "같은 패널의 네 밝기는 결함 위치·크기·채널별 출력 저하 비율을 공유합니다. 사진은 1024×1024 원본 타일을 저장합니다. "
        f"가장 어두운 조건에서도 최대 채널 차이 2 DN 이상인 픽셀이 생기도록 레시피 {generated['rejected_below_2dn_recipes']}건을 다시 뽑았습니다. "
        "이 기준은 완전히 보이지 않는 결함을 시험에서 제외합니다. 픽셀 중복과 패널 누출은 0건입니다.", "",
        "![같은 패널의 정상·불량·마스크](gray_pairs.png)", "",
        f"흰 화면 가중치에서 {len(epochs)} epoch 미세 조정했습니다. RGB 공통 이득으로 각 타일의 중앙 밝기를 맞췄으며 "
        "채널별 차이는 유지했습니다. 입력은 사진 한 장입니다. "
        f"동일 전처리의 초기 검증 Dice는 {float(initial['macro_dice']):.1%}, 최고 체크포인트인 epoch {best['epoch']}는 "
        f"{float(best['validation_macro_dice']):.1%}입니다. 체크포인트와 후처리 임계값은 검증 자료로 결정했습니다.", "",
        "시험은 5888×3368 전체 장면을 1024 타일로 나누어 처리했습니다. 같은 유형의 박스를 IoU ≥ 0.5로 일대일 대응했습니다. "
        "재현율은 실제 불량 중 찾은 비율, 정밀도는 검출 결과 중 정답인 비율입니다.", "",
        "| 회색 단계 | 점 재현율 / 정밀도 | 선 재현율 / 정밀도 | 영역 재현율 / 정밀도 | 1패널픽셀 선 | 정상 오탐 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level, data in [*test["by_level"].items(), ("전체", test["overall"])]:
        cells = []
        for kinds in GROUPS:
            hit, total, fp = grouped(data, kinds)
            cells.append(f"{percent(hit, total)} ({hit}/{total}) / {percent(hit, hit+fp)}")
        thin = data["one_pixel_lines"]
        lines.append("| " + " | ".join([level, *cells, f"{thin['hit']}/{thin['total']}",
                                              f"{data['normal_false_alarms']}/{data['normal_images']}"]) + " |")
    lines += ["", "![회색 밝기별 재현율 및 정밀도](gray_test_chart.png)", "",
              "| 회색 단계 | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|---:|---:|"]
    for level, data in test["by_level"].items():
        cells = [f"{data['classes'][kind]['tp']}/{data['classes'][kind]['total']}" for kind in KINDS]
        lines.append("| " + " | ".join([level, *cells]) + " |")
    lines += ["", "영역의 생성 대비 구간별 검출 수입니다.", "",
              "| 회색 단계 | 약 | 중 | 강 |", "|---|---:|---:|---:|"]
    for level, data in test["by_level"].items():
        detail = data["regions_by_strength"]
        cells = [f"{detail[tier]['found']}/{detail[tier]['total']}" for tier in ("weak", "medium", "strong")]
        lines.append("| " + " | ".join([level, *cells]) + " |")
    lines += ["", "아래 표는 시험 타일에서 집계한 픽셀 Dice입니다.", "",
              "| 회색 단계 | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|---:|---:|"]
    for level, data in result["tile_pixel_metrics_by_level"].items():
        lines.append("| " + " | ".join([level, *(f"{data[kind]['dice']:.1%}" for kind in KINDS)]) + " |")
    lines += ["", "회색 값은 합성 촬영 영상의 명목 디지털 값입니다. 패널 픽셀 1개는 합성 촬영 이미지 3픽셀에 해당합니다. "
              "완전 검정, 네 단계 사이의 밝기 및 실제 카메라 촬영은 이번 시험 범위 밖입니다.", "",
              "실행 순서는 데이터 생성, 데이터 검증, U-Net 학습, U-Net 평가, 보고서 생성입니다. 저장소 루트에서 실행합니다.", "",
              "~~~bash", ".venv/bin/python scripts/generate_oled_gray_dataset.py",
              ".venv/bin/python scripts/check_oled_gray_dataset.py data/oled-gray-paired-v1",
              "artifacts/yolo-env/bin/python scripts/train_oled_gray_unet.py",
              "artifacts/yolo-env/bin/python scripts/evaluate_oled_gray_unet.py",
              "artifacts/yolo-env/bin/python scripts/render_oled_gray_report.py", "~~~", ""]
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-gray-paired-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/oled-gray-unet-v1"))
    parser.add_argument("--examples-only", action="store_true")
    args = parser.parse_args()
    examples(args.dataset, args.output)
    if not args.examples_only:
        report(args.dataset, args.output)
