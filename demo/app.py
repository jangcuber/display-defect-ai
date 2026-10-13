"""Run from the repository root: python demo/app.py."""

import json
from pathlib import Path
import sys

# Keep the local demo runnable without editable installation or packaging changes.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspection import DemoSession, UNSUPPORTED


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        print("Tkinter가 필요합니다. demo/README.md의 운영체제별 설치 안내를 확인하세요.", file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError:
        print("그래픽 화면을 열 수 없습니다. 데스크톱에서 실행하고 Tkinter 설치를 확인하세요.", file=sys.stderr)
        return 1
    root.title("디스플레이 이미지 검사 · 합성 데모")
    root.geometry("880x780")
    session = DemoSession()
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="합성 테스트 / 실제 성능 미검증", font=("", 18, "bold")).pack(anchor="w")
    ttk.Label(frame, text="학습된 AI가 아닙니다. 회색 기준과의 차이를 계산하는 입출력 연습입니다.").pack(anchor="w")
    ttk.Label(frame, text="로컬 파일만 읽으며 저장·전송하지 않습니다. 실제 사진은 선택하지 마세요.").pack(anchor="w")
    mode = tk.StringVar(value="synthetic")
    status = tk.StringVar(value="합성 예제를 열거나 표시가 있는 P2 PGM 파일을 선택하세요.")
    summary = tk.StringVar(value="결과 없음")
    images = ttk.Frame(frame)
    images.pack(fill="x", pady=12)
    canvases = []
    for title in ("원본 회색조 (좌상단 원점)", "합성 히트맵 (검정 0 → 빨강 1)"):
        panel = ttk.Frame(images)
        panel.pack(side="left", padx=8)
        ttk.Label(panel, text=title).pack()
        canvas = tk.Canvas(panel, width=256, height=256, background="#eeeeee", highlightthickness=0)
        canvas.pack()
        canvases.append(canvas)
    ttk.Label(frame, textvariable=summary, wraplength=820).pack(anchor="w")
    ttk.Label(frame, text="히트맵은 원본 크기의 값이며 화면에서 정수 배율로 확대합니다. 이미지별 재정규화 없음.").pack(anchor="w")
    ttk.Label(frame, text="점수와 히트맵 밝기는 결함 확률이 아닙니다. 임계값을 임의로 정하지 않습니다.").pack(anchor="w")
    ttk.Label(frame, text="출처 및 보정 메타데이터 (training_run_id=null: 학습 없음)").pack(anchor="w", pady=(8, 0))
    details = tk.Text(frame, height=9, wrap="word", state="disabled")
    details.pack(fill="both", expand=True)
    # Keep PhotoImage objects alive while displayed.
    photos = []

    def clear():
        session.clear()
        photos.clear()
        for canvas in canvases:
            canvas.delete("all")
        summary.set("결과 없음")
        details.configure(state="normal")
        details.delete("1.0", "end")
        details.configure(state="disabled")

    def draw(canvas, grid, heatmap=False):
        height, width = len(grid), len(grid[0])
        photo = tk.PhotoImage(width=width, height=height)
        rows = []
        for row in grid:
            colors = []
            for value in row:
                level = round(value * 255)
                colors.append(f"#{level:02x}0000" if heatmap else f"#{level:02x}{level:02x}{level:02x}")
            rows.append("{" + " ".join(colors) + "}")
        photo.put(" ".join(rows))
        photo = photo.zoom(max(1, min(256 // width, 256 // height)))
        photos.append(photo)
        canvas.create_image(0, 0, image=photo, anchor="nw")

    def inspect(path):
        clear()
        try:
            result = session.inspect(path, synthetic=mode.get() == "synthetic")
            draw(canvases[0], session.image.pixels)
            if result.heatmap is None:
                canvases[1].create_text(128, 128, text="위치 정보 없음")
            else:
                draw(canvases[1], result.heatmap, heatmap=True)
            decision = {"uncalibrated": "미보정 — 정상/불량 판정 없음",
                        "normal": "보정 기준 미만 (normal)", "anomaly": "보정 기준 이상 (anomaly)"}[result.decision]
            threshold = "없음" if result.calibration is None else str(result.calibration.threshold)
            summary.set(f"합성 결과 · {result.width}×{result.height} · 점수 {result.score:.6g}\n{decision} · 임계값: {threshold}")
            payload = result.to_dict()
            # Pixels stay on the canvases; show compact provenance/calibration metadata.
            payload.pop("heatmap")
            details.configure(state="normal")
            details.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            details.configure(state="disabled")
            status.set("합성 표시 완료. 실제 결함 탐지 결과가 아닙니다.")
        except (ValueError, OSError):
            clear()
            status.set("파일을 읽거나 검사할 수 없습니다. 합성 P2 PGM, 1 MiB 이하, 각 변 256 이하인지 확인하세요.")
        except Exception:
            clear()
            status.set("데모 실행 오류입니다. 결과를 지웠습니다. 다시 선택하거나 터미널 테스트를 실행하세요.")

    def choose():
        clear()
        path = filedialog.askopenfilename(title="합성 PGM 선택 (1 MiB, 256×256 이하)",
                                          filetypes=[("합성 ASCII PGM", "*.pgm")])
        if path:
            inspect(path)
        else:
            status.set("파일 선택 취소 · 결과 없음")

    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=8)
    buttons = []
    for label, command in (("합성 밝은 패치 예제", lambda: inspect(ROOT / "tests/fixtures/synthetic/bright_patch.pgm")),
                           ("합성 평면 예제", lambda: inspect(ROOT / "tests/fixtures/synthetic/flat.pgm")),
                           ("합성 파일 선택", choose)):
        button = ttk.Button(controls, text=label, command=command)
        button.pack(side="left", padx=3)
        buttons.append(button)

    def change_mode():
        clear()
        synthetic = mode.get() == "synthetic"
        for button in buttons:
            button.configure(state="normal" if synthetic else "disabled")
        status.set("합성 파일을 선택하세요." if synthetic else UNSUPPORTED)

    modes = ttk.Frame(frame)
    modes.pack(fill="x")
    ttk.Radiobutton(modes, text="합성 시연", variable=mode, value="synthetic", command=change_mode).pack(side="left")
    ttk.Radiobutton(modes, text="실제 추론 (미지원)", variable=mode, value="real", command=change_mode).pack(side="left")
    ttk.Label(frame, textvariable=status, wraplength=820).pack(anchor="w", pady=8)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
