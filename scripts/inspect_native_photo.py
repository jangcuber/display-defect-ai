"""Resident native-resolution inspector using the validation-selected model."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from display_defect_ai.native_inference import KINDS, NativeEngine
from display_defect_ai.photo_inspection import _load_photo, InvalidPhoto

COLORS = ((175, 150, 0), (155, 45, 225), (0, 145, 245), (235, 105, 30), (210, 75, 145))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(value):
    # Model diagnostics may temporarily redirect sys.stdout; protocol events must
    # still reach the original pipe, including progress callbacks inside inference.
    print(json.dumps(value, ensure_ascii=False), file=sys.__stdout__, flush=True)


def fixed_frame_aligned(image):
    """Verify the registered border at both ends of each edge without resampling pixels."""
    h, w = image.shape[:2]
    if (w, h) != (5888, 3368):
        return False
    profiles = []
    for center in (w // 4, 3 * w // 4):
        profiles.extend((image[:128, center-128:center+128:8].mean((1, 2)),
                         image[-128:, center-128:center+128:8][::-1].mean((1, 2))))
    for center in (h // 4, 3 * h // 4):
        profiles.extend((image[center-128:center+128:8, :128].mean((0, 2)),
                         image[center-128:center+128:8, -128:][:, ::-1].mean((0, 2))))
    return all(int(np.argmax(np.diff(profile[56:72]))) + 57 == 64
               and np.median(profile[:48]) < 80 and np.median(profile[80:]) > 170
               for profile in profiles)


class Inspector:
    def __init__(self, experiment, device='auto', *, source_root=ROOT):
        import torch
        self.experiment = experiment.resolve()
        self.frozen = json.loads((experiment / 'frozen_selection.json').read_text())
        self.kind = self.frozen['selected_model']
        config = self.frozen['models'][self.kind]
        self.selected = config['selected']
        weights = experiment / ('unet/best.pt' if self.kind == 'unet' else 'yolo/train/weights/best.pt')
        self.weights_sha = sha(weights)
        self.inference_sha = sha(source_root / 'src/display_defect_ai/native_inference.py')
        if self.weights_sha != config['weights_sha256']:
            raise ValueError('검증된 모델과 가중치가 다릅니다. 비교 실험 설정을 확인해 주세요.')
        if self.inference_sha != self.frozen['inference_source_sha256']:
            raise ValueError('검증 이후 추론 코드가 바뀌었습니다. 비교 실험을 다시 검증해 주세요.')
        for name, expected in self.frozen.get("runtime_source_sha256", {}).items():
            if sha(source_root / "src/display_defect_ai" / name) != expected:
                raise ValueError("검증 이후 모델 실행 코드가 바뀌었습니다: " + name)
        self.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        with redirect_stdout(sys.stderr):
            self.engine = NativeEngine(self.kind, weights, self.device)
        self.name = 'U-Net · 픽셀 분할' if self.kind == 'unet' else 'YOLO26n · 객체 검출'

    def inspect(self, photo, output, progress=None):
        started = time.perf_counter()
        if photo.resolve().parent == output.resolve():
            raise InvalidPhoto("원본 사진 폴더와 다른 결과 폴더를 지정해 주세요.")
        image = _load_photo(photo)
        h, w = image.shape[:2]
        # Fixed registered framing: no interpolation, perspective warp or color correction.
        registered = (w, h) == (5888, 3368)
        tile = (w, h) == (1024, 1024)
        roi = (67, 67, w - 67, h - 67) if registered else (0, 0, w, h)
        sample = image[roi[1]:roi[3]:8, roi[0]:roi[2]:8]
        median = float(np.median(sample))
        clipped = float(np.mean(np.max(sample, axis=2) >= 254))
        framing_ok = tile or (registered and fixed_frame_aligned(image))
        profile_ok = framing_ok and 200 <= median <= 249 and clipped < .01
        with redirect_stdout(sys.stderr):
            detections, maps = self.engine.predict_image(image, self.selected, roi, progress)
        output.mkdir(parents=True, exist_ok=True)
        preview = image.copy()
        scale = min(1., 1700 / max(w, h))
        preview = cv2.resize(preview, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output / 'preview.png'), preview)
        records = []
        for index, item in enumerate(detections):
            bx1, by1, bx2, by2 = item.box
            x1, y1, x2, y2 = map(int, item.box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if maps is not None:
                channel = KINDS.index(item.kind)
                cutoff = int(np.ceil(self.selected['pixel_thresholds'][item.kind] * 255))
                if maps[channel, cy, cx] < cutoff:
                    yy, xx = np.nonzero(maps[channel, y1:y2, x1:x2] >= cutoff)
                    if len(xx):
                        nearest = np.argmin((xx + x1 - cx) ** 2 + (yy + y1 - cy) ** 2)
                        cx, cy = x1 + int(xx[nearest]), y1 + int(yy[nearest])
            # Native raw detail: enough context to see a broken line's gaps.
            half = 64 if item.kind in KINDS[2:4] else 128 if item.kind == 'dark_region' else 32
            left, top, right, bottom = max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)
            crop = output / f'detail_{index:03d}.png'
            cv2.imwrite(str(crop), image[top:bottom, left:right])
            records.append(dict(kind=item.kind, score=item.score, box=list(item.box),
                polygon=[[bx1 / w, by1 / h], [bx2 / w, by1 / h], [bx2 / w, by2 / h], [bx1 / w, by2 / h]],
                crop=str(crop.resolve()), crop_box=[left, top, right, bottom], crop_anchor=[cx, cy],
                detail=f'검출 박스 {bx2-bx1:.1f} × {by2-by1:.1f} 이미지 px · 모델 점수 {item.score:.3f}'))
        overlay = None
        if maps is not None:
            # Save actual independent class masks, including class overlaps. Display tint only.
            bitmask = np.zeros((h, w), np.uint8)
            tint = np.zeros((h, w, 3), np.uint8)
            active = np.zeros((h, w), bool)
            for i, kind in enumerate(KINDS):
                plane = maps[i] >= int(np.ceil(self.selected['pixel_thresholds'][kind] * 255))
                bitmask[plane] |= 1 << i
                tint[plane] = COLORS[i]
                active |= plane
            cv2.imwrite(str(output / 'pixel_mask.png'), bitmask)
            thumb_tint = cv2.resize(tint, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)
            thumb_active = cv2.resize(active.astype(np.uint8), (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            colored = preview.copy()
            colored[thumb_active] = (preview[thumb_active].astype(float) * .55 + thumb_tint[thumb_active] * .45).astype(np.uint8)
            overlay = str((output / 'mask_preview.png').resolve())
            cv2.imwrite(overlay, colored)
        decision = ('defect' if records else 'normal') if profile_ok else 'review'
        message = ('합성 데이터의 촬영 조건을 기준으로 판정했습니다.' if profile_ok else
                   '학습한 촬영 크기·구도·노출 범위와 다릅니다. 표시 결과를 확인하고 촬영 조건을 검토해 주세요.')
        result = dict(schema='oled-white-3', decision=decision, message=message,
            model_name=self.name, selected_model=self.kind, device=self.device, width=w, height=h,
            weights_sha256=self.weights_sha, inference_source_sha256=self.inference_sha,
            detections=records, preview=str((output / 'preview.png').resolve()), mask_preview=overlay,
            seconds=time.perf_counter()-started, profile_ok=profile_ok,
            note='합성 데이터 기준 · 이미지 px와 실제 패널 픽셀은 별도 · 실제 촬영 성능 미검증')
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo', nargs='?', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/native-inspection'))
    parser.add_argument('--resource-root', type=Path, default=ROOT, help='Root of bundled data and verified source')
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--server', action='store_true')
    args = parser.parse_args()
    source_root = args.resource_root.resolve()
    experiment = args.experiment or source_root / 'outputs/paired-model-comparison-v1'
    inspector = None
    def run(request):
        nonlocal inspector
        identity = request.get('id', '')
        try:
            if inspector is None:
                emit(dict(event='loading', id=identity))
                inspector = Inspector(experiment, args.device, source_root=source_root)
            result = inspector.inspect(Path(request['photo']), Path(request['output']),
                lambda done, total: emit(dict(event='progress', id=identity, done=done, total=total)))
            emit(dict(event='result', id=identity, result=result))
        except Exception as error:
            emit(dict(event='error', id=identity, message=str(error)))
            if not args.server:
                raise
    if args.server:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except ValueError:
                emit(dict(event='error', id='', message='검사 요청을 읽을 수 없습니다.'))
                continue
            run(request)
    else:
        if args.photo is None:
            parser.error('사진 경로나 --server가 필요합니다.')
        run(dict(photo=str(args.photo), output=str(args.output)))


if __name__ == '__main__':
    cv2.setNumThreads(2)
    main()
