"""Measure loaded-model, uncached end-to-end native photo inspection on this Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import torch
from display_defect_ai.native_inference import NativeEngine


def main(root, dataset, device, repeats):
    frozen = json.loads((root / 'frozen_selection.json').read_text())
    images = [dataset / 'review/full' / f'panel0022_{kind}.png' for kind in ('normal', 'mixed', 'dark_region')]
    result = dict(device=device, platform=platform.platform(), torch=torch.__version__,
                  repeats=repeats, input_size=[5888, 3368], overlap=128, tile_size=1024,
                  cache=False, scope='file decode + native inference + merging; loaded model; excludes synthetic rendering', models={})
    for kind in ('unet', 'yolo'):
        weights = root / ('unet/best.pt' if kind == 'unet' else 'yolo/train/weights/best.pt')
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == frozen['models'][kind]['weights_sha256']
        begin = time.perf_counter()
        engine = NativeEngine(kind, weights, device)
        selected = frozen['models'][kind]['selected']
        probe = cv2.imread(str(images[0]))[64:1088, 64:1088]
        engine.predict_tiles([probe])
        if device == 'mps': torch.mps.synchronize()
        loading = time.perf_counter() - begin
        samples = []
        for path in images:
            seconds, counts = [], []
            for _ in range(repeats):
                start = time.perf_counter()
                image = cv2.imread(str(path))
                detections, maps = engine.predict_image(image, selected, roi=(67, 67, 5821, 3301))
                if device == 'mps': torch.mps.synchronize()
                seconds.append(time.perf_counter()-start); counts.append(len(detections))
                del image, maps
            assert len(set(counts)) == 1
            samples.append(dict(image=path.name, seconds=seconds, median_seconds=statistics.median(seconds), detections=counts[0]))
        result['models'][kind] = dict(load_and_warmup_seconds=loading, samples=samples,
            median_seconds=statistics.median([v for sample in samples for v in sample['seconds']]))
        del engine
        if device == 'mps': torch.mps.empty_cache()
    (root / 'inference_timing.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/paired-model-comparison-v1'))
    p.add_argument('--dataset', type=Path, default=Path('data/oled-white-paired-v2'))
    p.add_argument('--device', default='mps'); p.add_argument('--repeats', type=int, default=3)
    a=p.parse_args(); cv2.setNumThreads(2); main(a.output,a.dataset,a.device,a.repeats)
