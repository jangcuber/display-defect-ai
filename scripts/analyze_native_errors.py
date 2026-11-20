"""Describe fixed-test errors without changing models, scores or thresholds."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from display_defect_ai.line_postprocess import box_iou


def analyze(root):
    reference = json.loads((root / 'test_comparison.json').read_text())
    truth = {r['image_id']: r['objects'] for r in
             json.loads((root / 'full_test_ground_truth.json').read_text())['rows']}
    result = dict(split='test', matching='same class, score order, one-to-one IoU >= 0.5',
                  purpose='Post-test description only; no parameter changes', models={})
    for model in ('unet', 'yolo'):
        misses, extras = [], []
        for row in map(json.loads, (root / model / 'test_predictions.jsonl').read_text().splitlines()):
            objects = truth[row['image_id']]
            detections = row['detections']
            claimed = set()
            for item in sorted(detections, key=lambda d: d['score'], reverse=True):
                overlap, index = max(((box_iou(item['box'], obj['bbox_xyxy']), i)
                                      for i, obj in enumerate(objects)
                                      if i not in claimed and item['kind'] == obj['kind']), default=(0, -1))
                if overlap >= .5:
                    claimed.add(index)
                else:
                    extras.append(dict(image_id=row['image_id'], detection=item))
            for i, obj in enumerate(objects):
                if i in claimed:
                    continue
                best = max((box_iou(d['box'], obj['bbox_xyxy']) for d in detections
                            if d['kind'] == obj['kind']), default=0)
                misses.append(dict(image_id=row['image_id'], object=obj, best_same_class_iou=best))
        by_kind = Counter(r['object']['kind'] for r in misses)
        fp = Counter(r['detection']['kind'] for r in extras)
        for kind, metrics in reference['models'][model]['metrics']['classes'].items():
            assert by_kind[kind] == metrics['fn'], (model, kind, by_kind, metrics)
            assert fp[kind] == metrics['fp'], (model, kind, fp, metrics)
        result['models'][model] = dict(
            missed_by_class=dict(by_kind), extra_by_class=dict(fp),
            missed_by_contrast_tier=dict(Counter(r['object']['attributes'].get('tier', 'unknown') for r in misses)),
            missed_by_channel_pattern=dict(Counter(r['object']['attributes'].get('channel_pattern', 'unknown') for r in misses)),
            misses=misses, extras=extras)
    (root / 'error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({m: {k: v for k, v in r.items() if k not in ('misses', 'extras')}
                      for m, r in result['models'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/paired-model-comparison-v1'))
    analyze(parser.parse_args().output)
