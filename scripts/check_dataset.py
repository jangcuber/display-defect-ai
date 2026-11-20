"""Validate exported YOLO data and source-case isolation without training."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def check(root):
    records = [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    config = json.loads((root/'generation.json').read_text())
    classes = config.get('classes', ['dot', 'line'])
    image_hashes = set()
    assert records, 'Empty dataset'
    groups, ids, counts = {}, set(), {}
    for r in records:
        assert r['image_id'] not in ids, 'Duplicate image ID'
        ids.add(r['image_id'])
        groups.setdefault(r['source_case_id'],set()).add(r['split'])
        image = cv2.imread(str(root/r['image']))
        assert image is not None, r['image']
        h,w = image.shape[:2]
        rows = (root/r['label']).read_text().splitlines()
        objects = r.get('objects')
        expected = [obj['kind'] for obj in objects] if objects is not None else (
            [] if r['kind']=='normal' else [r['kind']])
        assert len(rows) == len(r['bbox_xyxy']) == len(expected)
        assert bool(rows) == (r['kind'] != 'normal')
        for row, box, kind in zip(rows,r['bbox_xyxy'],expected,strict=True):
            cls,x,y,bw,bh = map(float,row.split())
            assert cls == classes.index(kind)
            assert 0 <= x-bw/2 < x+bw/2 <= 1.00000001
            assert 0 <= y-bh/2 < y+bh/2 <= 1.00000001
            restored = [(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
            assert max(abs(a-b) for a,b in zip(restored,box)) < .001
        digest = hashlib.sha256(image.tobytes()).digest()
        assert digest not in image_hashes, f"Duplicate image pixels: {r['image_id']}"
        image_hashes.add(digest)
        assert (root/'previews'/r['split']/(r['image_id']+'.png')).is_file()
        key = r['split']+'/'+r['kind']
        counts[key] = counts.get(key,0)+1
    assert all(len(s)==1 for s in groups.values()), 'Source-case leakage'
    assert {r['split'] for r in records} == {'train','val','test'}
    assert len(list((root/'images').rglob('*.png'))) == len(records)
    assert len(list((root/'labels').rglob('*.txt'))) == len(records)
    print(json.dumps(dict(images=len(records),source_cases=len(groups),counts=counts,
                         source_case_leakage=False, duplicate_images=0),indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    check(p.parse_args().dataset)
