"""Check evaluation-runtime pixels against materialized PNGs before model evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from display_defect_ai import paired_renderer as renderer


def check(dataset,output,split):
    if split=='test' and not (output/'frozen_selection.json').exists():
        raise ValueError('Freeze validation selection before checking test evaluation inputs')
    sources={r['source_case_id']:r for r in map(json.loads,(dataset/'source_cases.jsonl').read_text().splitlines())}
    groups=defaultdict(list)
    for row in map(json.loads,(dataset/'manifest.jsonl').read_text().splitlines()):
        if row['split']==split:groups[row['source_case_id']].append(row)
    checked=0
    total=sum(len(rows) for rows in groups.values())
    for sid,rows in groups.items():
        source=sources[sid];window=source['tile_xyxy'];base=renderer.base_window(source,window)
        for row in rows:
            actual=renderer.render_from_base(source,row['scene_objects'],window,base)
            stored=cv2.imread(str(dataset/row['image']))
            if not np.array_equal(actual,stored):raise ValueError('Evaluation RGB replay differs: '+row['image_id'])
            checked+=1
        if checked%140==0:print(f'Evaluation runtime {split} RGB replay: {checked}/{total}',flush=True)
    result=dict(split=split,images=checked,pixel_exact=True,numpy=np.__version__,opencv=cv2.__version__,
        dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest())
    (output/f'runtime_{split}_render_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/oled-white-paired-v2'));p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));p.add_argument('--split',choices=('val','test'),default='val');a=p.parse_args();cv2.setNumThreads(2);check(a.dataset,a.output,a.split)
