"""Materialize full-scene evaluation boxes once, independently of either model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from display_defect_ai import paired_renderer as renderer
from evaluate_paired_models import full_objects


def prepare(dataset,root,split):
    if split=='test' and not (root/'frozen_selection.json').exists():
        raise ValueError('Freeze validation selection before preparing full-scene test evaluation')
    destination=root/f'full_{split}_ground_truth.json'
    if destination.exists():raise FileExistsError(destination)
    rows=[r for r in map(json.loads,(dataset/'manifest.jsonl').read_text().splitlines()) if r['split']==split]
    sources={r['source_case_id']:r for r in map(json.loads,(dataset/'source_cases.jsonl').read_text().splitlines())}
    result=[]
    for i,row in enumerate(rows):
        result.append(dict(image_id=row['image_id'],objects=full_objects(renderer,sources[row['source_case_id']],row['scene_objects'])))
        if (i+1)%100==0:print(f'Full {split} ground truth: {i+1}/{len(rows)}',flush=True)
    config=dict(dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest(),
                renderer_sha256=hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),rows=result)
    pending=destination.with_suffix('.pending');pending.write_text(json.dumps(config)+'\n');pending.replace(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/oled-white-paired-v2'));p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));p.add_argument('--split',choices=('val','test'),default='val');a=p.parse_args();prepare(a.dataset,a.output,a.split)
