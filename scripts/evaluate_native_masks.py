"""Additional U-Net pixel metrics on existing test tiles; YOLO has no pixel masks."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from display_defect_ai.native_inference import KINDS,NativeEngine
from evaluate_paired_models import check_frozen,paths,read_rows


def evaluate(dataset,root,device):
    frozen=check_frozen(dataset,root,'unet')
    selected=frozen['models']['unet']['selected'];run,weights=paths(root,'unet')
    engine=NativeEngine('unet',weights,device)
    tp=np.zeros(5,np.int64);predicted=tp.copy();actual=tp.copy()
    rows=read_rows(dataset,'test')
    for index,row in enumerate(rows):
        image=cv2.imread(str(dataset/row['image']))
        mask=cv2.imread(str(dataset/'masks/test'/f"{row['image_id']}.png"),0)
        maps=engine.predict_tiles([image])[0]
        for i,kind in enumerate(KINDS):
            gt=(mask & (1<<i))!=0
            pred=maps[i]>=int(np.ceil(selected['pixel_thresholds'][kind]*255))
            tp[i]+=np.count_nonzero(gt&pred);predicted[i]+=np.count_nonzero(pred);actual[i]+=np.count_nonzero(gt)
        if (index+1)%100==0:print(f'U-Net test pixel metrics: {index+1}/{len(rows)}',flush=True)
    classes={kind:dict(true_positive_pixels=int(tp[i]),predicted_pixels=int(predicted[i]),actual_pixels=int(actual[i]),
            precision=float(tp[i]/max(1,predicted[i])),recall=float(tp[i]/max(1,actual[i])),
            dice=float(2*tp[i]/max(1,predicted[i]+actual[i])),iou=float(tp[i]/max(1,predicted[i]+actual[i]-tp[i])))
            for i,kind in enumerate(KINDS)}
    result=dict(split='test',mode='native_1024_tiles',images=len(rows),classes=classes,
        pixel_thresholds=selected['pixel_thresholds'],weights_sha256=frozen['models']['unet']['weights_sha256'],
        dataset_manifest_sha256=frozen['dataset_manifest_sha256'],mask_manifest_sha256=frozen['mask_manifest_sha256'],
        note='Additional pixel metrics, distinct from common full-scene box matching; YOLO does not output pixel masks')
    (run/'test_pixel_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/oled-white-paired-v2'));p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));p.add_argument('--device',default='mps');a=p.parse_args();cv2.setNumThreads(2);evaluate(a.dataset,a.output,a.device)
