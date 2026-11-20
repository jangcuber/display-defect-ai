"""Verify the real resident worker, raw crops and cached full-test/file-input parity."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
import cv2
import numpy as np


def verify(dataset,root,device):
    dataset=dataset.resolve();root=root.resolve()
    frozen=json.loads((root/'frozen_selection.json').read_text());kind=frozen['selected_model']
    reference={r['image_id']:r['detections'] for r in map(json.loads,(root/kind/'test_predictions.jsonl').read_text().splitlines())}
    rows=[r for r in map(json.loads,(dataset/'manifest.jsonl').read_text().splitlines()) if r['split']=='test']
    sid=rows[0]['source_case_id'];examples=[r for r in rows if r['source_case_id']==sid]
    qa=root/'app-verification';(qa/'inputs').mkdir(parents=True,exist_ok=True)
    full=[]
    for name in ('normal','mixed'):
        row=next(r for r in examples if r['kind']==name)
        image=qa/'inputs'/f"{row['image_id']}.png"
        if not image.exists():
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/export_oled_scene.py'),str(dataset),row['image_id'],'--output',str(image)],cwd=ROOT,check=True)
        full.append((row,image))
    env={**os.environ,'YOLO_CONFIG_DIR':str(ROOT/'artifacts/ultralytics'),'MPLCONFIGDIR':str(ROOT/'artifacts/matplotlib'),'YOLO_AUTOINSTALL':'false'}
    records=[];loads=0;buffer=b''
    with (qa/'worker.log').open('w') as log:
        process=subprocess.Popen([str(ROOT/'artifacts/yolo-env/bin/python'),str(ROOT/'scripts/inspect_native_photo.py'),'--server','--experiment',str(root),'--device',device],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0)
        def event(deadline):
            nonlocal buffer
            while b'\n' not in buffer:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Resident worker response timed out')
                if not select.select([process.stdout],[],[],remaining)[0]:raise TimeoutError('Resident worker response timed out')
                data=os.read(process.stdout.fileno(),65536)
                if not data:raise RuntimeError('Resident worker closed unexpectedly')
                buffer+=data
            line,buffer=buffer.split(b'\n',1)
            return json.loads(line)
        def request(identity,photo):
            nonlocal loads
            output=qa/'results'/identity
            process.stdin.write((json.dumps(dict(id=identity,photo=str(photo),output=str(output)))+'\n').encode())
            deadline=time.monotonic()+180;progress=[]
            while True:
                value=event(deadline)
                assert value['id']==identity,value
                if value['event']=='loading':loads+=1
                if value['event']=='progress':progress.append([value['done'],value['total']])
                if value['event']=='error':return value,None
                if value['event']=='result':
                    result=value['result']
                    assert progress and progress[-1][0]==progress[-1][1]
                    assert result['weights_sha256']==frozen['models'][kind]['weights_sha256']
                    assert result['inference_source_sha256']==frozen['inference_source_sha256']
                    assert result['decision']==('defect' if result['detections'] else 'normal'),result
                    image=cv2.imread(str(photo))
                    for mark in result['detections']:
                        x1,y1,x2,y2=mark['crop_box']
                        assert np.array_equal(cv2.imread(mark['crop']),image[y1:y2,x1:x2]),identity
                    assert (result['mask_preview'] is not None)==(kind=='unet')
                    return value,dict(id=identity,decision=result['decision'],detections=len(result['detections']),progress=progress[-1],seconds=result['seconds'])
        try:
            error,_=request('missing-photo',qa/'inputs/missing.png')
            assert error['event']=='error'
            for row in examples:
                value,record=request('tile-'+row['image_id'],dataset/row['image'])
                assert value['event']=='result',value
                record['generation_kind']=row['kind'];records.append(record)
            for row,path in full:
                value,record=request('full-'+row['image_id'],path)
                assert value['event']=='result',value
                observed=sorted(value['result']['detections'],key=lambda d:(d['kind'],*d['box']))
                expected=sorted(reference[row['image_id']],key=lambda d:(d['kind'],*d['box']))
                assert len(observed)==len(expected),(row['image_id'],observed,expected)
                for actual,saved in zip(observed,expected,strict=True):
                    assert actual['kind']==saved['kind']
                    assert np.allclose(actual['box'],saved['box'],atol=.0001,rtol=0),(actual,saved)
                    assert abs(actual['score']-saved['score'])<=.000001,(actual,saved)
                record['cached_test_predictions_match']=True;records.append(record)
            assert loads==1,loads
        finally:
            process.stdin.close()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
    report=dict(selected_model=kind,device=device,source_case_id=sid,worker_checks_passed=True,
                model_load_events=loads,error_recovery=True,raw_crops_exact=True,full_file_cached_predictions_match=True,
                weights_sha256=frozen['models'][kind]['weights_sha256'],cases=records,
                note='Worker functional verification; classification accuracy is reported separately on all1050 test scenes')
    (root/'app_worker_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/oled-white-paired-v2'));p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));p.add_argument('--device',default='mps');a=p.parse_args();verify(a.dataset,a.output,a.device)
