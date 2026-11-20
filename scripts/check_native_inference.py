"""Focused checks for native tiling, independent masks and line joining."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from display_defect_ai.line_postprocess import Detection
from display_defect_ai.native_inference import KINDS, components, merge_detections, tile_windows

thresholds = dict.fromkeys(KINDS, .5)
line = lambda box: Detection(.9, 'horizontal_line', box)
assert len(merge_detections([line((10, 30, 100, 33)), line((110, 30, 190, 33))], thresholds)) == 1
assert len(merge_detections([line((10, 30, 100, 33)), line((10, 35, 100, 38))], thresholds)) == 2
crossing = [line((0, 50, 100, 53)), Detection(.9, 'vertical_line', (50, 0, 53, 100))]
assert len(merge_detections(crossing, thresholds)) == 2
assert not merge_detections([Detection(.9, 'dark_dot', (0, 0, 80, 80))], thresholds)
for w, h in ((5888, 3368), (1024, 1024), (700, 600), (1500, 2100)):
    coverage = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in tile_windows(w, h):
        assert x2 - x1 <= 1024 and y2 - y1 <= 1024
        coverage[y1:y2, x1:x2] = True
    assert coverage.all()
plane = np.zeros((128, 128), np.uint8)
plane[20:23, 10:110] = 230
assert components(plane, 'horizontal_line', .5, 6)[0].box == (10., 20., 110., 23.)
assert not components(plane, 'dark_dot', .5, 3)
assert not components(plane, 'horizontal_line', 1.01, 6)
print('Native inference: coverage, crossing/parallel/broken lines, mask components and geometry passed')
from evaluate_paired_models import native_metrics
rows=[dict(objects=[dict(kind='horizontal_line',bbox_xyxy=[0,50,1000,53],attributes=dict(line_form='partial',thickness=1))])]
metric=native_metrics(rows,[[line((0,50,300,53)),line((400,50,700,53))]])
assert metric['classes']['horizontal_line']['duplicates']==0
assert metric['line_fragmentation']['lines_with_multiple_boxes']==1
assert metric['line_fragmentation']['surplus_aligned_boxes']==1
print('Fragmentation metric counts partial boxes that whole-box IoU duplication misses')

def uncropped_components(plane, kind, threshold, minimum):
    import cv2
    from display_defect_ai.native_inference import geometry_valid
    count,labels,stats,_=cv2.connectedComponentsWithStats((plane>=int(np.ceil(threshold*255))).astype(np.uint8),8)
    sums=np.bincount(labels.ravel(),weights=plane.ravel(),minlength=count)
    detections=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<minimum:continue
        item=Detection(float(sums[i]/area/255),kind,(float(x),float(y),float(x+w),float(y+h)))
        if geometry_valid(item):detections.append(item)
    return sorted(detections,key=lambda item:item.score,reverse=True)[:300]

rng=np.random.default_rng(43)
plane=rng.integers(0,40,(512,640),dtype=np.uint8)
plane[0:5,0:7]=160
plane[210:217,310:319]=190
plane[350:354,30:600]=220
plane[250:410,100:190]=200
for kind in KINDS:
    assert components(plane,kind,.5,3)==uncropped_components(plane,kind,.5,3)
assert components(np.full((128,128),200,np.uint8),'dark_region',.5,24)==uncropped_components(np.full((128,128),200,np.uint8),'dark_region',.5,24)
print('Cropped component analysis preserves boxes and scores exactly, including foreground at image edges')
region_parts=[Detection(.9,'dark_region',(0.,0.,365.,365.)),Detection(.88,'dark_region',(237.,0.,603.,365.)),Detection(.86,'dark_region',(0.,237.,365.,603.)),Detection(.84,'dark_region',(237.,237.,603.,603.))]
merged=merge_detections(region_parts,thresholds,region_iou=.15,merge_regions=True)
assert len(merged)==1 and merged[0].box==(0.,0.,603.,603.)
separate=[Detection(.9,'dark_region',(0.,0.,100.,100.)),Detection(.9,'dark_region',(200.,200.,300.,300.))]
assert len(merge_detections(separate,thresholds,region_iou=.15,merge_regions=True))==2
print('Region tile fragments can merge without joining separate areas; mode is selected on validation only')
cluster=dict(objects=[dict(kind='dark_dot',bbox_xyxy=[0,0,5,5]),dict(kind='dark_dot',bbox_xyxy=[20,20,25,25])])
cluster_result=native_metrics([cluster],[[Detection(.9,'dark_dot',(0.,0.,5.,5.)),Detection(.9,'dark_dot',(20.,20.,25.,25.))]])
assert cluster_result['multiple_defect_images']['total']==1 and cluster_result['mixed_type_images']['total']==0
mixed=dict(objects=[dict(kind='dark_dot',bbox_xyxy=[0,0,5,5]),dict(kind='vertical_line',bbox_xyxy=[50,0,53,100],attributes=dict(line_form='partial',thickness=1))])
mixed_result=native_metrics([mixed],[[Detection(.9,'dark_dot',(0.,0.,5.,5.)),Detection(.9,'vertical_line',(50.,0.,53.,100.))]])
assert mixed_result['mixed_type_images']==dict(total=1,all_objects_found=1)
print('Mixed-type success is separate from same-class clustered points')
