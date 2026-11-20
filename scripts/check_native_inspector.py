"""Check that the app worker preserves raw pixels and does not invent YOLO masks."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from inspect_native_photo import Inspector, fixed_frame_aligned, emit
from display_defect_ai.line_postprocess import Detection
from display_defect_ai.native_inference import KINDS


class Engine:
    def predict_image(self, image, selected, roi, progress):
        if progress: progress(1, 1)
        if selected.get('test_line'):
            maps = np.zeros((5, *image.shape[:2]), np.uint8)
            maps[2, 500:503, 100:200] = 235
            maps[2, 500:503, 300:400] = 235
            return [Detection(.92, 'horizontal_line', (100.,500.,400.,503.))], maps
        marks = [Detection(.91, 'color_dot', (300.25, 400.5, 305.75, 405.5))]
        maps = None
        if selected.get('test_masks'):
            maps = np.zeros((5, *image.shape[:2]), np.uint8)
            maps[1, 400:405, 300:305] = 235
        return marks, maps


with tempfile.TemporaryDirectory(prefix='native-inspector-check-') as temporary:
    root = Path(temporary)
    rng = np.random.default_rng(43)
    image = np.clip(rng.normal(231, .6, (1024, 1024, 3)), 0, 255).astype(np.uint8)
    image[400:405, 300:305, 1] -= 8
    photo = root / 'photo.png'; cv2.imwrite(str(photo), image)
    inspector = Inspector.__new__(Inspector)
    inspector.engine = Engine(); inspector.kind = 'yolo'; inspector.name = 'test fixture'; inspector.device = 'cpu'
    inspector.weights_sha = 'test-fixture'; inspector.inference_sha = 'test-fixture'
    inspector.selected = {}
    result = inspector.inspect(photo, root / 'yolo')
    assert result['decision'] == 'defect' and result['mask_preview'] is None
    assert not (root / 'yolo/pixel_mask.png').exists()
    mark = result['detections'][0]; x1,y1,x2,y2 = mark['crop_box']
    assert np.array_equal(cv2.imread(mark['crop']), image[y1:y2,x1:x2])
    assert mark['box'] == [300.25,400.5,305.75,405.5]
    assert mark['polygon'][0] == [300.25/1024,400.5/1024]
    inspector.kind = 'unet'; inspector.selected = dict(test_masks=True,pixel_thresholds=dict.fromkeys(KINDS,.5))
    result = inspector.inspect(photo, root / 'unet')
    mask = cv2.imread(str(root / 'unet/pixel_mask.png'), 0)
    assert np.count_nonzero(mask) == 25 and mask[402,302] == 2
    inspector.selected['test_line'] = True
    line_result = inspector.inspect(photo, root / 'broken-line')
    line = line_result['detections'][0]
    assert line['crop_anchor'] == [300,501]
    assert cv2.imread(line['crop']).shape[:2] == (128,128)
    inspector.selected.pop('test_line')
    other = root / 'other.png';cv2.imwrite(str(other), image[:800,:900])
    assert inspector.inspect(other, root / 'review')['decision'] == 'review'
    protocol, diagnostics = io.StringIO(), io.StringIO()
    with patch.object(sys, '__stdout__', protocol), redirect_stdout(diagnostics):
        inspector.inspect(photo, root / 'protocol',
                          lambda done, total: emit(dict(event='progress', done=done, total=total)))
    assert json.loads(protocol.getvalue()) == dict(event='progress', done=1, total=1)
    assert diagnostics.getvalue() == ''
print('Native inspector: raw crops, coordinates, independent class masks, YOLO absence of masks and review state passed')
print('Progress protocol survives redirected model diagnostics')

frame = np.full((3368,5888,3),230,np.uint8)
frame[:64]=24;frame[-64:]=24;frame[:,:64]=24;frame[:,-64:]=24
assert fixed_frame_aligned(frame)
frame[100:103,64:-64] = 0  # a real interior dark line must not be mistaken for the frame edge
assert fixed_frame_aligned(frame)
assert not fixed_frame_aligned(np.roll(frame,1,axis=1))
assert not fixed_frame_aligned(np.roll(frame,10,axis=1))
del frame
print('Registered framing: exact border accepted; translated panel rejected for review')
