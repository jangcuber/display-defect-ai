"""Deterministic synthetic panel/camera simulator. Not evidence of real accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def write_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'PNG encoding failed: {path}')
    path.write_bytes(encoded.tobytes())


def render_photo(panel, matrix, sigma, noise):
    size = panel.shape[0]
    photo = cv2.warpPerspective(panel, matrix, (size, size), borderValue=(24, 24, 24))
    photo = cv2.GaussianBlur(photo, (5, 5), sigma)
    return np.clip(photo.astype(float) + noise, 0, 255).astype(np.uint8)


def make_solid_panel(level, size, settings):
    x = np.arange(size, dtype=np.float32) / (size - 1)
    tint = np.asarray(settings['tint_bgr'])
    values = level * settings['gain'] * tint * (1 + settings['gradient'] * x[:, None]) + settings['offset']
    row = np.clip(values, 0, 255).astype(np.uint8)
    return np.broadcast_to(row, (size, size, 3)).copy()


def draw_defect_mask(mask, kind, rng, *, horizontal=True, line_form='partial'):
    size = mask.shape[0]
    if kind == 'dot':
        radius = max(1, round(int(rng.integers(1, 8)) * size / 512))
        margin = radius + 3
        x, y = (int(v) for v in rng.integers(margin, size - margin, 2))
        cv2.circle(mask, (x, y), radius, 255, -1)
        return dict(center=[x, y], radius=radius)

    broken = line_form == 'broken'
    if line_form == 'full':
        length = size
    else:
        length = round(size * rng.uniform(.15, .88))
    thickness = max(1, round(int(rng.integers(1, 6)) * size / 512))
    start = 0 if line_form == 'full' else int(rng.integers(3, size - length - 2))
    cross = int(rng.integers(3, size - thickness - 2))
    segment_count = int(rng.integers(2, 4)) if broken else 1
    gap = max(2, round(length * rng.uniform(.06, .14))) if broken else 0
    painted_length = length - gap * (segment_count - 1)
    segments = []
    for segment in range(segment_count):
        first = start + segment * painted_length // segment_count + segment * gap
        end = start + (segment + 1) * painted_length // segment_count + segment * gap
        if horizontal:
            mask[cross:cross + thickness, first:end] = 255
        else:
            mask[first:end, cross:cross + thickness] = 255
        axis = cross + thickness // 2
        segments.append([[first, axis], [end - 1, axis]] if horizontal else
                        [[axis, first], [axis, end - 1]])
    return dict(
        orientation='horizontal' if horizontal else 'vertical',
        length=length, thickness=thickness, broken=broken, segments=segments,
    )


def draw_varied_defect(panel, mask, kind, rng, sample_index):
    defect = draw_defect_mask(
        mask, kind, rng, horizontal=sample_index % 2 == 0,
        line_form='broken' if (sample_index // 2) % 2 == 1 else 'partial',
    )

    # Blend toward black or white relative to the local panel, preserving its gradient.
    bright = (sample_index // 4) % 2 == 1
    target = 255 if bright else 0
    strength = rng.uniform(.35, .95) if bright else rng.uniform(.04, .60)
    selected = mask > 0
    original = panel[selected].astype(float)
    changed = np.rint(original + strength * (target - original))
    panel[selected] = changed.astype(np.uint8)
    defect.update(polarity='bright' if bright else 'dark', mean_delta=float((changed - original).mean()))
    return defect


def generate(
    output, content=None, seed=42, cases=30, variants=2, size=512, *,
    normal_only=False, fixed_setup=False, varied_defects=False,
):
    if cases < 10 or variants < 1 or size < 128:
        raise ValueError('cases >= 10, variants >= 1, size >= 128 required')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if content is None:
        source = np.full((size, size, 3), 180, np.uint8)
        content_id = 'builtin-gray-180-v1'
    else:
        raw = content.read_bytes()
        source = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError('Cannot decode content image')
        content_id = hashlib.sha256(raw).hexdigest()
    source = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    # A separate stream keeps the paired normal images independent of defect choices.
    defect_rng = np.random.default_rng([seed, 1]) if varied_defects else None
    # Case assignment occurs BEFORE normal/defect/camera variants are generated.
    order = rng.permutation(cases)
    train_end = int(cases * .7)
    val_end = train_end + max(1, int(cases * .15))
    split_for = {int(i): ('train' if n < train_end else 'val' if n < val_end else 'test')
                 for n, i in enumerate(order)}
    kinds = ('normal',) if normal_only else ('normal', 'dot', 'line')
    records = []
    yy, xx = np.mgrid[:size, :size].astype(np.float32) / (size - 1)
    edge = size - 1
    margin = size * .055
    source_corners = np.float32([[0, 0], [edge, 0], [edge, edge], [0, edge]])
    centered_corners = np.float32([
        [margin, margin], [edge - margin, margin],
        [edge - margin, edge - margin], [margin, edge - margin],
    ])
    started = time.perf_counter()
    for case in range(cases):
        split = split_for[case]
        if fixed_setup:
            # Keep white below clipping, with small differences between panels.
            gain = float(rng.uniform(.90, .94))
            offset = 0.0
            tint = np.ones(3)
            gradient = float(rng.uniform(-.01, .01))
        else:
            gain = float(rng.uniform(.85, 1.1))
            offset = float(rng.uniform(-8, 8))
            tint = rng.uniform(.96, 1.04, 3)
            gradient = float(rng.uniform(-.08, .08))
        panel = np.clip(source.astype(float) * gain * tint * (1 + gradient * xx[..., None]) + offset, 0, 255).astype(np.uint8)
        for variant in range(variants):
            # Reuse camera conditions across normal/dot/line for this case/variant.
            camera_seed = int(rng.integers(0, 2**32))
            camera = np.random.default_rng(camera_seed)
            if fixed_setup:
                corners = centered_corners
                sigma = .55
                noise_std = 1.0
            else:
                corners = centered_corners + camera.uniform(-size*.015, size*.015, (4, 2)).astype(np.float32)
                sigma = float(camera.uniform(.3, .8))
                noise_std = float(camera.uniform(.5, 2))
            matrix = cv2.getPerspectiveTransform(source_corners, corners)
            noise = camera.normal(0, noise_std, (size, size, 3))
            for kind in kinds:
                panel_defect = panel.copy()
                mask = np.zeros((size, size), np.uint8)
                defect = None
                if kind != 'normal' and defect_rng is not None:
                    defect = draw_varied_defect(panel_defect, mask, kind, defect_rng, case * variants + variant)
                elif kind != 'normal':
                    x, y = (int(v) for v in rng.integers(size//8, size*7//8, 2))
                    color = int(rng.choice([0, 255]))
                    if kind == 'dot':
                        radius = int(rng.integers(2, max(3, size//90)))
                        cv2.circle(mask, (x,y), radius, 255, -1)
                        defect = dict(center=[x,y], radius=radius, intensity=color)
                    else:
                        thickness = int(rng.integers(1, max(2, size//120)))
                        horizontal = bool(rng.integers(0, 2))
                        end = (size-1-x, y) if horizontal else (x, size-1-y)
                        # Ensure a visible long line even when sampled near center.
                        start = (size//8, y) if horizontal else (x, size//8)
                        end = (size*7//8, y) if horizontal else (x, size*7//8)
                        cv2.line(mask, start, end, 255, thickness)
                        defect = dict(start=start, end=end, thickness=thickness, intensity=color)
                    panel_defect[mask > 0] = color
                photo = render_photo(panel_defect, matrix, sigma, noise)
                boxes = []
                if kind != 'normal':
                    # A thin axis-aligned line can fall between destination pixel centers.
                    interpolation = cv2.INTER_LINEAR if varied_defects else cv2.INTER_NEAREST
                    warped = cv2.warpPerspective(mask, matrix, (size,size), flags=interpolation)
                    # Include blur footprint; bbox uses exclusive upper pixel edge.
                    warped = cv2.dilate(warped, np.ones((5,5),np.uint8))
                    ys, xs = np.nonzero(warped)
                    if not len(xs):
                        raise RuntimeError('Defect disappeared in projection')
                    boxes = [[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]]
                image_id = f'case{case:04d}_v{variant:02d}_{kind}'
                for folder in ('images', 'labels'):
                    (output / folder / split).mkdir(parents=True, exist_ok=True)
                image_rel = f'images/{split}/{image_id}.png'
                label_rel = f'labels/{split}/{image_id}.txt'
                write_image(output / image_rel, photo)
                rows = []
                annotated = photo.copy()
                for x1,y1,x2,y2 in boxes:
                    cls = 0 if kind == 'dot' else 1
                    rows.append(f'{cls} {(x1+x2)/2/size:.8f} {(y1+y2)/2/size:.8f} {(x2-x1)/size:.8f} {(y2-y1)/size:.8f}')
                    cv2.rectangle(annotated,(x1,y1),(x2-1,y2-1),(0,0,255),1)
                cv2.putText(annotated, image_id, (8,20), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                (output / label_rel).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
                (output / 'previews' / split).mkdir(parents=True, exist_ok=True)
                write_image(output / 'previews' / split / f'{image_id}.png', annotated)
                records.append(dict(image_id=image_id, source_case_id=f'case{case:04d}', content_id=content_id,
                    split=split, kind=kind, image=image_rel, label=label_rel, bbox_xyxy=boxes,
                    panel=dict(gain=gain,offset=offset,tint_bgr=tint.tolist(),gradient=gradient),
                    camera=dict(seed=camera_seed,homography=matrix.tolist(),blur_sigma=sigma,noise_std=noise_std),
                    defect=defect, is_synthetic=True))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    # Absolute path is generated for this location; regenerate/update when transferring.
    (output / 'dataset.yaml').write_text(f'path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: dot\n  1: line\n',encoding='utf-8')
    config = dict(generator_version=1,seed=seed,cases=cases,variants=variants,size=size,content_id=content_id,
                  images=len(records),numpy=np.__version__,opencv=cv2.__version__,seconds=time.perf_counter()-started)
    if normal_only:
        config['normal_only'] = True
    if fixed_setup:
        config['fixed_setup'] = True
    if varied_defects:
        config['varied_defects'] = True
    (output / 'generation.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(config,indent=2))
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--content',type=Path,help='Image displayed on the simulated panel; default gray')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cases',type=int,default=30)
    p.add_argument('--variants',type=int,default=2)
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--normal-only', action='store_true', help='Generate normal images without dot or line defects')
    p.add_argument('--fixed-setup', action='store_true', help='Use fixed framing and controlled lighting for comparisons')
    p.add_argument('--varied-defects', action='store_true', help='Vary dot contrast and size, and include interrupted lines')
    a = p.parse_args()
    generate(
        a.output, a.content, a.seed, a.cases, a.variants, a.size,
        normal_only=a.normal_only, fixed_setup=a.fixed_setup, varied_defects=a.varied_defects,
    )
