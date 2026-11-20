"""Build paired white/black YOLO data from a fixed normal dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from generate_dataset import draw_defect_mask, make_solid_panel, render_photo, write_image


CLASS_IDS = {'dot': 0, 'line': 1}
PATTERNS = {'white': 255, 'black': 0}
CHANNEL_INDEX = {'B': 0, 'G': 1, 'R': 2}
MIN_PEAK_DELTA = 12
MIN_PIXEL_DELTA = 8
MIN_AFFECTED_PIXELS = 2


def apply_drive_fault(panel, white_panel, mask, channels, strength, held_high):
    result = panel.copy()
    pixels = panel[mask].astype(float)
    target = white_panel[mask] if held_high else 0
    changed = np.rint(pixels + strength * (target - pixels)).astype(np.uint8)
    indices = [CHANNEL_INDEX[channel] for channel in channels]
    pixels[:, indices] = changed[:, indices]
    result[mask] = pixels.astype(np.uint8)
    return result


def make_fault(pattern, kind, panels, normals, capture, rng):
    held_high = pattern == 'black'
    other_pattern = 'white' if held_high else 'black'
    size = panels[pattern].shape[0]
    horizontal = bool(rng.integers(0, 2)) if kind == 'line' else False
    line_form = str(rng.choice(('full', 'partial', 'broken'))) if kind == 'line' else None
    channels = 'RGB' if kind == 'line' and horizontal and line_form == 'full' else str(
        rng.choice(('R', 'G', 'B', 'RGB')),
    )
    if kind == 'dot':
        location = 'pixel' if channels == 'RGB' else 'subpixel'
    elif line_form == 'broken':
        location = 'local_drive_segments'
    else:
        location = 'row' if horizontal else 'data_column'

    contrast_ranges = ((.07, .20), (.20, .55), (.55, 1.0))
    contrast_tier = int(rng.integers(0, len(contrast_ranges)))
    for _ in range(24):
        footprint = np.zeros((size, size), np.uint8)
        geometry = draw_defect_mask(footprint, kind, rng, horizontal=horizontal, line_form=line_form)
        mask = footprint != 0
        strength = float(rng.uniform(*contrast_ranges[contrast_tier]))
        if contrast_tier == 2 and rng.random() < .2:
            strength = 1.0
        failed_panel = apply_drive_fault(panels[pattern], panels['white'], mask, channels, strength, held_high)
        photo = capture(failed_panel)
        delta = np.abs(photo.astype(np.int16) - normals[pattern]).max(axis=2)
        peak = int(delta.max())
        pixel_count = int(np.count_nonzero(delta >= MIN_PIXEL_DELTA))
        if peak >= MIN_PEAK_DELTA and pixel_count >= MIN_AFFECTED_PIXELS:
            break
    else:
        raise ValueError(f'Could not produce a visible {pattern} {kind}')

    hidden_panel = apply_drive_fault(panels[other_pattern], panels['white'], mask, channels, strength, held_high)
    if not np.array_equal(hidden_panel, panels[other_pattern]):
        raise ValueError(f'{pattern} {kind} changed the complementary pattern')
    ys, xs = np.nonzero(delta)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    defect = dict(
        **geometry, location=location, line_form=line_form, affected_channels=channels,
        drive_effect='held_high' if held_high else 'output_loss', strength=strength,
        peak_delta=peak, pixels_at_minimum_delta=pixel_count,
    )
    return photo, box, defect, other_pattern


def export_capture(root, reference_row, pattern, kind, photo, box=None, defect=None, original=None):
    split = reference_row['split']
    image_id = f"{reference_row['image_id'].removesuffix('_normal')}_{pattern}_{kind}"
    image = Path('images') / split / f'{image_id}.png'
    label = Path('labels') / split / f'{image_id}.txt'
    preview = Path('previews') / split / f'{image_id}.png'
    if original is None:
        write_image(root / image, photo)
    else:
        shutil.copyfile(original, root / image)

    annotated = photo.copy()
    label_text = ''
    if box:
        x1, y1, x2, y2 = box
        h, w = photo.shape[:2]
        coordinates = ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h)
        label_text = f'{CLASS_IDS[kind]} ' + ' '.join(f'{value:.8f}' for value in coordinates) + '\n'
        cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 1)
    (root / label).write_text(label_text, encoding='utf-8')
    write_image(root / preview, annotated)
    return dict(
        image_id=image_id, source_case_id=reference_row['source_case_id'], split=split,
        content_id=reference_row['content_id'] if pattern == 'white' else 'builtin-black-0-v1',
        kind=kind, pattern=pattern, image=image.as_posix(), label=label.as_posix(),
        bbox_xyxy=[box] if box else [], panel=reference_row['panel'], camera=reference_row['camera'],
        defect=defect, is_synthetic=True,
    )


def build_dataset(root, output, reference, reference_rows, seed):
    rng = np.random.default_rng(seed)
    for folder in ('images', 'labels', 'previews'):
        for split in ('train', 'val', 'test'):
            (root / folder / split).mkdir(parents=True)

    counts = Counter()
    started = time.perf_counter()
    with (root / 'manifest.jsonl').open('w', encoding='utf-8') as manifest, (
        root / 'visibility.jsonl'
    ).open('w', encoding='utf-8') as visibility:
        for index, record in enumerate(reference_rows, 1):
            camera = record['camera']
            matrix = np.asarray(camera['homography'])
            noise = np.random.default_rng(camera['seed']).normal(0, camera['noise_std'], (512, 512, 3))

            def capture(panel):
                return render_photo(panel, matrix, camera['blur_sigma'], noise)

            panels = {pattern: make_solid_panel(level, 512, record['panel']) for pattern, level in PATTERNS.items()}
            normals = {pattern: capture(panel) for pattern, panel in panels.items()}
            original = reference / record['image']
            if not np.array_equal(normals['white'], cv2.imread(str(original))):
                raise ValueError(f"Could not replay white normal {record['image_id']}")
            normal_rows = {
                pattern: export_capture(root, record, pattern, 'normal', photo,
                                        original=original if pattern == 'white' else None)
                for pattern, photo in normals.items()
            }
            captures = list(normal_rows.values())
            for pattern in PATTERNS:
                for kind in CLASS_IDS:
                    photo, box, defect, hidden = make_fault(pattern, kind, panels, normals, capture, rng)
                    row = export_capture(root, record, pattern, kind, photo, box, defect)
                    captures.append(row)
                    visibility.write(json.dumps(dict(
                        image_id=row['image_id'], source_case_id=row['source_case_id'],
                        visible_pattern=pattern, hidden_pattern=hidden,
                        hidden_reference_image=normal_rows[hidden]['image'],
                    )) + '\n')
            for row in captures:
                counts[f"{row['split']}/{row['pattern']}/{row['kind']}"] += 1
                manifest.write(json.dumps(row) + '\n')
            if index % 100 == 0 or index == len(reference_rows):
                print(f'{index}/{len(reference_rows)} panels exported', flush=True)

    (root / 'dataset.yaml').write_text(
        f'path: {json.dumps(str(output.resolve()))}\n'
        'train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: dot\n  1: line\n',
        encoding='utf-8',
    )
    config = dict(
        generator='paired-drive-faults-v1', seed=seed, size=512, reference=reference.name,
        source_cases=len(reference_rows), images=sum(counts.values()), counts=dict(sorted(counts.items())),
        visibility=dict(minimum_peak_delta=MIN_PEAK_DELTA, minimum_pixel_delta=MIN_PIXEL_DELTA,
                        minimum_affected_pixels=MIN_AFFECTED_PIXELS),
        numpy=np.__version__, opencv=cv2.__version__, seconds=time.perf_counter() - started,
    )
    (root / 'generation.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    return config


def generate(output, reference, seed):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    config = json.loads((reference / 'generation.json').read_text())
    if config['size'] != 512 or not config.get('fixed_setup') or not config.get('normal_only'):
        raise ValueError('Reference must be a 512×512 fixed setup normal dataset')
    reference_rows = [json.loads(line) for line in (reference / 'manifest.jsonl').read_text().splitlines()]
    if (not reference_rows or len(reference_rows) != config['images']
            or any(row['kind'] != 'normal' for row in reference_rows)
            or len({row['source_case_id'] for row in reference_rows}) != len(reference_rows)):
        raise ValueError('Expected one normal capture per reference case')

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f'.{output.name}-') as workspace:
        root = Path(workspace) / 'dataset'
        result = build_dataset(root, output, reference, reference_rows, seed)
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
        root.rename(output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.reference, args.seed)
