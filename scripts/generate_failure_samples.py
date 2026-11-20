"""Render paired white and black examples of simplified RGB drive failures."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from generate_dataset import make_solid_panel, render_photo, write_image


@dataclass(frozen=True)
class Failure:
    name: str
    yolo_class: int | None
    channels: str = 'BGR'
    regions: tuple[tuple[int, int, int, int], ...] = ()
    remaining_output: float = 0.0
    held_high: bool = False


# These rectangles are camera-scale footprints in the simulated panel plane.
# They do not describe the physical pitch or layout of a particular panel.
POINT = ((307, 248, 313, 254),)
FAILURES = (
    Failure('pixel-off', 0, regions=POINT),
    Failure('red-subpixel-off', 0, 'R', POINT),
    Failure('green-subpixel-off', 0, 'G', POINT),
    Failure('blue-subpixel-off', 0, 'B', POINT),
    Failure('red-subpixel-held-high', 0, 'R', POINT, held_high=True),
    Failure('red-data-column-low', 1, 'R', ((344, 0, 347, 512),), .10),
    Failure('row-addressing-loss', 1, regions=((0, 190, 512, 194),), remaining_output=.28),
    Failure('local-drive-segments', 1, regions=((60, 340, 150, 343), (170, 340, 265, 343),
                                                (290, 340, 415, 343)), remaining_output=.40),
    Failure('supply-voltage-drop', None),
)
PATTERNS = {'white': 255, 'black': 0}
CHANNEL_INDEX = {'B': 0, 'G': 1, 'R': 2}


def apply_failure(panel, white_panel, failure):
    if failure.yolo_class is None:
        # Illustrative output falloff toward one edge, not a circuit solver.
        x = np.linspace(0, 1, panel.shape[1])
        loss = .20 * np.clip((x - .55) / .45, 0, 1)
        panel[:] = np.rint(panel * (1 - loss[None, :, None])).astype(np.uint8)
        return

    for x1, y1, x2, y2 in failure.regions:
        for channel in failure.channels:
            region = np.s_[y1:y2, x1:x2, CHANNEL_INDEX[channel]]
            if failure.held_high:
                panel[region] = white_panel[region]
            else:
                panel[region] = np.rint(panel[region] * failure.remaining_output).astype(np.uint8)


def visible_box(photo, normal):
    ys, xs = np.nonzero(np.any(photo != normal, axis=2))
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def generate(output, reference):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    config = json.loads((reference / 'generation.json').read_text())
    size = config['size']
    if size != 512 or not config.get('fixed_setup') or not config.get('normal_only'):
        raise ValueError('Reference must be a 512×512 fixed setup normal dataset')
    with (reference / 'manifest.jsonl').open() as stream:
        record = json.loads(next(stream))
    camera = record['camera']
    matrix = np.asarray(camera['homography'])
    noise = np.random.default_rng(camera['seed']).normal(0, camera['noise_std'], (size, size, 3))
    panels = {name: make_solid_panel(level, size, record['panel']) for name, level in PATTERNS.items()}
    normals = {name: render_photo(panel, matrix, camera['blur_sigma'], noise) for name, panel in panels.items()}
    if not np.array_equal(normals['white'], cv2.imread(str(reference / record['image']))):
        raise ValueError('White reference could not be replayed exactly')

    output.mkdir(parents=True)
    rows = []
    for pattern, base_panel in panels.items():
        normal_path = Path('images') / pattern / 'normal.png'
        (output / normal_path.parent).mkdir(parents=True)
        write_image(output / normal_path, normals[pattern])
        for failure in FAILURES:
            panel = base_panel.copy()
            apply_failure(panel, panels['white'], failure)
            photo = render_photo(panel, matrix, camera['blur_sigma'], noise)
            box = visible_box(photo, normals[pattern])
            photo_path = normal_path.with_name(f'{failure.name}.png')
            write_image(output / photo_path, photo)
            label = None
            if box and failure.yolo_class is not None:
                x1, y1, x2, y2 = box
                label = Path('labels') / pattern / f'{failure.name}.txt'
                (output / label.parent).mkdir(parents=True, exist_ok=True)
                values = ((x1 + x2) / (2 * size), (y1 + y2) / (2 * size),
                          (x2 - x1) / size, (y2 - y1) / size)
                (output / label).write_text(
                    f'{failure.yolo_class} ' + ' '.join(f'{value:.8f}' for value in values) + '\n',
                    encoding='utf-8',
                )
            rows.append(dict(
                scenario=failure.name, pattern=pattern, visible=box is not None,
                affected_channels=failure.channels, regions_xyxy=failure.regions,
                remaining_output=None if failure.held_high or failure.yolo_class is None else failure.remaining_output,
                held_high=failure.held_high, yolo_class=failure.yolo_class, bbox_xyxy=box,
                image=photo_path.as_posix(), normal_image=normal_path.as_posix(),
                label=label.as_posix() if label else None,
                source_case_id=record['source_case_id'], is_synthetic=True,
            ))
    (output / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8',
    )
    (output / 'reference.json').write_text(
        json.dumps(record, indent=2) + '\n', encoding='utf-8',
    )
    print(f'{len(FAILURES)} scenarios, {len(rows)} captures, {len(panels)} matched normals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.reference)
