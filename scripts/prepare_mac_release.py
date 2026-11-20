"""Stage the verified U-Net, seven synthetic examples, and runtime notices."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'normal', 'dark_dot', 'color_dot', 'horizontal_line', 'vertical_line', 'dark_region', 'mixed'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(resources):
    target = resources / 'InspectorData'
    experiment_path = Path('outputs/paired-model-comparison-v1')
    experiment = ROOT / experiment_path
    frozen_file = experiment / 'frozen_selection.json'
    frozen = json.loads(frozen_file.read_text())
    if frozen['selected_model'] != 'unet':
        raise ValueError('This bundle contains the U-Net worker')
    weight = experiment / 'unet/best.pt'
    if sha(weight) != frozen['models']['unet']['weights_sha256']:
        raise ValueError('Deployment weights differ from the validation checkpoint')
    runtime = {**frozen['runtime_source_sha256'],
               'native_inference.py': frozen['inference_source_sha256']}
    for name, digest in runtime.items():
        path = Path('src/display_defect_ai') / name
        if sha(ROOT / path) != digest:
            raise ValueError(f'Runtime source differs from validation: {name}')
        copy(ROOT / path, target / path)
    copy(frozen_file, target / experiment_path / 'frozen_selection.json')
    copy(weight, target / experiment_path / 'unet/best.pt')

    data_path = Path('data/oled-white-paired-v2')
    dataset = ROOT / data_path
    if sha(dataset / 'manifest.jsonl') != frozen['dataset_manifest_sha256']:
        raise ValueError('Example manifest differs from the evaluation data')
    with (dataset / 'manifest.jsonl').open() as stream:
        test = [row for row in map(json.loads, stream) if row['split'] == 'test']
    source_id = test[0]['source_case_id']
    examples = [row for row in test if row['source_case_id'] == source_id]
    if len(examples) != len(KINDS) or {row['kind'] for row in examples} != KINDS:
        raise ValueError('Expected all seven example types from one panel')
    manifest = []
    for row in examples:
        source = dataset / row['image']
        if sha(source) != row['image_sha256']:
            raise ValueError(f'Example checksum mismatch: {source.name}')
        relative = Path('images') / source.name
        copy(source, target / data_path / relative)
        manifest.append(dict(split='test', kind=row['kind'], image=str(relative),
                             image_sha256=row['image_sha256'], is_synthetic=True))
    (target / data_path / 'manifest.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in manifest), encoding='utf-8')

    notices = resources / 'Third Party Notices'
    notices.mkdir()
    python_license = Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt'
    copy(python_license, notices / 'CPython-LICENSE.txt')
    versions = {'python': sys.version.split()[0]}
    for package in ('torch', 'numpy', 'opencv-python', 'pillow', 'PyInstaller', 'filelock',
                    'typing-extensions', 'sympy', 'networkx', 'jinja2', 'MarkupSafe', 'fsspec', 'mpmath'):
        dist = metadata.distribution(package)
        versions[package] = dist.version
        for entry in dist.files or ():
            if 'licenses' not in entry.parts and not entry.name.lower().startswith(('license', 'copying', 'notice')):
                continue
            if any(part == '..' for part in entry.parts):
                continue
            source = Path(dist.locate_file(entry))
            if source.is_file():
                copy(source, notices / package / entry)
    (notices / 'runtime-versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    copy(ROOT / 'docs/mac-release.md', notices / 'source-and-installation.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resources', type=Path)
    args = parser.parse_args()
    prepare(args.resources)
