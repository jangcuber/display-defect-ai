"""Verify the archived worker from a relocated app without the source checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verify(archive, location):
    subprocess.run(['ditto', '-x', '-k', archive, location], check=True)
    app = location / 'White Screen Inspector.app'
    subprocess.run(['codesign', '--verify', '--deep', '--strict', app], check=True)
    resources = app / 'Contents/Resources'
    worker = resources / 'InspectorWorker/inspector-worker'
    root = resources / 'InspectorData'
    data = root / 'data/oled-white-paired-v2'
    examples = [json.loads(line) for line in (data / 'manifest.jsonl').read_text().splitlines()]
    expected = {
        'normal': set(), 'dark_dot': {'dark_dot'}, 'color_dot': {'color_dot'},
        'horizontal_line': {'horizontal_line'}, 'vertical_line': {'vertical_line'},
        'dark_region': {'dark_region'}, 'mixed': {'dark_dot', 'horizontal_line', 'dark_region'},
    }
    if {row['kind'] for row in examples} != expected.keys() or len(examples) != len(expected):
        raise ValueError('Bundled sample manifest is missing examples or contains duplicates')
    requests = []
    for row in examples:
        photo = data / row['image']
        if hashlib.sha256(photo.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError(f'Packaged sample changed: {photo.name}')
        requests.append(dict(id=row['kind'], photo=str(photo), output=str(location / row['kind'])))
    completed = subprocess.run([worker, '--server', '--device', 'cpu', '--resource-root', root],
                               cwd=location, env={'PATH': '/usr/bin:/bin', 'LANG': 'en_US.UTF-8'},
                               input=''.join(json.dumps(r) + '\n' for r in requests),
                               text=True, capture_output=True, timeout=180)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    errors = [event for event in events if event['event'] == 'error']
    if errors:
        raise RuntimeError(errors)
    results = {event['id']: event['result'] for event in events if event['event'] == 'result'}
    if results.keys() != expected.keys():
        raise ValueError('Missing inspection results')
    for kind, names in expected.items():
        result = results[kind]
        got = {detection['kind'] for detection in result['detections']}
        if (got != names or result['decision'] != ('normal' if not names else 'defect')
                or result['selected_model'] != 'unet' or not Path(result['preview']).is_file()):
            raise ValueError(f'{kind}: unexpected packaged result {got}')
        print(f'{kind}: {result["decision"]}; {len(result["detections"])} detections')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--extract-to', type=Path)
    args = parser.parse_args()
    if args.extract_to:
        args.extract_to.mkdir(parents=True, exist_ok=False)
        verify(args.archive.resolve(), args.extract_to.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='OLED release check ') as folder:
            verify(args.archive.resolve(), Path(folder))
