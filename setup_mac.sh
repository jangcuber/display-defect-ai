#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/generation.txt
.venv/bin/python scripts/check.py
.venv/bin/python scripts/check_dataset.py data/pilot-v1
.venv/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('data/pilot-v1/dataset.yaml')
lines=p.read_text().splitlines()
lines[0]='path: '+json.dumps(str(Path('data/pilot-v1').resolve()))
p.write_text('\n'.join(lines)+'\n')
PY
printf 'Setup complete. Preview: open data/pilot-v1/previews\n'
