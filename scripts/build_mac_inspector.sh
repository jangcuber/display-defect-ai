#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x artifacts/yolo-env/bin/python ||
      ! -f outputs/paired-model-comparison-v1/frozen_selection.json ]]; then
    printf '로컬 학습 환경과 U-Net/YOLO 비교 후 동결한 모델 선택 결과가 필요합니다.\n' >&2
    exit 1
fi
artifacts/yolo-env/bin/python - <<'PYTHON'
import hashlib, json
from pathlib import Path
root = Path('outputs/paired-model-comparison-v1')
frozen = json.loads((root / 'frozen_selection.json').read_text())
kind = frozen['selected_model']
weights = root / ('unet/best.pt' if kind == 'unet' else 'yolo/train/weights/best.pt')
assert hashlib.sha256(weights.read_bytes()).hexdigest() == frozen['models'][kind]['weights_sha256'], '검증된 가중치가 아닙니다.'
PYTHON

output="outputs/white-screen-inspector"
destination="$output/White Screen Inspector.app"
mkdir -p "$output" artifacts/swift-module-cache
staging="$(mktemp -d "$output/.build.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
app="$staging/White Screen Inspector.app"
mkdir -p "$app/Contents/MacOS"
cp macos/Info.plist "$app/Contents/Info.plist"
swiftc -parse-as-library -O -target "$(uname -m)-apple-macosx15.0" \
    -module-cache-path "$PWD/artifacts/swift-module-cache" \
    macos/WhiteScreenInspector.swift -o "$app/Contents/MacOS/WhiteScreenInspector"
codesign --force --sign - "$app"

if [[ -e "$destination" ]]; then
    mv "$destination" "$staging/previous.app"
fi
if ! mv "$app" "$destination"; then
    if [[ -e "$staging/previous.app" ]]; then
        mv "$staging/previous.app" "$destination"
    fi
    exit 1
fi
printf '%s\n' "$PWD/$destination"
