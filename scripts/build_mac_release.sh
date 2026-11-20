#!/usr/bin/env bash
# Package the evaluated white-screen inspector as a portable Apple Silicon app.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != arm64 || "$(uname -s)" != Darwin ]]; then
    printf 'Apple Silicon macOS에서 실행하세요.\n' >&2
    exit 1
fi
python="${OLED_RELEASE_PYTHON:-artifacts/release-mac-env/bin/python}"
if [[ ! -x "$python" ]]; then
    printf 'requirements/release-mac.txt를 설치한 환경을 OLED_RELEASE_PYTHON에 지정하세요.\n' >&2
    exit 1
fi
output="$PWD/outputs/mac-release"
mkdir -p "$output" artifacts/swift-module-cache
staging="$(mktemp -d "$output/.build.XXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
app="$staging/White Screen Inspector.app"
resources="$app/Contents/Resources"
mkdir -p "$resources" "$app/Contents/MacOS"

"$python" scripts/prepare_mac_release.py "$resources"
PYINSTALLER_CONFIG_DIR="$output/pyinstaller-cache" \
    "$python" -m PyInstaller --noconfirm --onedir --console --name inspector-worker \
    --target-architecture arm64 --paths "$PWD/src" \
    --exclude-module ultralytics --exclude-module matplotlib --exclude-module IPython \
    --distpath "$staging/worker-dist" --workpath "$staging/worker-build" --specpath "$staging" \
    scripts/inspect_native_photo.py
ditto "$staging/worker-dist/inspector-worker" "$resources/InspectorWorker"

cp macos/Info.plist "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleIdentifier io.github.jangcuber.displaydefect.inspector' "$app/Contents/Info.plist"
swiftc -parse-as-library -O -target arm64-apple-macosx15.0 \
    -module-cache-path "$PWD/artifacts/swift-module-cache" \
    macos/WhiteScreenInspector.swift -o "$app/Contents/MacOS/WhiteScreenInspector"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
ditto -c -k --sequesterRsrc --keepParent "$app" "$staging/WhiteScreenInspector-macos-arm64.zip"
mv -f "$staging/WhiteScreenInspector-macos-arm64.zip" "$output/WhiteScreenInspector-macos-arm64.zip"
printf '%s\n' "$output/WhiteScreenInspector-macos-arm64.zip"
