#!/usr/bin/env bash
# Run only after BOTH training jobs finish. Keeps test sealed until validation freeze.
set -euo pipefail
cd "$(dirname "$0")/.."
export YOLO_CONFIG_DIR="$PWD/artifacts/ultralytics"
export MPLCONFIGDIR="$PWD/artifacts/matplotlib"
export YOLO_AUTOINSTALL=false
python="artifacts/yolo-env/bin/python"
root="outputs/paired-model-comparison-v1"
if [[ ! -f "$root/full_val_ground_truth.json" ]]; then
    "$python" scripts/prepare_native_ground_truth.py --split val
fi
"$python" scripts/check_runtime_renderer.py --split val
for model in unet yolo; do
    while [[ -f "$root/${model}_validation_running" ]]; do sleep 5; done
    if [[ ! -f "$root/$model/raw_tile_val.jsonl" ]]; then
        "$python" scripts/evaluate_paired_models.py collect-tiles --model "$model"
    fi
    "$python" scripts/evaluate_paired_models.py calibrate-tiles --model "$model"
    if [[ ! -f "$root/$model/raw_full_val.jsonl" ]]; then
        "$python" scripts/evaluate_paired_models.py collect-full --model "$model" --split val
    fi
    "$python" scripts/evaluate_paired_models.py calibrate-full --model "$model"
done
# Optional local checkpoint for inspecting validation quality before opening test data.
if [[ -f "$root/pause_before_test" ]]; then
    touch "$root/validation_ready"
    printf 'Validation reports are ready; waiting for local experiment review.\n'
    while [[ -f "$root/pause_before_test" ]]; do sleep 5; done
fi
if [[ ! -f "$root/frozen_selection.json" ]]; then
    "$python" scripts/evaluate_paired_models.py freeze
fi
if [[ ! -f "$root/full_test_ground_truth.json" ]]; then
    "$python" scripts/prepare_native_ground_truth.py --split test
fi
"$python" scripts/check_runtime_renderer.py --split test
for model in unet yolo; do
    if [[ ! -f "$root/$model/raw_full_test.jsonl" ]]; then
        "$python" scripts/evaluate_paired_models.py collect-full --model "$model" --split test
    fi
done
"$python" scripts/evaluate_paired_models.py compare
"$python" scripts/evaluate_native_masks.py
"$python" scripts/benchmark_native_models.py
"$python" scripts/render_native_comparison.py
