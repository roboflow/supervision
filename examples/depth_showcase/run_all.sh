#!/usr/bin/env bash
# Generate one depth video per model for every sample video of supervision into outputs/.
# Usage: ./run_all.sh            (full length, all models, device picked automatically)
#        DEVICE=cuda MODELS='["yolo26"]' MAX_SECONDS=8 ./run_all.sh
set -uo pipefail
cd "$(dirname "$0")"

if [ -z "${DEVICE:-}" ]; then
    if [ "$(uname)" = "Darwin" ]; then DEVICE=mps
    elif command -v nvidia-smi > /dev/null 2>&1; then DEVICE=cuda
    else DEVICE=cpu
    fi
fi

extra_args=()
[ -n "${MODELS:-}" ] && extra_args+=(--models "$MODELS")
[ -n "${MAX_SECONDS:-}" ] && extra_args+=(--max_seconds "$MAX_SECONDS")

# the members of supervision.assets.VideoAssets
for asset in VEHICLES VEHICLES_2 MILK_BOTTLING_PLANT GROCERY_STORE SUBWAY MARKET_SQUARE PEOPLE_WALKING BEACH BASKETBALL SKIING; do
    python depth_showcase.py --source_video_path "$asset" --device "$DEVICE" \
        --target_directory outputs ${extra_args[@]+"${extra_args[@]}"} \
        || echo "skipping $asset: failed"
done
