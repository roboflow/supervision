#!/usr/bin/env bash
# Generate one depth video per model for every sample clip into outputs/.
# Usage: ./run_all.sh            (device is picked automatically)
#        DEVICE=cuda ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${DEVICE:-}" ]; then
    if [ "$(uname)" = "Darwin" ]; then DEVICE=mps
    elif command -v nvidia-smi > /dev/null 2>&1; then DEVICE=cuda
    else DEVICE=cpu
    fi
fi

mkdir -p data
(
    cd data
    python -c "
from supervision.assets import VideoAssets, download_assets
for asset in (VideoAssets.VEHICLES, VideoAssets.PEOPLE_WALKING):
    download_assets(asset)
"
)

for video in data/vehicles.mp4 data/people-walking.mp4; do
    python depth_showcase.py --source_video_path "$video" --device "$DEVICE"
done
