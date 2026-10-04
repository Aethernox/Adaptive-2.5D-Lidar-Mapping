#!/usr/bin/env bash
set -e

# Default parameters
MODE=${1:-"ground_truth"} # 'ground_truth', 'ai', or 'benchmark'
SEQUENCE=${2:-"00"}
START_FRAME=${3:-0}
END_FRAME=${4:-200}
FPS=${5:-30}

echo "================================================================="
echo " Adaptive Variable-Resolution 2.5D LiDAR Perception & Mapping"
echo " Mode: $MODE | Sequence: $SEQUENCE | Frames: $START_FRAME - $END_FRAME | Target FPS: $FPS"
echo "================================================================="

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -m tools.replay_kitti \
    --mode "$MODE" \
    --sequence "$SEQUENCE" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME" \
    --fps "$FPS" \
    --dashboard
