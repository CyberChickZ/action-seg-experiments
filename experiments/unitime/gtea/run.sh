#!/bin/bash
# UniTime + GTEA: top-level entry point.
#
# Usage:
#   bash run.sh setup       # one-time: create symlinks to Tieqiao's data
#   bash run.sh smoke       # quick: 5 training steps to verify pipeline
#   bash run.sh train       # full: 20-epoch training on split1
#   bash run.sh eval        # inference + dump predictions
#   bash run.sh status      # show current state of the dir + checkpoints

set -euo pipefail

GTEA_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$GTEA_DIR"

# Tieqiao's existing artifacts on HPC (read-only via group ACL)
TIEQIAO_BASE=/nfs/hpc/dgx2-4/tmp/2026/4/6
TIEQIAO_QWEN=/nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct
TIEQIAO_VIDEOS=/nfs/hpc/dgx2-4/data/TAS_videos/gtea

cmd=${1:-help}

case "$cmd" in
    setup)
        echo "[setup] creating symlinks to Tieqiao's data + features + base model"
        mkdir -p data video feature
        ln -sfn $TIEQIAO_BASE/data/gtea               data/gtea
        ln -sfn $TIEQIAO_VIDEOS                       video/gtea
        ln -sfn $TIEQIAO_BASE/feature/Qwen2-VL-7B-Instruct  feature/Qwen2-VL-7B-Instruct
        ln -sfn $TIEQIAO_QWEN                         Qwen2-VL-7B-Instruct
        echo "[setup] done. ls -la:"
        ls -la
        ;;
    smoke)
        bash scripts/train.sh gtea_smoke 1 --max_steps 5 --save_strategy no --report_to none
        ;;
    train)
        bash scripts/train.sh "${2:-gtea_split1_v1}" "${3:-20}"
        ;;
    eval)
        bash scripts/eval.sh "${2:-gtea_split1_v1}"
        ;;
    extract)
        bash scripts/extract_features.sh
        ;;
    status)
        echo "=== gtea/ contents ==="
        ls -la
        echo
        echo "=== checkpoints ==="
        ls -la checkpoints/ 2>/dev/null || echo "(none yet)"
        echo
        echo "=== results ==="
        ls -la results/ 2>/dev/null || echo "(none yet)"
        ;;
    help|*)
        sed -n '3,11p' "$0"
        exit 0
        ;;
esac
