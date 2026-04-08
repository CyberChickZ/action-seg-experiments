#!/bin/bash
# UniTime + GTEA: feature extraction launcher
#
# Only needed if you want to RE-extract features. Tieqiao already has 28
# pre-extracted .pt files at /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/, which we
# symlink in via the README setup, so normally you don't need to run this.
#
# Usage:
#   bash scripts/extract_features.sh

set -euo pipefail

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate UniTime
export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480

GTEA_DIR=$(cd "$(dirname "$0")/.." && pwd)
UNITIME_DIR=$(cd "$GTEA_DIR/../UniTime" && pwd)

MODEL_LOCAL_PATH=$GTEA_DIR/Qwen2-VL-7B-Instruct
VIDEO_ROOT=$GTEA_DIR/video/gtea
FEAT_ROOT=$GTEA_DIR/feature/Qwen2-VL-7B-Instruct

mkdir -p "$FEAT_ROOT"

# extract_qwen_features.py imports models.* and collators.* from UniTime, so
# run it with UniTime as cwd / on PYTHONPATH.
cd "$UNITIME_DIR"

SRUN_PREFIX=""
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SRUN_PREFIX="srun -p dgxh --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2:00:00"
fi

$SRUN_PREFIX python "$GTEA_DIR/scripts/extract_qwen_features.py" \
    --video_root "$VIDEO_ROOT" \
    --feat_root  "$FEAT_ROOT" \
    --model_local_path "$MODEL_LOCAL_PATH" \
    --dataset_name gtea \
    --gpu 0
