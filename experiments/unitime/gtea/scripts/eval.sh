#!/bin/bash
# UniTime + GTEA: inference / eval launcher
#
# Usage:
#   bash scripts/eval.sh                  # eval default RUN_ID
#   bash scripts/eval.sh my_run           # eval a specific RUN_ID

set -euo pipefail

source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate UniTime
export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480

RUN_ID=${1:-gtea_split1_v1}

GTEA_DIR=$(cd "$(dirname "$0")/.." && pwd)
UNITIME_DIR=$(cd "$GTEA_DIR/../UniTime" && pwd)

MODEL_LOCAL_PATH=$GTEA_DIR/Qwen2-VL-7B-Instruct
EVAL_DATA_PATH=$GTEA_DIR/data/gtea/annot/test.json
VIDEO_FOLDER=$GTEA_DIR/video/gtea
FEAT_FOLDER=$GTEA_DIR/feature/Qwen2-VL-7B-Instruct/gtea
CKPT=$GTEA_DIR/checkpoints/$RUN_ID
RESULT_DIR=$GTEA_DIR/results/$RUN_ID

mkdir -p "$RESULT_DIR"
cd "$UNITIME_DIR"

SRUN_PREFIX=""
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SRUN_PREFIX="srun -p dgxh --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2:00:00"
fi

$SRUN_PREFIX python inference.py \
    --model_local_path "$MODEL_LOCAL_PATH" \
    --model_finetune_path "$CKPT" \
    --video_root "$VIDEO_FOLDER" \
    --feat_folder "$FEAT_FOLDER" \
    --data_path "$EVAL_DATA_PATH" \
    --output_dir "$RESULT_DIR" \
    --nf_short 128

# Optional: per-class metrics — eval_metrics.py was written for VTG R@k IoU,
# not TAS frame F1; we'll compute F1@10/25/50 separately. See README.
echo "results saved to $RESULT_DIR"
