#!/bin/bash
# UniTime + GTEA: training launcher (single H100 via SLURM srun)
#
# Usage:
#   bash scripts/train.sh                       # default RUN_ID + EPOCHS
#   bash scripts/train.sh my_run 20             # custom name + epochs
#   bash scripts/train.sh smoke 1 --max_steps 5 # smoke test
#
# This script must be run from experiments/unitime/gtea/ on the HPC.
# It chdir's into the sibling UniTime/ submodule (since UniTime expects cwd
# to be its own root for relative imports + ds_configs/).
#
# Symlink layout this script expects:
#   gtea/data/gtea/annot/train.json        -> tieqiao annot
#   gtea/data/gtea/annot/test.json         -> tieqiao annot
#   gtea/feature/Qwen2-VL-7B-Instruct/gtea -> tieqiao features
#   gtea/Qwen2-VL-7B-Instruct              -> shared base model
#   gtea/video/gtea                        -> shared video dir
#
# (See README for the symlink commands.)

set -euo pipefail

# ---------- env ----------
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate UniTime
export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480
# Important: dgxh-1 has a MIG slice on GPU 1 that confuses default torch.cuda
# enumeration; let SLURM allocate a real GPU for us instead.

# ---------- args ----------
RUN_ID=${1:-gtea_split1_v1}
NUM_EPOCHS=${2:-20}
shift 2 || true
EXTRA_ARGS="$@"     # forwarded to train.py (e.g. --max_steps 5)

# ---------- paths (resolved relative to gtea/) ----------
GTEA_DIR=$(cd "$(dirname "$0")/.." && pwd)
UNITIME_DIR=$(cd "$GTEA_DIR/../UniTime" && pwd)

MODEL_LOCAL_PATH=$GTEA_DIR/Qwen2-VL-7B-Instruct
TRAIN_DATA_PATH=$GTEA_DIR/data/gtea/annot/train.json
EVAL_DATA_PATH=$GTEA_DIR/data/gtea/annot/test.json
VIDEO_FOLDER=$GTEA_DIR/video/gtea
FEAT_FOLDER=$GTEA_DIR/feature/Qwen2-VL-7B-Instruct/gtea
OUTPUT_DIR=$GTEA_DIR/checkpoints/$RUN_ID

# Sanity-check the paths exist before allocating a GPU
for p in "$MODEL_LOCAL_PATH" "$TRAIN_DATA_PATH" "$EVAL_DATA_PATH" "$VIDEO_FOLDER" "$FEAT_FOLDER"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: missing path: $p"
        echo "       fix the symlinks in $GTEA_DIR (see README)"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# ---------- hyperparams (mostly from upstream paper) ----------
NUM_GPUS=1
MODEL_ID=qwen2-vl-7b-instruct
FPS=2
CLIP_LENGTH=32

USE_LORA=True
Q_LORA=False
LORA_R=8
LORA_ALPHA=8

DS_STAGE=zero2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=1
LR=2e-4
MODEL_MAX_LEN=32768

DISTRIBUTED_ARGS="--nnodes=1 --nproc_per_node ${NUM_GPUS} --rdzv_backend c10d --rdzv_endpoint localhost:0"

# ---------- launch ----------
# UniTime's train.py uses relative paths for ds_configs/, so chdir there.
cd "$UNITIME_DIR"

# Use srun to grab an exclusive GPU (avoids MIG contention on dgxh-1).
# If we're already inside a SLURM allocation, srun reuses it; otherwise it
# requests a new one.
SRUN_PREFIX=""
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SRUN_PREFIX="srun -p dgxh --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=4:00:00"
fi

$SRUN_PREFIX torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --model_local_path "$MODEL_LOCAL_PATH" \
    --data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --image_folder None \
    --video_folder "$VIDEO_FOLDER" \
    --feat_folder "$FEAT_FOLDER" \
    --fps $FPS \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    --run_name "$RUN_ID" \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --train_vision_encoder False \
    --use_vision_lora False \
    --train_vision_projector False \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --save_strategy epoch \
    --clip_length $CLIP_LENGTH \
    $EXTRA_ARGS
