#!/bin/bash
# UniTime + Qwen2-VL-2B training, GTEA dataset.
#
# Uses the ORIGINAL UniTime pipeline (Qwen2VL family) — zero code changes needed.
# Only swaps model_id from 7B to 2B and points to 2B model + features.
#
# Prerequisites:
#   1. Download model:
#        export HF_HOME=/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache
#        huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
#            --local-dir /nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct
#
#   2. Extract features (single GPU, ~5 min for 28 GTEA videos):
#        conda activate UniTime
#        cd experiments/unitime/UniTime
#        python feature_offline.py \
#            --data_path /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/train.json \
#            --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct \
#            --feat_root /nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct/features \
#            --video_root /nfs/hpc/dgx2-4/data/TAS_videos/gtea \
#            --num_parts 1 --part 0 --gpu 0
#        python feature_offline.py \
#            --data_path /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/test.json \
#            --model_local_path /nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct \
#            --feat_root /nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct/features \
#            --video_root /nfs/hpc/dgx2-4/data/TAS_videos/gtea \
#            --num_parts 1 --part 0 --gpu 0
#
# How to run:
#     conda activate UniTime
#     cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/UniTime
#     bash ../scripts/train_qwen2vl_2b.sh

export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480

NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_ID=qwen2-vl-2b-instruct
model_local_path=/nfs/hpc/share/zhanhaoc/MODLE/Qwen2-VL-2B-Instruct
TRAIN_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/train.json
EVAL_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/test.json
IMAGE_FOLDER=None
VIDEO_FOLDER=/nfs/hpc/dgx2-4/data/TAS_videos/gtea
FEAT_FOLDER=/nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen2-VL-2B-Instruct/gtea

FPS=2
CLIP_LENGTH=-1

TRAIN_VISION_ENCODER=False
USE_VISION_LORA=False
TRAIN_VISION_PROJECTOR=False

USE_LORA=True
Q_LORA=False
LORA_R=8
LORA_ALPHA=8

RUN_ID=qwen2vl_2b_gtea_run1

DS_STAGE=zero2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=1
NUM_EPOCHS=2

LR=2e-4
MODEL_MAX_LEN=24576

torchrun $DISTRIBUTED_ARGS train.py \
    --model_id $MODEL_ID \
    --model_local_path $model_local_path \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --fps $FPS \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --save_strategy "epoch" \
    --clip_length $CLIP_LENGTH \
    --feat_folder $FEAT_FOLDER
