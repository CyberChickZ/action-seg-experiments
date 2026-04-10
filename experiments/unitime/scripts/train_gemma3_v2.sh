#!/bin/bash
# UniTime + Gemma3-4B training, GTEA v2 dataset.
#
# v2 improvements over v1:
#   1. Descriptive action labels ("The person takes an object..." instead of "take")
#   2. Binary 0/1 frame output ("01101000..." instead of "1.0 seconds, 2.0 seconds, ...")
#
# Same features/model as train_gemma3.sh, only annotations + target format changed.
#
# How to run:
#     conda activate UniTime
#     cd experiments/unitime/UniTime
#     bash ../scripts/train_gemma3_v2.sh

export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480
export UNITIME_TARGET_MODE=binary   # ← key change: binary per-frame 0/1 output

NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_ID=gemma3-4b-it
model_local_path=/nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it
# v2 annotations with descriptive labels
TRAIN_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/train_v2.json
EVAL_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/test_v2.json
IMAGE_FOLDER=None
VIDEO_FOLDER=/nfs/hpc/dgx2-4/data/TAS_videos/gtea
FEAT_FOLDER=/nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it/features/gtea

FPS=2
CLIP_LENGTH=32

TRAIN_VISION_ENCODER=False
USE_VISION_LORA=False
TRAIN_VISION_PROJECTOR=False

USE_LORA=True
Q_LORA=False
LORA_R=8
LORA_ALPHA=8

RUN_ID=gemma3_gtea_v2_binary

DS_STAGE=zero2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=1
NUM_EPOCHS=2

LR=2e-4
MODEL_MAX_LEN=16384

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
