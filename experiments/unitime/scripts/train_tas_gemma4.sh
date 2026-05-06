#!/bin/bash
# TAS training: Gemma4-E4B-it + GTEA + 30s sliding window + 12x12 spatial pool
#
# Run on dgxh (OnDemand shell):
#     conda activate UniTime-gemma4
#     cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/UniTime
#     bash ../scripts/train_tas_gemma4.sh

export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480

NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_LOCAL=/nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it
VIDEO_FOLDER=/nfs/hpc/dgx2-4/data/TAS_videos/gtea
GT_FOLDER=/nfs/hpc/dgx2-4/data/gtea/groundTruth
FEAT_FOLDER=/nfs/hpc/dgx2-4/tmp/2026/5/5/feature/Gemma4-E4B-it-15fps/gtea
TRAIN_SPLIT=/nfs/hpc/dgx2-4/data/gtea/splits/train.split1.bundle
TEST_SPLIT=/nfs/hpc/dgx2-4/data/gtea/splits/test.split1.bundle

RUN_ID=tas_gemma4_12x12_run1

torchrun $DISTRIBUTED_ARGS train_tas.py \
    --model_id gemma4-e4b-it \
    --model_local_path $MODEL_LOCAL \
    --video_folder $VIDEO_FOLDER \
    --gt_folder $GT_FOLDER \
    --feat_folder $FEAT_FOLDER \
    --train_split $TRAIN_SPLIT \
    --test_split $TEST_SPLIT \
    --spatial_pool_h 12 \
    --spatial_pool_w 12 \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/zero2.json \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --train_vision_encoder False \
    --train_vision_projector False \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 8 \
    --save_strategy epoch
