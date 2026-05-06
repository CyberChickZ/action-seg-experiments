#!/bin/bash
# TAS 在线训练: Gemma4-E4B-it + GTEA + 30s 滑窗
# 在线 vision encoder (不预提取) + vision LoRA + LLM LoRA
# 需要 H200 141GB (dgxh-4)
#
# 运行 (noVNC):
#     conda activate UniTime-gemma4
#     cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/UniTime
#     bash ../scripts/train_tas_gemma4_online.sh

set -e

export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480
export HF_HOME=/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache

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
TRAIN_SPLIT=/nfs/hpc/dgx2-4/data/gtea/splits/train.split1.bundle
TEST_SPLIT=/nfs/hpc/dgx2-4/data/gtea/splits/test.split1.bundle

RUN_ID=tas_gemma4_online_run1
LOG_FILE=./checkpoints/${RUN_ID}/train.log

mkdir -p ./checkpoints/${RUN_ID}

echo "=== TAS Online Training Start: $(date) ===" | tee $LOG_FILE
echo "RUN_ID: $RUN_ID" | tee -a $LOG_FILE
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" | tee -a $LOG_FILE
START_TIME=$(date +%s)

torchrun $DISTRIBUTED_ARGS train_tas.py \
    --model_id gemma4-e4b-it \
    --model_local_path $MODEL_LOCAL \
    --video_folder $VIDEO_FOLDER \
    --gt_folder $GT_FOLDER \
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
    --model_max_length 24576 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --train_vision_encoder True \
    --use_vision_lora True \
    --train_vision_projector False \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 8 \
    --save_strategy epoch 2>&1 | tee -a $LOG_FILE

END_TIME=$(date +%s)
echo "=== Done: $(date) ===" | tee -a $LOG_FILE
echo "Total wall time: $((END_TIME - START_TIME))s" | tee -a $LOG_FILE
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a $LOG_FILE
