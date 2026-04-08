#!/bin/bash
# UniTime evaluation / inference, GTEA dataset.
#
# Synced from Tieqiao Wang's TaCoS eval.sh, paths swapped to GTEA.
#
# How to run on HPC:
#     conda activate UniTime
#     cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/UniTime
#     bash ../scripts/eval.sh

export CUDA_HOME=/usr/local/apps/cuda/12.1
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

model_local_path=/nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct

TRAIN_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/train.json
EVAL_DATA_PATH=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/test.json
RUN_ID=run1

python inference.py --model_local_path $model_local_path \
    --model_finetune_path ./checkpoints/$RUN_ID \
    --video_root /nfs/hpc/dgx2-4/tmp/2026/4/6/video/gtea \
    --feat_folder /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen2-VL-7B-Instruct/gtea \
    --data_path $EVAL_DATA_PATH \
    --output_dir ./results/$RUN_ID \
    --nf_short 128
