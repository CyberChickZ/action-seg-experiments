#!/bin/bash
# UniTime feature extraction.
#
# NOTE: For GTEA, Tieqiao已经预提取好了 28 个 video features 在
#   /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen2-VL-7B-Instruct/gtea/
# 直接通过 train.sh / eval.sh 里的 FEAT_FOLDER 引用, **不需要再跑 feature.sh**.
#
# 这个 script 仅在重新提取 / 换 base model / 加新 video 时才用到.
#
# 用法跟上游 README 一致, 路径需要 [ToModify] 改成实际位置.

export CUDA_HOME=/usr/local/apps/cuda/12.1
export DECORD_EOF_RETRY_MAX=20480

data_paths=(
  "/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/train.json"
  "/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/data/gtea/annot/test.json"
)

# Tieqiao's setup uses GPUs 4-7 in parallel for 4-way data sharding;
# on a node where you only have 1 GPU, change to gpu_list=(0); part_list=(0); --num_parts 1
gpu_list=(4 5 6 7)
part_list=(0 1 2 3)
model_local_path=/nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct
feat_root=/nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/feature
video_root=/nfs/hpc/dgx2-4/tmp/2026/4/6/video/gtea

for data_path in "${data_paths[@]}"; do
  for i in ${!gpu_list[@]}; do
    python feature_offline.py --data_path $data_path --part ${part_list[$i]} --gpu ${gpu_list[$i]} --num_parts 4 --model_local_path $model_local_path --feat_root $feat_root --video_root $video_root &
  done
  wait
done
