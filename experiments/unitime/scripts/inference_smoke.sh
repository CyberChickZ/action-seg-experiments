#!/bin/bash
# Smoke test: run upstream UniTime inference on its bundled sample (data/test.json,
# 1 video, 1 query). 用来验证 Qwen2-VL-7B base + zeqianli/UniTime LoRA adapter 加载正确.
# 跟 GTEA 没关系, 跟 train.sh 也没关系 — 这一步就是按上游 README 的 Quick Start 跑一次.
#
# 用法 (HPC dgxh-1):
#     conda activate UniTime
#     cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/unitime/UniTime
#     bash ../scripts/inference_smoke.sh
#
# 期望: results/smoke/results.json 里 prediction 在 [24.x, 30.x] 附近 (GT 是 24.3-30.4 "person turn a light on").

export CUDA_HOME=/usr/local/apps/cuda/12.1
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

# 学长已下载的:
QWEN_BASE=/nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct
# zeqianli/UniTime LoRA adapter (~150 MB), 学长已下到这里:
UNITIME_LORA=/nfs/stak/users/wangtie/2026/3/15/UniTime/UniTime

python inference.py \
    --model_local_path  $QWEN_BASE \
    --model_finetune_path $UNITIME_LORA \
    --data_path data/test.json \
    --output_dir ./results/smoke \
    --nf_short 128
