# UniTime — HPC runbook

目标: 在 HPC (DGX H100 / A100) 上 clone 本 repo, 拉起 UniTime 的 train + inference.

- Upstream: [Lzq5/UniTime](https://github.com/Lzq5/UniTime) (NeurIPS 2025)
- Fork (submodule): [CyberChickZ/UniTime](https://github.com/CyberChickZ/UniTime) → `experiments/unitime/UniTime/`
- Pretrained adapter: [zeqianli/UniTime](https://huggingface.co/zeqianli/UniTime) — **LoRA adapter only** (`adapter_config.json` + `adapter_model.safetensors`, ~150 MB)
- Base model: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (~16 GB)
- Annotations: [zeqianli/UniTime-Data](https://huggingface.co/datasets/zeqianli/UniTime-Data) (anet / charades / ego4d / qvhl / tacos / pretrain, ~900 MB; **不含 video**)

## 0. Clone (含 submodule)

```bash
git clone --recurse-submodules git@github.com:CyberChickZ/action-seg-experiments.git
cd action-seg-experiments/experiments/unitime/UniTime
# 已经 clone 但忘了 --recurse-submodules:
#   git submodule update --init --recursive
```

## 1. Conda env

upstream 要求 Python 3.10 + torch 2.1.2 + CUDA 12.1 + flash-attn 2.7.2.post1. 严格按它的固定版本.

```bash
conda create -n unitime python=3.10 -y
conda activate unitime

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
# requirements 锁定: accelerate 1.0.1, transformers 4.49.0, peft 0.14.0,
#                   decord 0.6.0, numpy 1.26.4, deepspeed 0.16.4,
#                   flash-attn 2.7.2.post1, tensorboard 2.18.0, nncore 0.4.5
```

flash-attn 装不上 → 先 `pip install packaging ninja`, 然后
`pip install flash-attn==2.7.2.post1 --no-build-isolation`. H100 / A100 都支持.

## 2. 下载 checkpoints + 数据

```bash
# huggingface-cli (推荐, resume + 多线程)
pip install -U "huggingface_hub[cli]"
export HF_HOME=$SCRATCH/hf_cache   # 不要塞家目录, HPC quota 会爆

# 2a. base model (~16 GB)
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct \
    --local-dir $SCRATCH/models/Qwen2-VL-7B-Instruct

# 2b. UniTime LoRA adapter (~150 MB)
huggingface-cli download zeqianli/UniTime \
    --local-dir $SCRATCH/models/UniTime

# 2c. 标注 (anet/charades/ego4d/qvhl/tacos/pretrain, ~900 MB)
huggingface-cli download zeqianli/UniTime-Data --repo-type dataset \
    --local-dir $SCRATCH/data/UniTime-Data
```

> **Gated model**: Qwen2-VL-7B-Instruct 不需要 access request, 直接下. 若以后换成 gated model, `export HF_TOKEN=hf_xxx` 后再下.

## 3. 视频源 (上游不提供)

`UniTime-Data` 只有 annotation JSON, **video 要自己从原始 dataset 拿**:

| Dataset | 来源 |
|---|---|
| Charades-STA | [Charades release](https://prior.allenai.org/projects/charades) |
| ActivityNet Captions | [ActivityNet](http://activity-net.org/download.html) |
| TACoS | [TACoS](https://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos) |
| QVHighlights | [QVHL repo](https://github.com/jayleicn/moment_detr) |
| Ego4D-NLQ | [Ego4D](https://ego4d-data.org/) (要 license) |

annotation 里的 `video_path` 字段需要指到 HPC 上 video 文件的路径 (一般是相对 `--video_folder` 根目录). Ego4D 的 loader 在 `datasets/data_ego4d.py:load_data_to_dict()`, 其它 dataset 套用即可.

跑通 pipeline 之前先用一个小 dataset (如 Charades-STA, ~5k clips, ~10 GB) 端到端验证, 不要一上来全下.

## 4. Inference (single GPU, 验证 checkpoint)

upstream `data/test.json` 自带一个 sample (`3MSZA.mp4`, person turn a light on). 用它最小化验证 LoRA + base 加载正确:

```bash
cd experiments/unitime/UniTime
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

python inference.py \
    --model_local_path  $SCRATCH/models/Qwen2-VL-7B-Instruct \
    --model_finetune_path $SCRATCH/models/UniTime \
    --data_path data/test.json \
    --output_dir ./results/smoke \
    --nf_short 128
```

成功 → `results/smoke/results.json` 里能看到 `[24.x, 30.x]` 附近的 prediction.

## 5. Training (multi-GPU, deepspeed)

`scripts/train.sh` 默认 `NUM_GPUS=8`, deepspeed zero2, bf16, flash-attn, LoRA r=8 α=8, lr 2e-4, model_max_length 32768, gradient_checkpointing.

每个 dataset 的 epoch 数:
- ego4d / tacos / pretrain: 1
- charades / anet / qvhl: 2

跑前要 `[ToModify]` 的字段 (脚本里全是 placeholder):

```bash
# scripts/train.sh
model_local_path=$SCRATCH/models/Qwen2-VL-7B-Instruct
TRAIN_DATA_PATH=$SCRATCH/data/UniTime-Data/charades/train.json
EVAL_DATA_PATH=$SCRATCH/data/UniTime-Data/charades/test.json
VIDEO_FOLDER=$SCRATCH/data/charades/videos
FEAT_FOLDER=$SCRATCH/data/charades/features    # 跑 feature.sh 之后才有
RUN_ID=charades_v0
NUM_EPOCHS=2
```

完整 pipeline (按顺序):

```bash
# (a) 离线抽特征 — feature.sh 默认用 GPU 4-7, 4-way data parallel
bash scripts/feature.sh

# (b) 训练
bash scripts/train.sh

# (c) eval
bash scripts/eval.sh

# (d) metrics
python eval_metrics.py --res ./results/charades_v0/results.json
```

## 6. SLURM (DGX H100)

DGX H100 节点 8×H100 80GB, 8 GPU 训练. 模板:

```bash
#!/bin/bash
#SBATCH --job-name=unitime-charades
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitime
cd $HOME/git/action-seg-experiments/experiments/unitime/UniTime
bash scripts/train.sh
```

## 7. 常见坑

1. **decord EOF**: video 截断导致 worker hang → `export DECORD_EOF_RETRY_MAX=20480` (上游脚本已加).
2. **MPS**: 严禁. 此 repo 只 cuda. (符合全局 CLAUDE.md 约束.)
3. **flash-attn 编译慢**: 第一次装 ~20 min, 加 `MAX_JOBS=8` 限制并行避免 OOM.
4. **LoRA adapter 加载**: `model_finetune_path` 必须指向**包含 `adapter_config.json` 的目录**, 不是单个文件. peft 0.14 对路径敏感.
5. **HF cache**: 默认在 `~/.cache/huggingface/`, HPC 家目录小 → 提前 `export HF_HOME=$SCRATCH/hf_cache`.
6. **deepspeed zero3 OOM**: 7B 在 8×80GB 上 zero2 够用, 不用切 zero3 (zero3 对小模型反而慢).

## 8. Status / TODO

- [ ] HPC env 装好, smoke inference 跑通 (sample test.json)
- [ ] 选第一个验证 dataset (建议 Charades-STA, 最小)
- [ ] feature 抽完
- [ ] reproduce charades 数字, 跟论文 Table 对比
- [ ] 决定 follow-up 方向 (跟 Tieqiao 的 TAS 怎么连)
