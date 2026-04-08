# experiments/unitime

复现 UniTime ([Lzq5/UniTime](https://github.com/Lzq5/UniTime), NeurIPS 2025) 跑学长 Tieqiao Wang 准备的 GTEA dataset.

- **Paper**: [`../../paper_notes/01_unitime.md`](../../paper_notes/01_unitime.md) — arxiv [2506.18883](https://arxiv.org/abs/2506.18883)
- **Upstream code**: [Lzq5/UniTime](https://github.com/Lzq5/UniTime)
- **Fork (submodule)**: [`UniTime/`](./UniTime) → [CyberChickZ/UniTime](https://github.com/CyberChickZ/UniTime), 是 vanilla upstream, 不动
- **Pretrained**: [zeqianli/UniTime](https://huggingface.co/zeqianli/UniTime) (LoRA adapter) + [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) base
- **Owner (research)**: Tieqiao Wang
- **Runner (experiments)**: Harry

## Layout

```
experiments/unitime/
├── README.md              # this file
├── HPC_RUNBOOK.md         # general HPC env setup (conda, flash-attn, ...)
├── UniTime/               # submodule, vanilla upstream Lzq5/UniTime — DO NOT modify
├── scripts/               # 我们的运行脚本 (sync 自学长 TaCoS 版本, paths 改 GTEA)
│   ├── train.sh
│   ├── eval.sh
│   ├── feature.sh         # 一般不用 (features 已预提取)
│   └── inference_smoke.sh # 按上游 README Quick Start 验证 Qwen base + LoRA 加载
├── data/
│   └── gtea/
│       ├── gtea_csv_to_json.py # ⭐ 学长原版 (4/6/gtea_gtea_csv_to_json.py) 的 sync, MS-TCN GT csv → mr_seg JSON
│       └── annot/
│           ├── train.json # 21 train videos, 162 entries (committed, ~74KB)
│           └── test.json  #  7 test videos,  54 entries (committed, ~24KB)
├── feature/               # gitignored, HPC 上 symlink 到学长 features
└── video/                 # gitignored, HPC 上 symlink 到 raw videos
```

## What's already prepared (by Tieqiao on HPC)

| Item | HPC path | Note |
|---|---|---|
| Qwen2-VL-7B-Instruct base | `/nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct` | ~16 GB |
| zeqianli/UniTime LoRA adapter | `/nfs/stak/users/wangtie/2026/3/15/UniTime/UniTime` | ~150 MB |
| GTEA videos (28 .mp4) | `/nfs/hpc/dgx2-4/tmp/2026/4/6/video/gtea/` | symlink → `/nfs/hpc/dgx2-4/data/TAS_videos/gtea/` |
| GTEA pre-extracted features | `/nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen2-VL-7B-Instruct/gtea/` | 28 × .pt, ~bf16 [256,8,8,3584] each |
| GTEA mr_seg annotations | `/nfs/hpc/dgx2-4/tmp/2026/4/6/data/gtea/annot/{train,test}.json` | also committed in this repo at `data/gtea/annot/` |

GTEA 11 action classes: `background, close, fold, open, pour, put, scoop, shake, spread, stir, take`.
Split1: 21 train videos (S2-S4), 7 test videos (S1).

## Annotation format

Tieqiao 用 [`data/gtea/gtea_csv_to_json.py`](./data/gtea/gtea_csv_to_json.py) (sync 自他 4/6/ dir 的 `gtea_gtea_csv_to_json.py`) 把 MS-TCN GT csv 转成 UniTime `mr_seg` JSON. 每个 (video, action_class) 一条 entry, `window` 是该 action 在 video 内出现的所有区间:

```json
{
  "qid": 0,
  "id": "S2_Cheese_C1",
  "annos": [{"query": "take", "window": [[0.667, 4.267], [4.733, 6.933], ...]}],
  "duration": 42.2,
  "mode": "mr_seg"
}
```

UniTime upstream 原生支持 `mr_seg` multi-window — 看 `UniTime/collators/qwen2_vl.py:96-102`. **不需要改 model 代码**.

## 在 HPC 上运行 (Harry 的流程)

### 0. 一次性 setup

```bash
ssh osu-hpc && ssh dgxh-1
cd /nfs/hpc/share/zhanhaoc/action-seg-experiments
git pull
git submodule update --init --recursive

cd experiments/unitime
# 创建两个 symlink (gitignored)
ln -sfn /nfs/hpc/dgx2-4/tmp/2026/4/6/feature feature
ln -sfn /nfs/hpc/dgx2-4/data/TAS_videos      video

# 验证 conda env 没问题
conda activate UniTime
python -c "import torch, deepspeed, flash_attn; print('OK')"
```

env 不在的话先看 [`HPC_RUNBOOK.md`](./HPC_RUNBOOK.md), 注意 flash-attn 必须用 prebuilt wheel (源码编译会被 dgxh-1 默认 ninja `-j 112` OOM 干掉).

### 1. Smoke test (按上游 README Quick Start)

```bash
cd UniTime
bash ../scripts/inference_smoke.sh
```

跑通 → `results/smoke/results.json` 里 prediction 在 `[24.x, 30.x]` 附近. 这一步**只**验证 Qwen base + zeqianli/UniTime LoRA adapter 装载正确, 跟 GTEA 没关系.

### 2. Training (GTEA) — **必须用 srun!**

⚠️ **绝对不能直接 `bash ../scripts/train.sh`**. 你 ssh 直连 dgxh-1 后默认拿到的 device 0 是个 **MIG 4g.40gb 切片** (40 GB), 还跟别人共享 (一般已经被吃了 36 GB), 直接跑必 OOM:
```
GPU 0 has a total capacity of 39.50 GiB; Process X has 36.62 GiB memory in use
```

正确做法 — 用 SLURM srun 申请独占 GPU:

```bash
# 还在 UniTime/ cwd
srun -p dgxh --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=4:00:00 \
    bash ../scripts/train.sh
```

或者先 `srun --pty bash` 拿一个交互 shell, 在 shell 里手动 `bash ../scripts/train.sh`.

输出: `checkpoints/run1/`, tensorboard log 同目录.

### 3. Evaluation — 同样用 srun

```bash
srun -p dgxh --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=2:00:00 \
    bash ../scripts/eval.sh
```

输出: `results/run1/results.json`.

### 4. Metrics

```bash
python eval_metrics.py --res ./results/run1/results.json
```

⚠️ `eval_metrics.py` 算的是 VTG 的 R@k IoU, 不是 TAS 的 frame F1@10/25/50. 真要 benchmark 还得自己写 TAS metrics — 后续再说.

## ⚠️ Open issues (先不动, 跑通再回头看)

1. **GPU allocation**: dgxh-1 默认 device 0 是个 MIG 4g.40gb 切片 (40GB), 还跟别人共享, 直接跑必 OOM. 解决: SLURM `srun -p dgxh --gres=gpu:1 --mem=80G --time=4:00:00 bash ../scripts/train.sh` 申请独占 H100. (脚本里没加 srun, 因为学长版本没有, 跟原版一致.)
2. **Annotation 正确性**: 学长说 GTEA 转出来的格式可能不对, 跑通基线后再 debug. 我们 commit 了 annotation 到 `data/gtea/annot/` 方便修改.
3. **fps=2 vs short actions**: GTEA 一些 action 短到 0.07 秒, fps=2 会丢. 后面要 ablation fps=4/8.
4. **eval_metrics.py 跟 TAS metric 不对齐**: TAS 想要 F1@10/25/50 + Edit, 不是 R@k IoU.
5. **Multi-GPU 分布式**: paper 8 GPU, 我们暂时单 GPU. 收敛慢但够 debug.

## 给学长开权限

```bash
setfacl -R -m u:wangtie:rwx /nfs/hpc/share/zhanhaoc/action-seg-experiments
setfacl -R -d -m u:wangtie:rwx /nfs/hpc/share/zhanhaoc/action-seg-experiments
```

(default ACL 让新文件自动继承, 已经设置过.)
