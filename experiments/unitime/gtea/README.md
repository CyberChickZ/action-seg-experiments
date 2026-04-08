# UniTime → GTEA (TAS reproduction)

> 用 UniTime ([Lzq5/UniTime](https://github.com/Lzq5/UniTime), NeurIPS 2025) 的方法在 **GTEA** (4 subjects × 7 recipes = 28 videos, 11 action classes) 上跑 temporal action segmentation. **不是 GTEA dataset reproduction**, 是 **UniTime model reproduction with GTEA as application target**.

- **Owner (research)**: Tieqiao Wang
- **Runner (experiments)**: Harry
- **Upstream**: [`../UniTime/`](../UniTime) (submodule, fork of Lzq5/UniTime)
- **Paper notes**: [`../../../paper_notes/01_unitime.md`](../../../paper_notes/01_unitime.md)
- **Tieqiao 工作目录** (read-only ref): `/nfs/hpc/dgx2-4/tmp/2026/4/6/` (data prep + features) and `/nfs/stak/users/wangtie/2026/3/15/UniTime/` (his UniTime checkout)

## Layout

```
gtea/
├── README.md                       # this file
├── run.sh                          # entry point: bash run.sh {setup|smoke|train|eval}
├── scripts/
│   ├── train.sh                    # SLURM-aware training launcher
│   ├── eval.sh                     # inference launcher
│   ├── extract_features.sh         # (re-)extract Qwen features (usually skipped)
│   └── extract_qwen_features.py    # cleaned port of Tieqiao's extract_qwen_embeddings.py
├── data_prep/
│   └── gtea_csv_to_json.py         # MS-TCN GT csvs → UniTime mr_seg JSON
└── tieqiao_ref/                    # informational copies of Tieqiao's originals (DO NOT run)
    ├── extract_qwen_embeddings.py
    ├── extract_qwen_embeddings1.py
    └── F1.notes.md
```

After `bash run.sh setup` on HPC, the following symlinks appear (NOT in git):
```
data/gtea/                          → /nfs/hpc/dgx2-4/tmp/2026/4/6/data/gtea/
video/gtea                          → /nfs/hpc/dgx2-4/data/TAS_videos/gtea/
feature/Qwen2-VL-7B-Instruct/       → /nfs/hpc/dgx2-4/tmp/2026/4/6/feature/Qwen2-VL-7B-Instruct/
Qwen2-VL-7B-Instruct                → /nfs/stak/users/wangtie/2026/3/15/UniTime/Qwen2-VL-7B-Instruct
```

## What's already done by Tieqiao (we don't redo)

1. **Qwen2-VL-7B-Instruct downloaded** (~16 GB).
2. **GTEA videos** at `/nfs/hpc/dgx2-4/data/TAS_videos/gtea/` (28 .mp4 files).
3. **MS-TCN GTEA groundTruth** converted to UniTime mr_seg JSON via `gtea_csv_to_json.py`. Output:
   - `data/gtea/annot/train.json` — split1 train, 21 videos, **162 entries** (one per (video, action_class) pair, 11 actions vocab).
   - `data/gtea/annot/test.json` — split1 test, 7 videos, **54 entries**, qid 162-215.
   - Action vocab: `background, close, fold, open, pour, put, scoop, shake, spread, stir, take`.
4. **Qwen2-VL-7B features** pre-extracted for all 28 videos: `feature/Qwen2-VL-7B-Instruct/gtea/*.pt`. Each .pt has `feature: bf16[256,8,8,3584]`, `frame_idx: int64[256]`, `sample_fps: float`.

## Annotation format (mr_seg multi-window)

```json
{
  "qid": 0,
  "id": "S2_Cheese_C1",
  "annos": [{
    "query": "take",
    "window": [[0.667, 4.267], [4.733, 6.933], [13.133, 14.4], [24.067, 24.733], [32.133, 35.2]]
  }],
  "duration": 42.2,
  "mode": "mr_seg"
}
```

**`mr_seg` 模式的 multi-window list 是 UniTime upstream 原生支持的**, 不需要改 model / collator 代码. 见 `UniTime/collators/qwen2_vl.py:96-102`. 这个 mode 下训练 target 不是 `"From s seconds to e seconds"` 的 interval 串, 而是把 query 命中范围内**所有 sampled timestamps 列出来**:

```
"1.0 seconds, 2.0 seconds, 3.0 seconds, 5.0 seconds, 6.0 seconds, 13.0 seconds, ..."
```

效果上等价于 per-frame label, 完美对应 TAS 任务结构.

## ⚠️ 已知 caveats / TODO

1. **fps=2 vs short actions**: 一些 GTEA action 短到 0.07 秒 (e.g. `[7.0, 7.267]` 是 0.27 秒, `[22.0, 22.067]` 是 0.07 秒). UniTime 默认 fps=2, 一个 frame 间隔 0.5 秒, 这种短动作会被采样跳过或被 background 吞掉. 后面要 ablation fps=4 / fps=8.
2. **`background` class**: paper 没考虑 "no action" query, 学长这边把 background 当成一个普通 action class. 训出来怎么样要看.
3. **eval metric**: UniTime 的 `eval_metrics.py` 算的是 R@k IoU (VTG metric), 不是 TAS 的 F1@10/25/50 + Edit. 评测要写 TAS-style metrics, 把 model 输出的 timestamp list reconstruct 成 frame label 序列, 再跟 GT 算 F1.
4. **Multi-GPU**: 目前 single GPU. 8 卡分布式训练 (paper 默认) 需要解决 dgxh-1 GPU 1 上 MIG 切片让 torch.cuda multi-device init 失败的问题, 通过 SLURM 申请独占资源就行 (`--gres=gpu:8` + `srun`).
5. **dgxh-1 默认 GPU 0 = MIG 4g.40gb 切片** (40 GB), 还跟别的 user 共享. 直接 SSH 用基本必 OOM. **`run.sh` 里所有的 train/eval/extract 都通过 `srun -p dgxh --gres=gpu:1` 申请独占 GPU**, 不要直接在 ssh shell 里跑 python.

## Harry 的运行流程 (HPC, 第一次)

```bash
# 1. 登 HPC (你已经在 dgxh-1 了)
ssh osu-hpc
ssh dgxh-1

# 2. 进 repo
cd /nfs/hpc/share/zhanhaoc/action-seg-experiments
git pull                                  # pull 这次的 gtea/ 添加
git submodule update --init --recursive   # 确保 UniTime/ submodule 在

# 3. 进 gtea 实验目录
cd experiments/unitime/gtea

# 4. 一次性 setup: 创建 symlinks 到学长的数据
bash run.sh setup
# 应该看到 4 个 symlink: data/, video/, feature/, Qwen2-VL-7B-Instruct
ls -la

# 5. Smoke test (5 step, 验证 pipeline)
bash run.sh smoke
# 这个会自动 srun 申请一张 GPU, 跑 5 个 step, 退出.
# 看到 loss 数字, ckpt 不写盘, 通了说明 env / data / model 都对.

# 6. 完整 train (20 epoch, ~半小时-几小时, single GPU)
bash run.sh train gtea_split1_v1 20
# tensorboard log 在 checkpoints/gtea_split1_v1/, 可以另开 ssh 看

# 7. eval
bash run.sh eval gtea_split1_v1
# 结果在 results/gtea_split1_v1/results.json
```

如果 smoke 炸了, 把 stderr 整段贴出来, **不要自己 fix 然后再跑** (省得多次 srun 排队).

## 给学长开权限

```bash
setfacl -R -m u:wangtie:rwx /nfs/hpc/share/zhanhaoc/action-seg-experiments
```

学长就能读写整个 repo (跟他给我们 /nfs/stak/users/wangtie/2026/4/ 一样).

## Troubleshooting

| 现象 | 原因 / 解法 |
|---|---|
| `MissingCUDAException: CUDA_HOME does not exist` | env 没设, 已在 train.sh 顶部加了 `export CUDA_HOME=/usr/local/apps/cuda/12.1` |
| `torch.cuda.OutOfMemoryError ... 40 GiB` 出现 "MIG 4g.40gb" | 你直接 ssh 上 dgxh-1 后跑 python, 拿到的是 MIG 切片. 必须 `srun --gres=gpu:1`. `run.sh` 已经默认这样. |
| `flash_attn` import 报 404 / `nvcc` `code=255` 大量 FAILED | 不要 `pip install flash-attn==... --no-build-isolation` (dgxh-1 默认 ninja `-j 112` OOM). 用 prebuilt wheel: `wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && pip install <wheel>`. |
| `Permission denied` 读 `/nfs/stak/users/wangtie/...` | 学长还没给 ACL: 让他跑 `setfacl -R -m u:zhanhaoc:rwx /nfs/stak/users/wangtie/2026/4 /nfs/stak/users/wangtie/2026/3/15` |
