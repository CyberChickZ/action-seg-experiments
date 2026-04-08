# Research Journal — action-seg-experiments

> Harry 跟 Claude 协作时学到的 insights / 决策 / gotchas. 按时间顺序追加.
> Claude 学到任何新事实都应**立即** append, 不要等用户说"记一下".

## 2026-04-07 — Repo bootstrap
- 创建 `~/git/action-seg-experiments` 帮 Tieqiao Wang 学长跑 video understanding / TAS 实验
- 第一个 experiment: UniTime ([`paper_notes/01_unitime.md`](../paper_notes/01_unitime.md))
- 4 层记忆系统 from `claude-memory-compiler` fork (跟 sam-3d-body 同款)
- Repo 是多 experiment 容器, 每个 experiment 一个 `experiments/<name>/` 子目录

## 2026-04-07 — Tieqiao 学长 UniTime working dir 摸底
- **路径**: `/nfs/stak/users/wangtie/2026/3/15/UniTime` 在 dgxh-1 上 (NFS, group `upg14506` 共享读). 学长用 fork-less 工作流 (直接 git clone Lzq5/UniTime, HEAD `9c33ea0`, 跟我们 submodule 的 `d889a71` 差几个 commit, 学长的更早一些)
- **OSU HPC dgxh partition**: 4 节点 dgxh-[1-4], 每节点 8× **H100 80GB**, 单 job 最长 **2 天**, default 12h. ggd partition (dgxh-4 only) 有 7 天但限组. (`hpc` 文件是学长贴的 `scontrol show partition` 输出, 不是 SLURM 脚本.)
- **学长当前进度**: smoke test 阶段, **没复现任何 dataset**. 跑了 4 个 TaCoS feature .pt, 用 2 个 video 的 mini 集合 (`train_2.json` / `test_2.json` from `create_mini_tacos.py`). 1 GPU, NUM_EPOCHS=1.
- **数据格式坑** (你说的): `zeqianli/UniTime-Data` 的 tacos annotation `id` 字段是 `s17-d69-cam-002` 格式, 但实际 TaCoS 原始 video 是 `s17-d69.avi` (没有 cam 后缀, 而且是 `.avi` 不是 `.mp4`). 学长的处理:
  1. `trim_tacos_ids.py` 把 mini 文件里的 id trim 成 `s17-d69`
  2. `feature_offline.py` 里加 `vid = source["id"].split('-cam')[0]` 双保险
  3. **硬编码** `video_path_list.append(f'/nfs/hpc/dgx2-4/tmp/data/tacos_videos/{vid}.avi')` ← TaCoS-specific, 切别的 dataset 要改
- **学长 NOT 改 model 代码**: `models/qwen2_vl.py` 和 `collators/qwen_vision_process.py` 只有注释掉的 debug print, 实质 model logic = upstream. → **可以放心信任学长 dir 里训出来的东西**
- **学长已下好 (可复用)**: `Qwen2-VL-7B-Instruct/` (~16GB), `UniTime/` (LoRA adapter), `/nfs/hpc/dgx2-4/tmp/data/tacos_videos/*.avi`, `data/unitime_annotations/tacos/{train,test}.json` (zeqianli/UniTime-Data 的 tacos 部分)
- **学长试过 ego4d-nlq 但没成功**: 有 `verify_videos.py` 探查 ego4d clip_uid 命名 (UniTime 期望文件名是 `{clip_uid}.mp4`, 但 ego4d 原始下载是 `{video_uid}_{start}_{end}.mp4`), `data/ego4d_nlq_single/` 目录存在但 `featureroot/tacos` 只有 tacos features, 没 ego4d
- **训练时长 (Appendix E.1, Table 11, p.21)**: UniTime 默认 hybrid (resize short + compress long) 在 **Ego4D-NLQ 训 21 小时**. paper 还是没说几张 GPU 几张卡, 但配合 train.sh 默认 NUM_GPUS=8 + zero2 + bf16 + 1 epoch, 可推测是 8 GPU 21h. token compression for short = +5 days slower (p.21).

## 2026-04-07 — UniTime 论文精读
- 读完 main text (11 pages, [`paper_notes/01_unitime.md`](../paper_notes/01_unitime.md) 已替换 stub 为正式 annotation)
- **核心 trick**: timestamp-interleaved sequence —— 在每 frame 的 visual tokens 前插 `"timestamp: t seconds"` 文本 token, LLM 从 inserted timestamps 里 retrieve 一个输出 (`paper_notes/01_unitime.md:38`). 不是回归, 不是 special time token. **alignment-free + model-agnostic**.
- 三个 module **耦合**, ablation 拆开看 Adaptive Scaling 单独无用 (Table 7 row 2 vs 1, `paper_notes/01_unitime.md:148`). 必须 multi-stage + segment retrieval + adaptive scaling 三件套配齐, Ego4D-NLQ R1@0.3 才从 14.25 → 24.79.
- **Universal pre-training paradigm 真香**: zero-shot 提升幅度 (TaCoS R1@0.3 +25.47 over Mr.BLIP) > supervised SoTA 提升, 说明大规模混训比 dataset-specific 重要.
- 卖点不是绝对 SOTA, 是 **balance** (闭源 Seed1.5-VL Charades 强但 Ego4D 弱, Gemini Ego4D 强但 Charades 弱; UniTime 两边都 competitive, `paper_notes/01_unitime.md:130`).
- **5 个 benchmark**: Ego4D-NLQ, TaCoS, Charades-STA, ANet-Captions, QVHighlights (Table 2). ANet-Captions 是唯一 UniTime-Full 没明显涨的, paper appendix C.2 有 error analysis (没读).
- **跟 Tieqiao TAS 方向的 connection** (我的判断, `paper_notes/01_unitime.md:200`): 直接套 TAS 不行 — TAS dense per-frame label, UniTime sparse moment retrieval. 借鉴 timestamp-interleave + coarse-to-fine 的思路改成 dense decoder 可能有戏, 但风险高. **建议: 先复现 Charades-STA 验证 pipeline, 再讨论 TAS follow-up.**

## 2026-04-07 — UniTime upstream + checkpoint 调研
- 已 fork upstream `Lzq5/UniTime` → `CyberChickZ/UniTime`, 加为 submodule `experiments/unitime/UniTime/` (SSH remote)
- Repo push 到 `CyberChickZ/action-seg-experiments` (public)
- **关键事实 (HF API 验证)**: `zeqianli/UniTime` 只是 **LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`, ~150 MB), 不是 full checkpoint. 必须配 base model `Qwen/Qwen2-VL-7B-Instruct` (~16 GB) 才能跑.
- 配套 `zeqianli/UniTime-Data` (HF dataset, ~900 MB) 只有 anet/charades/ego4d/qvhl/tacos/pretrain 的 annotation JSON, **不含 video**. Video 要从原始 dataset 各自下载 (Ego4D 还要 license).
- Upstream 锁定环境: Python 3.10, torch 2.1.2 + cu121, transformers 4.49.0, peft 0.14.0, deepspeed 0.16.4, flash-attn 2.7.2.post1. 不要乱升级.
- 训练默认 8×GPU + deepspeed zero2 + bf16 + LoRA r=8, lr 2e-4, model_max_length 32768. 7B 在 8×80GB 跑 zero2 够用.
- Inference 单卡可跑, 上游自带 sample `data/test.json` (单 video, 单 query), 适合 smoke test.
- HPC 跑通的全流程写在 [`experiments/unitime/HPC_RUNBOOK.md`](../experiments/unitime/HPC_RUNBOOK.md), 包括 env / HF download / sample inference / train / SLURM 模板 / 6 个常见坑.
