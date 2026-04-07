# Research Journal — action-seg-experiments

> Harry 跟 Claude 协作时学到的 insights / 决策 / gotchas. 按时间顺序追加.
> Claude 学到任何新事实都应**立即** append, 不要等用户说"记一下".

## 2026-04-07 — Repo bootstrap
- 创建 `~/git/action-seg-experiments` 帮 Tieqiao Wang 学长跑 video understanding / TAS 实验
- 第一个 experiment: UniTime ([`paper_notes/01_unitime.md`](../paper_notes/01_unitime.md))
- 4 层记忆系统 from `claude-memory-compiler` fork (跟 sam-3d-body 同款)
- Repo 是多 experiment 容器, 每个 experiment 一个 `experiments/<name>/` 子目录

## 2026-04-07 — UniTime upstream + checkpoint 调研
- 已 fork upstream `Lzq5/UniTime` → `CyberChickZ/UniTime`, 加为 submodule `experiments/unitime/UniTime/` (SSH remote)
- Repo push 到 `CyberChickZ/action-seg-experiments` (public)
- **关键事实 (HF API 验证)**: `zeqianli/UniTime` 只是 **LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`, ~150 MB), 不是 full checkpoint. 必须配 base model `Qwen/Qwen2-VL-7B-Instruct` (~16 GB) 才能跑.
- 配套 `zeqianli/UniTime-Data` (HF dataset, ~900 MB) 只有 anet/charades/ego4d/qvhl/tacos/pretrain 的 annotation JSON, **不含 video**. Video 要从原始 dataset 各自下载 (Ego4D 还要 license).
- Upstream 锁定环境: Python 3.10, torch 2.1.2 + cu121, transformers 4.49.0, peft 0.14.0, deepspeed 0.16.4, flash-attn 2.7.2.post1. 不要乱升级.
- 训练默认 8×GPU + deepspeed zero2 + bf16 + LoRA r=8, lr 2e-4, model_max_length 32768. 7B 在 8×80GB 跑 zero2 够用.
- Inference 单卡可跑, 上游自带 sample `data/test.json` (单 video, 单 query), 适合 smoke test.
- HPC 跑通的全流程写在 [`experiments/unitime/HPC_RUNBOOK.md`](../experiments/unitime/HPC_RUNBOOK.md), 包括 env / HF download / sample inference / train / SLURM 模板 / 6 个常见坑.
