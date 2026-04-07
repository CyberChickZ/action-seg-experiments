# Research Journal — action-seg-experiments

> Harry 跟 Claude 协作时学到的 insights / 决策 / gotchas. 按时间顺序追加.
> Claude 学到任何新事实都应**立即** append, 不要等用户说"记一下".

## 2026-04-07 — Repo bootstrap
- 创建 `~/git/action-seg-experiments` 帮 Tieqiao Wang 学长跑 video understanding / TAS 实验
- 第一个 experiment: UniTime ([`paper_notes/01_unitime.md`](../paper_notes/01_unitime.md))
- 4 层记忆系统 from `claude-memory-compiler` fork (跟 sam-3d-body 同款)
- Repo 是多 experiment 容器, 每个 experiment 一个 `experiments/<name>/` 子目录
