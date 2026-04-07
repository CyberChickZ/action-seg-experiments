# experiments/unitime

复现 / follow-up UniTime (Universal Video Temporal Grounding with Generative MLLMs).

- **Paper**: [`paper_notes/01_unitime.md`](../../paper_notes/01_unitime.md) — arxiv [2506.18883](https://arxiv.org/abs/2506.18883)
- **Upstream code**: [Lzq5/UniTime](https://github.com/Lzq5/UniTime) (NeurIPS 2025)
- **Fork (submodule)**: [`UniTime/`](./UniTime) → [CyberChickZ/UniTime](https://github.com/CyberChickZ/UniTime)
- **Pretrained**: [zeqianli/UniTime](https://huggingface.co/zeqianli/UniTime) (LoRA adapter on Qwen2-VL-7B-Instruct)
- **HPC setup**: see [`HPC_RUNBOOK.md`](./HPC_RUNBOOK.md)
- **Owner (research)**: Tieqiao Wang
- **Runner (experiments)**: Harry

## Layout

```
experiments/unitime/
├── README.md          # this file
├── HPC_RUNBOOK.md     # clone → env → ckpt → smoke test → train (HPC)
└── UniTime/           # git submodule, upstream code (forked to CyberChickZ)
```

`UniTime/` 是 submodule. clone 时记得 `git clone --recurse-submodules`,
或者已经 clone 之后 `git submodule update --init --recursive`.

## TODO
- [ ] 读论文, 在 `paper_notes/01_unitime.md` 写 annotation
- [ ] HPC env 装好, smoke inference 跑通 sample test.json
- [ ] 选第一个 reproduce dataset (建议 Charades-STA)
- [ ] 跑通 feature → train → eval pipeline
- [ ] 跟 paper Table 对数
- [ ] 决定 follow-up 方向 (跟 Tieqiao 的 TAS 怎么连)

## 跟 Tieqiao 的 TAS 方向的 connection
> Harry 决定后填.
