# 01 — UniTime: Universal Video Temporal Grounding with Generative MLLMs

> **Status**: 待读 — Harry 读完后在本文件下方追加 annotation  
> **Added**: 2026-04-07  
> **Experiment**: [`experiments/unitime/`](../experiments/unitime/)

## Metadata
- **arxiv**: https://arxiv.org/abs/2506.18883
- **Authors**: Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, Weidi Xie
- **Group**: Weidi Xie 组 (Shanghai Jiao Tong / similar)
- **Topic**: Video temporal grounding, MLLM-based moment retrieval
- **Relation to Tieqiao's TAS**: VTG = "找一段", TAS = "标每一帧". 共享视频时间建模, 但 task formulation 不同 — 是否能做 TAS 启发要看具体方法.

## Abstract (from arxiv)
UniTime — universal video temporal grounding model. 用 generative MLLM 把 timestamp tokens 跟 video tokens 交错 (interleave) 输出精确时刻. Adaptive frame scaling 处理变长视频. 在 5 个 VTG benchmark 上 zero-shot + fine-tuned 都 SOTA. 还能做 long-form VideoQA 的 preliminary moment retriever.

## Key claims (待 Harry 验证)
1. Interleaved timestamp ↔ video token 输出 — 比传统 regression / classification head 准
2. Adaptive frame scaling — 处理短到长的 video
3. 5 个 benchmark 都 SOTA, zero-shot + fine-tuned
4. 作为 VideoQA 的前置 moment retriever 也涨点

## Harry 阅读笔记 (待补)
> 读完后, 在这里追加: method 细节, 跟 TAS 的可移植性, 缺陷, 复现要点.

```
## YYYY-MM-DD — Section X
- ...
```

## 引用规则
之后任何对话引用这篇论文必须 quote 本文件具体行号. 例: `paper_notes/01_unitime.md:42`.
