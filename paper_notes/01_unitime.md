# 01 — UniTime: Universal Video Temporal Grounding with Generative MLLMs

> **Status**: 已读 (主文 11 pages, 不含 appendix)
> **Read date**: 2026-04-07
> **PDF**: [`pdfs/01_unitime.pdf`](pdfs/01_unitime.pdf) (gitignored)
> **Experiment**: [`experiments/unitime/`](../experiments/unitime/)

## Metadata
- **arxiv**: 2506.18883 (v2, 2025-11-21)
- **Venue**: NeurIPS 2025
- **Authors**: Zeqian Li (SJTU SAI), Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, **Weidi Xie** (SJTU)
- **Affil**: SJTU SAI + ByteDance Seed
- **Project**: https://lzq5.github.io/UniTime
- **Code**: https://github.com/Lzq5/UniTime (forked → `experiments/unitime/UniTime/`)
- **Ckpt**: HF `zeqianli/UniTime` (LoRA adapter only on Qwen2-VL-7B)

## TL;DR (one paragraph)
UniTime 把 Qwen2-VL-7B 改造成一个 universal Video Temporal Grounding (VTG) 模型. 三个核心招数:
(1) **Timestamp-interleaved sequence** — 在每个 video frame 的 visual tokens 前插入 `"timestamp: t seconds"` 文本 token, 让 LLM 用 *retrieval* 方式读出时间戳 (输出 `"From s seconds to e seconds"`), 而不是用回归头或 dense positional encoding;
(2) **Adaptive frame scaling** — 按 video 长度动态分配每帧 token budget, 短 video 高分辨率长 video 低分辨率, 超过阈值就切片 divide-and-conquer;
(3) **Multi-stage coarse-to-fine inference** — 长 video 先在 segment 级别 retrieve 候选窗口, 再在窗口内 refine 精确边界.
配套一个 **video-centric** training 范式 (一个 video 的多 query 合到一条序列, 改 attention mask 防交叉) 大幅省 I/O. 在 5 个 VTG benchmark (Ego4D-NLQ, TaCoS, Charades-STA, ANet-Captions, QVHL) 上 dataset-specific + zero-shot 双 SOTA, 用作 long-video VideoQA 的前置 retriever 在 4 个 benchmark 上都涨点.

## 1. Problem (Section 2.1)

`Y = Φ_UniTime(V, T, Q)`
- `V = {f_1, ..., f_Nf}`: video frames
- `T = {t_1, ..., t_Nf}`: timestamps of those frames
- `Q`: free-form query (descrip / question)
- `Y = {(s_k, e_k)}`: 一组 start/end timestamps (注意是 set, 支持多 moment)

## 2. Method (Section 2.2)

### 2.1 Adaptive Frame Scaling
两个阈值 `N_f^short = 128`, `N_f^long = 1024`. 给定 fps=2 采到的 N_f 帧:

| 条件 | 处理 |
|---|---|
| `N_f < N_f^short` (短) | resize frames 让每帧得到 `N_res = N_total/N_f` 个 patches (高空间分辨率) |
| `N_f^short ≤ N_f < N_f^long` (中) | 用 vision encoder 全分辨率, 然后 token compression (双线性插值) 压到 N_res tokens |
| `N_f > N_f^long` (长) | 切成多个 clips, 每段 N_f^long 帧, divide-and-conquer |

总 token budget 上限 16,384.

### 2.2 Timestamp-Interleaved Sequence (核心 trick)
对每帧 f_i, timestamp 编码成 `τ_i = "timestamp: t_i seconds"` 文本 (eq. 1):

```
S = [T_1; V_1; T_2; V_2; ...; T_Nf; V_Nf; Q]
```

LLM 输出 `"From s seconds to e seconds"`. 关键:
- 模型不是回归一个浮点数, 也不是 decode 特殊 time token, **是从已经 inserted 的 timestamp 文本里 retrieve 一个**. 即 prediction ⊆ T (the sampled timestamps set).
- "alignment-free": timestamp 是 textual, 跟 LLM language space 天然对齐, 不像 LITA 学新的 time embedding 还要对齐.
- "model-agnostic": 这一招对所有支持动态 frame 的 MLLM 都能 plug-and-play, Section 3.4 在 Qwen2-VL-2B/7B, Qwen2.5-VL-7B, InternVL2.5-2B/8B 上都验证 (Table 6).

### 2.3 Coarse-grained Variant
对长 video, 不是每帧前都插 timestamp, 而是把 video 切成 N_s 段, 每段 L_s 帧, 只在段头插一个 timestamp t_sj (eq. 2). 这样减少 token 但损失帧级精度 → 后面 multi-stage refine 补回来.

### 2.4 Multi-stage Inference
长 video 处理流程: 第一轮在每个 clip (N_f^long) 内做 segment-level retrieval → 候选 segment 聚合 → 在候选里再做 segment retrieval → 最后在选中的 segment 里跑 fine-grained grounding. 是 hierarchical 而不是 single-pass.

### 2.5 Training
- Loss: 标准 auto-regressive NLL, 只对 target tokens (i.e., `"From s seconds to e seconds"`) 算 (Section 2.3).
- **Video-centric training**: 一条训练样本是 *一个 video + 它的所有 (query, answer) pairs*, 不是 (query, video) 对. attention mask 改成阻止 query-answer pair 之间互相看, 但每个 pair 都从 video tokens 后的相同 starting position index 开始. 这样多 query 共享一次 video forward, 大幅省 I/O 和重复 vision encoding.
- **Replication factor `N_rep`**: 数据里短 video 多长 video 少, 长 video 复制 N_rep 次平衡分布. 实验 N_rep=4 (Section 3.5, Figure 3b — segment retrieval 随 N_rep 涨然后饱和, fine-grained 稳定).

## 3. Experimental Setup (Section 3.1)

- **Base**: Qwen2-VL-7B
- **Trainable**: vision encoder 冻结, LLM 用 LoRA r=8 α=8
- **fps**: 2
- **N_f^short / N_f^long**: 128 / 1024
- **Token cap**: 16,384
- **Segment length L_s**: 32 frames (Figure 3a 显示 32 是 sweet spot)
- **N_rep**: 4
- **Optim**: AdamW, lr=2e-4, batch_size=8, 1 epoch, 3% linear warmup
- **Training data** (Table 1, page 5):
  - Part I (universal pre-training only): NaQ (1031K queries), DiDeMo, QuerYD, HiRest, COIN, Momentor, YouCook2 — 总计 ~1.2M queries
  - Part II (also used as benchmark train sets): Ego4D-NLQ, TaCoS, Charades-STA, QVHL, ANet-Captions

## 4. Results

### 4.1 Dataset-specific & Universal (Table 3, page 6)
两种 setting:
- **UniTime-SP**: 每个 dataset 单独 fine-tune
- **UniTime-Full**: 一个模型在 Part I + Part II 全部数据上 train

| Benchmark | Metric | Prev SoTA | UniTime-SP | UniTime-Full | Δ vs SoTA |
|---|---|---|---|---|---|
| Ego4D-NLQ | R1@0.3 | 18.28 (RGNet) | 24.79 | **27.09** | **+8.81** |
| Ego4D-NLQ | R1@0.5 | 12.04 | 16.83 | **18.41** | **+6.39** |
| TaCoS | R1@0.3 | 57.61 (LD-DETR) | 61.18 | **66.91** | **+9.30** |
| TaCoS | R1@0.5 | 44.70 | 48.31 | **55.14** | **+10.44** |
| Charades-STA | R1@0.5 | 70.20 (SG-DETR) | 74.33 | **75.27** | **+5.07** |
| Charades-STA | R1@0.7 | 49.50 | 53.71 | **56.85** | **+7.35** |
| ANet-Cap | R1@0.5 | 53.92 (Mr.BLIP) | 36.62 | **53.67** | -0.25 (持平) |
| QVHighlights | R1@0.5 | 74.77 (UniVTG-PT) | 77.76 | **76.72** | **+1.95** |
| QVHighlights | R1@0.7 | 60.51 | 63.29 | **62.65** | **+2.14** |

> 🚩 注意: ANet-Captions 是唯一 UniTime-Full 没明显涨的 benchmark. 论文在 Appendix C.2 有 error analysis (我没读 appendix).

### 4.2 Zero-shot (Table 4, page 7)
**UniTime-Zero**: 只在 Part I 上 train, **完全不见 benchmark in-domain 数据**. 跟其它 zero-shot MLLM-based 方法 (UniVTG, Mr.BLIP, VTG-LLM, Momentor, VTimeLLM, TimeChat, TimeMarker, TimeSuite) 比:

- TaCoS R1@0.3: **50.06** (UniTime-Zero) vs 24.59 (Mr.BLIP) — **+25.47**
- Charades-STA R1@0.5: **59.09** vs 51.90 (TimeMarker) — **+7.19**
- QVHighlights R1@0.7: **31.48** vs 11.16 (Momentor) — **+20.32**

zero-shot 提升幅度 > supervised SoTA 提升, 说明 universal pre-training paradigm 真香.

### 4.3 vs 闭源模型 (Table 5, page 7)
sampled subset, 比闭源大模型. 结论: Seed1.5-VL Charades 最强 (72.2 mIoU), Gemini-2.5-Pro Ego4D 最强 (20.5 mIoU), 但**没有一个闭源模型在长短 video 都好**. UniTime-Full 在两边都 competitive (Ego4D mIoU 17.20, Charades mIoU 70.11) → 卖点是 **balance**, 不是 absolute SOTA.

### 4.4 Flexibility (Table 6, page 7)
插到不同 backbone:
| Backbone | Charades R1@0.5 baseline | + UniTime |
|---|---|---|
| Qwen2-VL-2B | 2.23 | 27.25 (+25.02) |
| Qwen2-VL-7B | 27.29 | 57.25 (+29.96) |
| Qwen2.5-VL-7B | 7.18 | 53.71 (+46.53) |
| InternVL2.5-2B | 6.70 | 65.43 (+58.73) |
| InternVL2.5-8B | 8.66 | 60.50 (+51.84) |

> 🔥 InternVL2.5-2B 提升最大. 说明 timestamp-interleave 这一招对小模型也有效, 不是只靠 backbone 体量.

## 5. Ablations (Section 3.5)

### 5.1 Module ablation on Ego4D-NLQ (Table 7, page 8)
| Adaptive Scaling | Multi-stage Inf | Segment Retrieval | R1@0.3 | R1@0.5 | mIoU |
|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 14.25 | 7.54 | 9.83 |
| ✓ | ✗ | ✗ | 14.00 | 7.51 | 9.83 |
| ✗ | ✓ | ✗ | 18.42 | 12.13 | 12.78 |
| ✗ | ✓ | ✓ | 17.91 | 11.99 | 12.39 |
| ✓ | ✓ | ✓ | **24.79** | **16.83** | **17.25** |

读法:
1. **Adaptive scaling 单独用没用** (row 2 vs row 1, 持平). 它**不是独立涨点**, 是为长 video 服务的 — 只有当 multi-stage + segment retrieval 把上下文压力转嫁过来时它才发挥作用 (row 5 vs row 4, +6.88 R1@0.3).
2. **Multi-stage inference 是单 module 最大涨点** (+4.17 R1@0.3, row 3 vs row 1).
3. **Segment retrieval 必须配合 adaptive scaling**, 单独加 segment retrieval 反而掉 (row 4 vs row 3, -0.51 R1@0.3) — 因为长 video 没 adaptive scaling 时 spatial info 太糊, fine-grained refine 不准.

→ **三个 module 是耦合的, 不能拆开 cherry-pick.**

### 5.2 Hyperparameter ablation (Figure 3, page 9)
- **Segment length L_s**: 32 是 optimal (Figure 3a). 太长 → segment retrieval R@1 涨但 oracle grounding R1@0.3 掉, overall 反 U.
- **N_rep**: increasing 帮 segment retrieval 但 fine-grained 几乎不变, N_rep=4 后 saturate (Figure 3b).

### 5.3 Robustness (Section 3.6, Table 8)
Charades-STA 上做两个鲁棒性实验:
- **Time shift**: 把 event 重新洗到 video 各位置 (反 distributional bias). UniTime-SP shift/non-shift ratio = 80.20 (越高越鲁棒), VTimeLLM 53.67, Mr.BLIP 66.77, TimeSuite 66.26. → UniTime 对位置 bias 不敏感.
- **Query decomposition (IoG)**: 用 Qwen2 把 complex query 拆成 `"When does <object> appear?"` 子问题, 测每个子问题的 grounding. UniTime 74.88 vs VTimeLLM 68.99, Mr.BLIP 71.22, TimeSuite 47.07.

### 5.4 Downstream VideoQA (Section 3.7, Table 9, page 9)
作为 long-video QA 的 preliminary moment retriever, 比 uniform 32-frame baseline:
| Benchmark | Uniform Sample | UniTime-Full |
|---|---|---|
| QaEgo4D | 49.60 | **55.51** (+5.91) |
| CG-Bench | 33.87 | **40.30** (+6.43) |
| MLVU | 60.53 | **66.50** (+5.97) |
| LongVideoBench | 54.82 | **56.47** (+1.65) |

> 用 UniTime 先找相关 segment, 再 sample 32 帧喂 Qwen2-VL-7B 答题, 比直接 uniform sample 涨 ~6 acc pts (LongVideoBench 涨幅小, 可能因为它的 question 不那么时间敏感).

## 6. Related work positioning (Section 4)

把 MLLM-based VTG 分三类:
1. **Time-agnostic**: VTimeLLM (固定 100 frames + normalized position), LITA (special time tokens) — 没有 explicit temporal signal, sub-optimal.
2. **Implicit timestamp-encoded**: TimeChat, TimeSuite (timestamp-aware vision encoder), VTG-LLM (absolute time embed), Qwen2.5-VL (MRoPE) — 要 extensive pretrain, 容易 hallucinate.
3. **Explicit timestamp marking**: Mr.BLIP, TimeMarker, VideoLLaMA3 — prepend textual timestamp, 跟 UniTime 同一阵营. UniTime 的差异化在于 **adaptive scaling + multi-stage** 让它能 scale 到 long video.

## 7. 跟 Tieqiao 的 TAS 方向的 connection (我的判断)

**VTG vs TAS 任务差异**:
- VTG: 给一个 query, 输出一个 (or 几个) 时间窗口. **Sparse output**.
- TAS: 给一个 video, 标每一帧的 action class. **Dense output, 闭集 action vocab**.

**UniTime 直接套用 TAS 的可行性**:

1. **可以做的**: 把 TAS 的每个 action class 变成 query (`"person picking up cup"`), 跑 N_class 次 grounding, 把 predict 的窗口投票回到帧级 label. 但是:
   - TAS 数据 (Breakfast / 50Salads / GTEA / Assembly101) **不在** UniTime 的 pre-train mix 里, 域 gap 明显
   - TAS 一个 video 内 action 密集且短 (50Salads 平均 ~30 instances/video), UniTime 是给 sparse moment 设计的, retrieval-style 输出处理不了 dense
   - cooking 类 dataset (YouCook2, COIN, Momentor) 在 pretrain 里 → 这部分迁移可能还行

2. **更有意思的方向**: 借鉴 **timestamp-interleaved sequence** 这个 trick, 不是 retrieve 一个窗口而是让 LLM **decode action class per frame** (用文本 token 输出 frame label sequence). 这就把 UniTime 从 grounding 改成 dense prediction. 已知问题:
   - 输出 sequence 很长 (每帧一个 label), 推理慢
   - LLM 对长输出的一致性差
   - 没有现成 SOTA 对比

3. **最 incremental 的**: 复用 UniTime 的 **coarse-to-fine multi-stage inference** 思想 —— TAS 也可以 hierarchical (先粗粒度 segment level, 再 refine boundary). 这个不依赖 timestamp-interleave.

**reproduction 优先级建议**: 先按论文复现 Charades-STA (短 video, dataset 小, 4.8K videos / 31s 平均, ~10GB), 验证 pipeline 通了, 再想 TAS 怎么蹭.

## 8. 复现要点

- **算力**: 论文 8 GPU, batch=8 (per-device 1), 1 epoch on Charades 应该几小时. DGX H100 × 8 应该够.
- **关键 hyperparam**: fps=2, N_f^short=128, N_f^long=1024, L_s=32, N_rep=4, lr=2e-4. **不要乱改**.
- **环境锁版本** (上游 requirements.txt): Python 3.10, torch 2.1.2 + cu121, transformers 4.49.0, peft 0.14.0, deepspeed 0.16.4, flash-attn 2.7.2.post1.
- **video data**: UniTime-Data HF dataset 只给 annotation JSON, video 要从原始 dataset 各自下.
- **HF ckpt 是 LoRA only** (~150MB), 必须配 Qwen2-VL-7B-Instruct (~16GB) base.

## 9. 可能的弱点 / 没回答的问题

1. **ANet-Captions UniTime-Full 没涨** (R1@0.5 36.62 vs prev SoTA 53.92, UniTime-SP 36.62). 论文说 Appendix C.2 有 analysis 但 main text 没解释. ← 待读 appendix.
2. **Adaptive scaling 在 ablation 单独无用** (row 2 vs row 1) — 这是 paper 老实说的. 但意味着如果 video 都是短的, 这一招完全没价值. 对 TAS 这种 fixed length cooking video 可能用不上.
3. **没跟基于 CLIP 特征的 fast model 比效率** (UniVTG, Moment-DETR 等). 7B LLM 推理肯定慢, 但论文不报 latency / throughput.
4. **video-centric training 的 attention mask 实现细节在 Appendix B.2**, 还没读. 如果要改 training 必须搞清楚.
5. **没做 incremental decoding / streaming**. 长 video 的 multi-stage 是 offline iterative, 不能 streaming.

## 10. 引用规则

之后任何对话引用这篇论文必须 quote 本文件具体行号. 例: `paper_notes/01_unitime.md:120`.

References (本笔记里出现的关键 number/table 都来自):
- Table 1 (datasets stats) — page 5
- Table 3 (dataset-specific) — page 6
- Table 4 (zero-shot) — page 7
- Table 5 (closed-source) — page 7
- Table 6 (flexibility / backbones) — page 7
- Table 7 (module ablation) — page 8
- Table 8 (robustness) — page 8
- Table 9 (downstream VideoQA) — page 9
- Figure 3 (hyperparam ablation) — page 9
- eq. 1, 2 (sequence construction) — page 4
