# 01 — UniTime: Universal Video Temporal Grounding with Generative MLLMs

> **Status**: 已读全文 (main 10p + appendix 12p, 共 22 pages)
> **Read date**: 2026-04-07 (main), 2026-05-03 (appendix + 全文复查)
> **PDF**: [`pdfs/01_unitime.pdf`](pdfs/01_unitime.pdf)
> **Experiment**: [`experiments/unitime/`](../experiments/unitime/)

## Metadata
- arxiv 2506.18883v2, 2025-11-21
- NeurIPS 2025
- Zeqian Li (SJTU SAI), Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, Weidi Xie
- SJTU SAI + ByteDance Seed
- Code: https://github.com/Lzq5/UniTime
- Ckpt: HF `zeqianli/UniTime` (LoRA adapter on Qwen2-VL-7B)

## Problem (Section 2.1, p.3)
- 输入: video V = {f_1, ..., f_{N_f}} + timestamps T = {t_1, ..., t_{N_f}} + query Q
- 输出: Y = {(s_k, e_k)} 一组 temporal moments
- s_k, e_k ∈ T (输出是从采样 timestamps 里 retrieve, 不是回归浮点数)

## Method

### Adaptive Frame Scaling (Section 2.2, p.3)
- 以 fps=2 采样, 得到 N_f 帧
- 总 token budget N_total = 16,384 (固定上限)
- 每帧分配 N_res = ⌊N_total / N_f⌋ tokens
- 两个阈值 N_f^short = 128, N_f^long = 1024
- **短视频** (N_f < N_f^short = 128):
  - 用 **frame resizing** (ψ_resize): 改变输入图片分辨率, 使每帧被 vision encoder 分成 N_res 个 patches
  - V_i = φ_project(φ_vision(ψ_resize(f_i))) ∈ R^{N_res × d}
  - 每帧 tokens 多, 空间分辨率高
- **中等视频** (N_f^short ≤ N_f < N_f^long):
  - 用 **token compression** (ψ_compress): 全分辨率过 vision encoder, 然后用双线性插值压缩到 N_res tokens
  - V_i = ψ_compress(φ_projector(φ_vision(f_i))) ∈ R^{N_res × d}
  - 保留更多语义信息 (不丢输入分辨率), 但计算更贵
- **长视频** (N_f > N_f^long = 1024):
  - 切成多个 clips, 每段 N_f^long 帧, divide-and-conquer
- **关键**: frame resizing 和 token compression 是两种不同机制. Table 11 (Appendix E.1, p.21) 显示两者在短视频上性能相当, 但 token compression 训练慢 5 天. 最终选择 hybrid: 短视频 frame resizing, 长视频 token compression.

### Timestamp-Interleaved Sequence (Section 2.2, p.3-4)
- 每帧 f_i 的 timestamp 编码成文本: τ_i = "timestamp: t_i seconds"
- **Fine-grained** (eq.1, p.4): 每帧前都插 timestamp
  - S = [T_1; V_1; T_2; V_2; ...; T_{N_f}; V_{N_f}; Q]
  - T_i = φ_tokenizer(τ_i)
- **Coarse-grained** (eq.2, p.4): 把帧分成 N_s 个 segment, 每段 L_s 帧
  - S_j = [V_{s_j}; V_{s_j+1}; ...; V_{s_j+L_s-1}] — 段内所有帧的 visual tokens **都保留**
  - S = [T_1; S_1; T_2; S_2; ...; T_{N_s}; S_{N_s}; Q]
  - 每段只在段头插一个 timestamp t_{s_j}
  - **视觉 tokens 没减少, 只是 timestamp 文本变少了**
- 输出格式:
  - Segment retrieval: "the specific timestamp(s) when the given query appears" → 输出匹配的 timestamps (mr_seg mode)
  - Fine-grained: "the temporal window (start and end timestamps)" → 输出 "From s seconds to e seconds"

### Multi-stage Inference (Section 2.2, p.5)
- 长视频处理: coarse-to-fine
- 第一轮: coarse-grained (段级 timestamp) 在每个 clip 内做 segment retrieval → 候选 segments
- 第二轮: 在候选 segments 内做 fine-grained grounding (每帧 timestamp)
- 可递归重复

### Training (Section 2.3, p.5)
- Loss: 标准 auto-regressive NLL, 只对 target tokens 算
- **Video-centric training** (Appendix B.2, p.18):
  - 一条训练样本 = 一个 video + 它的所有 (query, answer) pairs
  - 序列: [v_1, v_2, ..., v_N, Q_1, A_1, Q_2, A_2, ..., Q_{N_sample}, A_{N_sample}]
  - Attention mask 阻止不同 Q-A pair 互相看
  - 每个 Q-A pair 共享相同 RoPE starting position index (= video tokens 之后)
- Replication factor N_rep = 4: 复制长视频样本平衡数据分布

## Experimental Setup (Section 3.1, p.6)
- Base model: Qwen2-VL-7B
- Vision encoder 冻结, LLM 用 LoRA (rank=8, alpha=8)
- fps = 2
- N_f^short = 128, N_f^long = 1024
- Token cap N_total = 16,384
- Segment length L_s = 32 (Figure 3a 显示 32 是 overall R1@0.3 的 sweet spot)
- N_rep = 4
- AdamW, lr=2e-4, batch_size=8, 1 epoch, 3% linear warmup
- Training data (Table 1, p.5):
  - Part I (pre-training only): NaQ 1031K, DiDeMo 33K, QuerYD 5.7K, HiRest 4K, COIN 46.4K, Momentor 136.4K, YouCook2 9.6K — 总计 ~1.27M queries
  - Part II (benchmark train): Ego4D-NLQ 10K, TaCoS 9.8K, Charades-STA 11.2K, QVHL 7.2K, ANet-Captions 37.4K

## Results

### Dataset-specific & Universal (Table 3, p.6)
- UniTime-SP: 每个 dataset 单独 fine-tune
- UniTime-Full: 一个模型在 Part I + Part II 全部数据训练
- Ego4D-NLQ R1@0.3: 24.79 (SP), **27.09** (Full), prev SoTA 18.28 → +8.81
- TaCoS R1@0.3: 61.18 (SP), **66.91** (Full), prev SoTA 57.61 → +9.30
- Charades-STA R1@0.5: 74.33 (SP), **75.27** (Full), prev SoTA 70.20 → +5.07
- ANet-Cap R1@0.5: 36.62 (SP), **53.67** (Full), prev SoTA 53.92 → -0.25 (唯一没涨的)
- QVHighlights R1@0.5: 77.76 (SP), **76.72** (Full), prev SoTA 74.77 → +1.95

### Zero-shot (Table 4, p.7)
- UniTime-Zero: 只在 Part I 训练
- TaCoS R1@0.3: **50.06** vs Mr.BLIP 24.59 → +25.47
- Charades R1@0.5: **59.09** vs TimeMarker 51.90 → +7.19
- Ego4D-NLQ R1@0.3: **14.67** vs TimeSuite 0.88 → +13.79

### Closed-source Comparison (Table 5, p.7)
- Seed1.5-VL Charades 最强 (mIoU 73.69), Gemini-2.5-Pro Ego4D 最强 (mIoU 20.45)
- 没有闭源模型在长短视频都好. UniTime-Full 两边都 competitive.

### Flexibility / Model-agnostic (Table 6, p.7)
- Qwen2-VL-2B: baseline 0.34 R1@0.3 → +UniTime 10.50 R1@0.3
- Qwen2-VL-7B: baseline 0.48 → +UniTime 24.79
- Qwen2.5-VL-7B: baseline 11.41 → +UniTime 34.27
- InternVL2.5-2B: baseline 0.33 → +UniTime 23.55
- InternVL2.5-8B: baseline 0.33 → +UniTime 16.25 (注: 8B 比 2B 低, 论文没解释)
- 结论: 方法 model-agnostic, 对小模型也有效

## Ablations

### Module Ablation (Table 7, p.8, Appendix A.2.1 p.15)
- Row 1 (baseline): uniform 32 frames, 每帧一个 timestamp → R1@0.3 = 14.25
- Row 2 (+ Adaptive Scaling): 2fps + frame scaling → R1@0.3 = 14.00 (**单独无效**)
- Row 3 (+ Multi-stage): uniform 32 frames → coarse prediction → re-sample within predicted interval → R1@0.3 = 18.42
- Row 4 (+ Multi-stage + Adaptive Scaling): 2fps + frame scaling + multi-stage → R1@0.3 = 17.91
- Row 5 (全部): Adaptive Scaling + Segment Retrieval + Multi-stage = **24.79** R1@0.3
- 关键: Row 5 用 coarse-to-fine 两阶段 (先 segment retrieval, 再 fine-grained), Row 4 用 single-scale multi-stage. 三个 module 耦合, 不能拆.

### Segment Length (Figure 3a, p.8, Appendix A.2.2 p.15-16)
- L_s = 32 是 overall R1@0.3 sweet spot
- L_s 越大: segment retrieval R@1 越高 (段越大越容易命中), 但 oracle grounding R1@0.3 越低 (段内定位变难)
- 这是 coarse-to-fine segment retrieval 用的, fine-grained 阶段每帧都有 timestamp

### Replication Factor (Figure 3b, p.8)
- N_rep = 4 最优, segment retrieval 随 N_rep 涨后饱和, fine-grained 稳定

### Video Processing Strategies (Table 11, Appendix E.1, p.21)
- Frame Resizing (短) + Token Compression (长): R1@0.3 = 24.06, R1@0.5 = 16.18, mIoU = 16.71 — **最终选择**
- Frame Resizing (短) + Frame Resizing (长): R1@0.3 = 18.53, R1@0.5 = 12.67 — 长视频用 frame resizing 性能差
- Token Compression 全用: R1@0.3 = 24.79, R1@0.5 = 16.83 — 性能最好但训练慢 5 天

### Temporal Information Encoding (Table 12, Appendix E.2, p.22)
- Dense Position Encodings (MRoPE): R1@0.5 = 48.44, R1@0.7 = 27.15
- Timestamp Token Insertion (UniTime-SP): R1@0.5 = **74.33**, R1@0.7 = **53.71** → +25.89
- 显式 timestamp 远优于隐式 position encoding

### Adaptive Frame Scaling Hyperparams (Appendix E.3, p.22)
- N_f^long: 变化影响小 (Table 13), 设 1024
- N_f^short: 64-128s 视频 multi-stage 和 single-stage 差不多, 128-256s 视频 multi-stage 显著更好 (Table 14), 设 128
- N_total (token budget): 性能随 budget 单调增加 (Table 15), 设 16,384 (GPU 上限)

## 与我们 GTEA 实验的关系

### GTEA 视频特征
- GTEA 视频 40-80 秒, fps=2 → N_f = 80-160 帧
- N_f < N_f^short (128) 的 → 用 frame resizing (高分辨率)
- N_f ≥ 128 的 → 用 token compression

### Combine / Segment 的正确理解
- **Segment (L_s=32)** 是 coarse-grained variant (eq.2) 用的: 段内所有帧 visual tokens 保留, 只是 timestamp 每段一个
- CLIP_LENGTH=32 在 train.sh 里对应的就是 L_s
- 对 GTEA: N_f=120 帧, L_s=32 → 3-4 个 segments, 每个 segment 只有一个 timestamp → 模型只能输出 3-4 个时间值
- 这在论文设计中是 **coarse-grained 第一阶段**, 后面还有 fine-grained refine. 但我们只跑了 single-stage, 没有 multi-stage → coarse-grained 的低精度没被 refine 补回来.

### 论文 ablation Row 1 的启示
- Table 7 Row 1: "uniformly sample 32 frames, 每帧一个 timestamp" → R1@0.3 = 14.25
- 这正是我们 Gemma3/4/Qwen3 做的事 (32 帧, 每帧一个 timestamp, 无 combine)
- 论文在 Ego4D-NLQ 上 Row 1 baseline = 14.25, 全部 module = 24.79 (+10.54)
- 说明 32-frame uniform + per-frame timestamp 是论文认可的 baseline 配置

## 引用规则
引用必须 quote 行号. 例: `paper_notes/01_unitime.md:42`
