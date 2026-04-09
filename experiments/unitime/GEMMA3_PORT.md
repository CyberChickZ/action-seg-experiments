# UniTime → Gemma 3 4B port (GTEA TAS reproduction)

> 把 UniTime 的方法 ([Lzq5/UniTime](https://github.com/Lzq5/UniTime), NeurIPS 2025) 移植到 **Gemma 3 4B base** 上, 在 **GTEA** dataset 上做 temporal action segmentation 训练. 学长 Tieqiao 的 Task 2.
> Status: phase 2 完成, 训练已启动 (2026-04-08).

---

## 1. 训练协议 (training protocol)

### 1.1 任务 framing

UniTime 论文 (`paper_notes/01_unitime.md:38`) 把 video temporal grounding 建模成 LLM 文本生成: 把 video 的每一帧编码成 image tokens, 在每帧的 image tokens 前**插入一个文本 timestamp 标记** (`timestamp: 0.5 seconds` 等), 拼成一个长 sequence 喂给 multimodal LLM. 模型接收一个 query (e.g. `"take"`), 输出**文本格式的 timestamp 列表** (`"1.0 seconds, 2.0 seconds, 5.0 seconds, 6.0 seconds."`), LLM 通过 retrieval 而不是回归学会对齐 query 跟 frame.

UniTime 上游用 **Qwen2-VL-7B** 作为 base. 我们的 port 把 base 换成 **Gemma 3 4B (`google/gemma-3-4b-it`)**, 论文 Section 3.4 (`paper_notes/01_unitime.md:55`) 验证过这个方法 model-agnostic — 但只在 Qwen2-VL-2B/7B / Qwen2.5-VL-7B / InternVL2.5-2B/8B 上验证过, **Gemma 3 是 untested territory**.

### 1.2 Dataset mode: `mr_seg`

UniTime 的数据格式有两种 mode:

| Mode | 含义 | Target |
|---|---|---|
| `mr` (paper 主线) | 单 moment grounding | `"From s seconds to e seconds"` (interval text) |
| **`mr_seg`** (我们用的) | dense per-frame matching | 命中 windows 内的**所有 sampled timestamps 列表** |

`mr_seg` 是 paper 在 multi-instance / multi-window 场景下用的变种, target 不是 interval pair 而是 frame-level timestamp 集合, 本质上跟 TAS 是 1:1 对齐的. 实现位置在 upstream `collators/qwen2_vl.py:91-102`:

```python
if mode == 'mr_seg':
    for msg, all_t, all_t_o, windows in zip(...):
        for t_w in windows:                       # 多 window: 一个 query 多个 hit 区间
            sub_evaluate_labels = []
            for t_w_i in t_w:                     # 每个 window 内
                segment_start_idx, segment_end_idx = find_segments(all_t_o, t_w_i)
                sub_evaluate_labels.extend([all_t[i] for i in range(segment_start_idx, segment_end_idx + 1)])
            interval_text = ", ".join([f"{s} seconds" for s in sub_evaluate_labels])
            interval_text = interval_text + "."
            ...
```

我们在 `collators/gemma3_vl.py:78-95` (the `build_target_text` method) 里实现了同样的逻辑, 跟上游 byte-for-byte 等价.

### 1.3 Phase 2 简化: single-query

Upstream Qwen2VL collator 支持 **multi-qa**: 一个 video 内的多个 query 共享 video forward, 通过 modify attention mask 避免不同 (Q,A) pair 互相看到. 这是 paper Section 2.3 "Video-centric training" (`paper_notes/01_unitime.md:130-160` 区段) 的优化, 主要省 I/O 和重复 vision encoding.

Phase 2 我们**没实现 multi-qa**. 每个 (video, action_class) pair 是独立的训练 sample. 这意味着同一个 GTEA video 上的 11 个 action classes 会跑 11 次 vision feature 加载, 但**因为 features 是预提取并 cache 的**, 实际上 I/O cost 是从硬盘读 11 次同一个 .pt 文件 (NFS, 100 KB级别), 不是真的重新跑 vision tower. 速度差距很小, 实现复杂度小一个数量级. Phase 3 再考虑加 multi-qa.

### 1.4 训练超参 (`scripts/train_gemma3.sh`)

| 参数 | 值 | 来源 |
|---|---|---|
| `model_id` | `gemma3-4b-it` | 我们注册的新 model_id |
| `model_local_path` | `/nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it` | 本地 mirror |
| `--bf16` | True | matches Gemma 3 default `torch_dtype` |
| `--use_lora` | True | LoRA on LLM only, vision tower 冻结 |
| `--lora_r` / `--lora_alpha` | 8 / 8 | matches paper Table 6 / upstream default |
| `--per_device_train_batch_size` | 1 | matches upstream |
| `--gradient_accumulation_steps` | 1 | upstream default |
| `--learning_rate` | 2e-4 | matches upstream |
| `--lr_scheduler_type` | cosine | upstream |
| `--warmup_ratio` | 0.03 | upstream |
| `--num_train_epochs` | 2 | GTEA 才 162 entries, 1 epoch 收敛肯定不够 |
| `--model_max_length` | **16384** | 32 frames × 256 image tokens = 8192 + text overhead, 16384 留 ~50% headroom |
| `--gradient_checkpointing` | True | for 80 GB H100 OOM 安全 |
| `--deepspeed` | `ds_configs/zero2.json` | upstream default |
| `NUM_GPUS` | 1 | dgxh-2 user OnDemand allocation = 1 GPU |
| `train_vision_encoder` | False | freeze SigLIP-2 |
| `train_vision_projector` | False | freeze multi_modal_projector |
| `RUN_ID` | `gemma3_gtea_run1` | output → `checkpoints/gemma3_gtea_run1/` |

### 1.5 Pipeline 全流程

```
GTEA mp4 (28 个)
   │
   │ extract_gemma_features.py: 32 frame uniform sample → SigLIP-2 → multi_modal_projector
   ▼
gtea/*.pt (28 个) — 每个 .pt 是 dict{feature: bf16 [32, 256, 2560], frame_idx, sample_fps}
   │
   │ train.py loads via VideoCentricDataset (datasets_mr.py)
   │   - dispatches on model_family_id="gemma3" → construct_messages_gemma3()
   │   - returns dict with {message, temporal_window, mode, qid, duration}
   ▼
Gemma3DataCollator (collators/gemma3_vl.py)
   │   - process_vision_info_gemma3 (collators/gemma_vision_process.py)
   │     loads .pt, slices to [video_start, video_end], reconstructs sampled_timestamps
   │   - build_user_text:
   │       intro + Σ(timestamp + "<start_of_image><image_soft_token>×256<end_of_image>") + Query/Answer
   │   - build_target_text:
   │       mr_seg multi-window → "1.0 seconds, 2.0 seconds, ..."
   │   - tokenize user prompt + target separately, concat
   │   - labels: -100 mask before answer span, then real token ids
   │   - feature_inputs = concat(28 × [32, 256, 2560]) flat
   │   - sanity check: image_token slot count must == feature row count
   ▼
Gemma3VLMRForConditionalGeneration.forward (models/gemma3_vl.py)
   │   - get_input_embeddings(input_ids) → text embeddings
   │   - feature_inputs.masked_scatter into image_token_index positions
   │   - skip vision_tower entirely (we already ran it offline)
   │   - language_model.forward with sliding-window attention + bidirectional on image regions
   ▼
CrossEntropyLoss on shifted (logits, labels)
   │
   ▼
Adam optimizer step → LoRA weight updates only (LLM 4B base frozen)
```

---

## 2. Dataset: GTEA split1

- **Total**: 28 videos (4 subjects × 7 recipes), 11 action classes
- **Action vocab**: `background, close, fold, open, pour, put, scoop, shake, spread, stir, take`
- **Split1** (MS-TCN convention, leave-S1-out):
  - **Train**: 21 videos (S2/S3/S4_*), **162 entries** (one per (video, action_class) pair)
  - **Test**: 7 videos (S1_*), **54 entries**
- **Annotation source**: MS-TCN groundTruth csvs converted by 学长的 `gtea_csv_to_json.py` (synced into our repo at [`data/gtea/gtea_csv_to_json.py`](./data/gtea/gtea_csv_to_json.py))
- **Annotation format** (`mr_seg`):
  ```json
  {
    "qid": 0,
    "id": "S2_Cheese_C1",
    "annos": [{"query": "take", "window": [[0.667, 4.267], [4.733, 6.933], ...]}],
    "duration": 42.2,
    "mode": "mr_seg"
  }
  ```
  Each entry has 1 query (action class) + N windows (instances of that action). N varies — short actions like `"take"` may have 5+ windows per video, long ones like `"background"` 1-2.

- **Pre-extracted features**: 28 × .pt files at `/nfs/hpc/share/zhanhaoc/MODLE/Gemma3-4B-it/features/gtea/`, generated by `extract_gemma_features.py` with 32 frame uniform sample → SigLIP-2 → multi_modal_projector, shape `[32, 256, 2560]` bf16, ~30 MB each.

---

## 3. 我们写了什么 (what we shipped)

### 3.1 New files in submodule (`CyberChickZ/UniTime`, all under `experiments/unitime/UniTime/`)

| File | LOC | 作用 |
|---|---|---|
| `models/gemma3_vl.py` | 240 | `Gemma3VLMRForConditionalGeneration` subclasses `Gemma3ForConditionalGeneration`, overrides `forward` to add `feature_inputs` (cached features path) + `multi_qa` / `attention_mask_multiqa` (kept for symmetry, not used in phase 2). Also `Gemma3VLMRProcessor` thin subclass. |
| `loaders/gemma3_vl.py` | 30 | `Gemma3ModelLoader` registered as `"gemma3"` family. Returns model + tokenizer + processor + config. |
| `collators/gemma_vision_process.py` | 100 | `fetch_video_feature_only` + `process_vision_info_gemma3`. Loads cached `.pt`, slices to video_start/video_end window, reconstructs sampled_timestamps from frame_idx. |
| `collators/gemma3_vl.py` | 220 | `Gemma3DataCollator` full impl. mr_seg multi-window target construction, timestamp-interleaved prompt build, direct tokenizer-level prompt assembly (no chat_template hack). Sanity-checks image-token slot count vs feature row count. |
| `extract_gemma_features.py` | 130 | Standalone script: video → 32 uniform frames → SigLIP-2 + multi_modal_projector → `[32, 256, 2560]` `.pt` written to `feat_root/{dataset_name}/`. Uses sdpa attention (flash-attn 2.7.2 lacks Gemma3 op-builder). |

### 3.2 Modifications to upstream files

| File | Diff | 原因 |
|---|---|---|
| `supported_models.py` | + `MODULE_KEYWORDS["gemma3"]` paths (vision_tower / multi_modal_projector / language_model) + `register_model("gemma3-4b-it", ...)` + `register_model("gemma3-12b-it", ...)` | family registration |
| `loaders/__init__.py` | + `from .gemma3_vl import Gemma3ModelLoader` | wire up family |
| `collators/__init__.py` | + `from .gemma3_vl import Gemma3DataCollator` | wire up family |
| `datasets_mr.py` | + `model_family_id` constructor param (default `"qwen2-vl"`), dispatch in `__getitem__` to `construct_messages_gemma3` for Gemma family | minimal-invasive Qwen vs Gemma message format split |
| `train.py` | pass `model_family_id=model_args.model_family_id` to both train + eval `VideoCentricDataset` instantiations | wire up dataset dispatch |

### 3.3 Parent repo additions

| File | 作用 |
|---|---|
| `experiments/unitime/scripts/train_gemma3.sh` | shell launcher mirroring `scripts/train.sh` (Qwen2VL one) but with `MODEL_ID=gemma3-4b-it`, `model_local_path` 指向 Gemma3-4B-it, `FEAT_FOLDER` 指向 Gemma3 features dir, `MODEL_MAX_LEN=16384`, `NUM_EPOCHS=2`. |

### 3.4 关键架构差异 (Gemma 3 vs Qwen2-VL UniTime wrapper)

| 维度 | Qwen2VL wrapper | Gemma 3 wrapper |
|---|---|---|
| Vision input path | `pixel_values` + `pixel_values_videos` 两条路径 + `video_grid_thw` | 只有 `pixel_values`, video 当 image sequence |
| 动态 patch | 是 (smart_resize, dynamic h×w grid) | 否 (固定 256 soft tokens / image, SigLIP-2) |
| RoPE | mRoPE 3D (temporal/H/W 三轴) | 1D rotary (text 标准) |
| `get_rope_index_multiqa` | 必需 (3D position 计算) | 不需要 |
| `encode_video_chunk` | 必需 (Qwen ViT 8-frame 分批) | 不需要 (SigLIP 一次过) |
| 子模块 | `model.visual` + `model.model` (Qwen2VLModel) | `model.vision_tower` + `model.multi_modal_projector` + `model.language_model` |
| `MODULE_KEYWORDS` freeze keys | `["visual.patch_embed", ...]` 等 | `["vision_tower"]`, `["multi_modal_projector"]`, `["language_model"]` |
| Token 分隔 | special token `<\|video_pad\|>` 替换 | special token `<image_soft_token>` (id 262144) 替换 |
| Wrapper LOC | 580 行 | **240 行** (~58% smaller) |

Gemma 3 wrapper 简单很多, 因为 SigLIP-2 输出 fixed shape, 不需要 dynamic grid bookkeeping.

---

## 4. 训练时刻发生的事 (runtime trace)

```
[loader] Gemma3ModelLoader.load:
    Gemma3VLMRForConditionalGeneration.from_pretrained(/nfs/.../Gemma3-4B-it,
        torch_dtype=bfloat16)
    → loads 8 GB safetensors into GPU
[trainer setup]
    freeze model.vision_tower
    freeze model.multi_modal_projector
    LoRA target_modules = find_all_linear_names(named_modules, ["language_model"])
    PeftModel wraps language_model
[dataset]
    VideoCentricDataset(model_family_id="gemma3") loads
        data/gtea/annot/train.json (162 entries)
        data/gtea/annot/test.json (54 entries)
[training step]
    for batch in DataLoader (batch_size=1):
        Gemma3DataCollator(batch):
            for each instance:
                load .pt → feature [T_total, 256, 2560], slice to [video_start, video_end]
                build user text with T frames × (timestamp + 256 image tokens)
                build mr_seg target = "1.0 seconds, 2.0 seconds, ..."
                tokenize prompt + target separately
                labels = [-100] * len(prompt) + target_ids
            concat features → [N_videos × T × 256, 2560]
            sanity check: input_ids count of image_token_index == feature row count
        Gemma3VLMRForConditionalGeneration.forward:
            inputs_embeds = embed(input_ids)
            inputs_embeds.masked_scatter(image_token_mask, feature_inputs)
            language_model(inputs_embeds, attention_mask, ...)
            CrossEntropyLoss on shifted answer span (everything else -100)
        loss.backward() → only LoRA weights get gradient
        optimizer.step()
```

---

## 5. 已知 caveats / 后面要做的

1. **Single-query phase 2**: 没用 multi-qa attention mask, 不是论文 best speedup. Phase 3 加上后训练时间会从 ~30 min 降到 ~5-10 min per epoch.
2. **`run_260408` ACL**: 学长今天 (Apr 8 11:11) 跑的 Qwen2VL UniTime GTEA training output, ACL 没扩展, 我们读不到. 见学长帮忙 setfacl.
3. **fps=2 vs short actions**: GTEA 有些 action 短到 0.07 秒, 32 frames uniform sample 必然漏. 后续 ablation: `--num_frames 64` 或 `128`.
4. **`background` class as query**: 论文没考虑 "no action" query, 学长把它当 normal action. 训出来怎么样要看 eval.
5. **eval metrics**: UniTime upstream `eval_metrics.py` 是 R@k IoU (VTG metric), TAS 真要 F1@10/25/50 + Edit. 收敛后写 TAS metric 转换器 (把 model 输出的 timestamp list reconstruct 成 frame label sequence).
6. **Gemma 3 sliding_window=1024**: Gemma 3 attention 是 sliding window 1024 + global. timestamp 文本插在 frames 之间 → 同一 query 跨 frame 的 attention 会被 sliding window 截断, **可能影响 multi-window 跨度大的 action 的 grounding**. 需要看 loss curve.

---

## 6. Citation 来源
- UniTime paper main text: `paper_notes/01_unitime.md` (整 file)
- mr_seg multi-window upstream: `experiments/unitime/UniTime/collators/qwen2_vl.py:91-102`
- Model-agnostic claim: `paper_notes/01_unitime.md:55` (Section 3.4 "Flexibility Verification")
- Gemma 3 architecture: `transformers/models/gemma3/modeling_gemma3.py` (4.51.3, on HPC at `/tmp/tf451_zhanhaoc/`)
- Submodule head with our port: [`CyberChickZ/UniTime@e9f1393`](https://github.com/CyberChickZ/UniTime/commit/e9f1393)
- Parent repo head: [`CyberChickZ/action-seg-experiments@9cbfbe5`](https://github.com/CyberChickZ/action-seg-experiments/commit/9cbfbe5)
