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

## 2026-04-08 16:00 — TASK 1 跑通: Qwen2VL UniTime GTEA real training (with caveat)
- Harry 在 dgxh-2 OnDemand shell 跑了 `bash ../scripts/train.sh` (Qwen2VL train, model_id=qwen2-vl-7b-instruct, single GPU H100 80GB).
- **162 steps, 23 分钟**, 1 epoch on GTEA split1 (162 train entries × 1 epoch).
- **Loss curve**: step 0 = 3.36 → step 5 = 1.98 → step 30 = 0.32 → step 80 = 0.18 → step 162 = 0.107. Smooth, no NaN, no divergence.
- final `train_loss: 0.258`, `train_runtime: 1397s`, `samples_per_second: 0.116`.
- Checkpoint at `experiments/unitime/UniTime/checkpoints/run1/` (LoRA adapter, ~150 MB).
- **CAVEAT (必须 record)**: HF tokenizer warning `Token indices sequence length is longer than the specified maximum sequence length for this model (16490 > 4096). Running this sequence through the model will result in indexing errors`. 即 GTEA mr_seg 序列实际是 16490 token, 但 `scripts/train.sh` MODEL_MAX_LEN=4096 (我之前为 fit 40 GB MIG slice 改的, 见 commit `3b56855`). **整段被截到前 4096 token, 后 ~75% 的 video frames 没进 LLM**. Loss 看起来好其实是模型在偷懒只看前 ~10 帧学一个 prior. 真 baseline 必须 bump max_len + 重训.
- **Action**: bump `scripts/train.sh:MODEL_MAX_LEN` 4096 → 24576 (UniTime paper 用 32768, 24576 fit 16490 + headroom 在 80 GB 上够). Re-run for the "真 Task 1 baseline".
- 学长 ACL 还没拿到, `run_260408/` 还是看不到 — 但**我们自己的 run1/** 现在就是 task 1 record 的 first-party 数据.

## 2026-04-08 16:00 — Gemma3 extract segfault: flash-attn 2.7.2 不支持 SigLIP-2
- Harry 跑 `python extract_gemma_features.py ... --num_frames 32` → `Segmentation fault (core dumped)` 无 traceback.
- 根因 (推测, 极高概率): `extract_gemma_features.py` 硬编码 `attn_implementation="flash_attention_2"` 加载 Gemma3VLMRForConditionalGeneration. **flash-attn 2.7.2 (UniTime upstream pin) 没有给 Gemma3 SigLIP-2 vision tower 的 op-builder**, 加载时 C 层崩.
- **Fix**: extract script 改成 `attn_implementation="sdpa"` (PyTorch 内置 scaled dot product attention, 支持任何架构, H100 上够快做 offline feature extraction).
- 我们的 train_gemma3.sh 不受影响 — 它走 train.py → loaders/base.py:BaseModelLoader, `use_flash_attn` default False → 不传 `attn_implementation`, transformers 自动选 sdpa. Extract script 是 standalone python 才需要显式 sdpa.

## 2026-04-08 14:10 — Tieqiao 实际需求 + Muse Spark 闭源 + Gemma 3 port plan
- 学长 Discord 截图明确两个**并行** task: (1) GTEA real training using UniTime, record what happens; (2) 微调 Muse Spark 或 Gemma, **(not necessary use Unitime)** 修饰的是这俩, 学长说不强制 UniTime 框架.
- **Muse Spark 闭源** (WebFetch https://ai.meta.com/blog/introducing-muse-spark-msl/ 验证): Meta Superintelligence Labs, multimodal reasoning, 只在 meta.ai web/app + private API preview, **没 weights, 没 finetuning code**. 按学长指示 skip.
- **Task 1 (real GTEA training UniTime+Qwen2-VL) 学长今天 (Apr 8 11:11) 已经跑了**, output 在 `/nfs/stak/users/wangtie/2026/3/15/UniTime/checkpoints/run_260408/`, **但权限 `drwxrwx---` 没 ACL 扩展, zhanhaoc 读不到** (我 ssh 试过 `Permission denied`). 必须让学长 `setfacl -R -m u:zhanhaoc:rx ...` 才能 record.
- **Task 2 = port UniTime 方法到 Gemma 3 base + 在 GTEA 上 train** (Harry confirmed: "A 对的, B 先用 4B 试试", 现在卡 dgxh-2 80GB).

### Transformers compat verification (重要)
- HPC env 当前 `transformers==4.49.0`, **里面没有 `Gemma3ForConditionalGeneration`** (4.49 早于 Gemma 3 release date).
- 测试 in-place upgrade 可行性: `pip install --target /tmp/tf451_zhanhaoc --no-deps transformers==4.51.3`, 再 `PYTHONPATH=/tmp/tf451_zhanhaoc python ...` import 测试.
- **结果**: 4.51.3 下 16/16 个 UniTime Qwen2VL wrapper imports 全 OK + Gemma3ForConditionalGeneration / Gemma3Config / Gemma3ForCausalLM / PaliGemma 全部 available + peft 0.14 + LoraConfig + UniTime arguments dataclass 全部 OK.
- → **决定**: 升级 `transformers==4.49.0 → 4.51.3` in place, **不**建 sibling env. 现有 Qwen2VL 训练流程不会被破坏.

### Gemma 3 architecture vs Qwen2VL (差异决定 port 复杂度)
- Vision input: Qwen2VL 有 `pixel_values` 和 `pixel_values_videos` 两条路径; **Gemma 3 只有 `pixel_values`**, video 当成 image sequence.
- Vision tower: Qwen2VL 用 `Qwen2VisionTransformerPretrainedModel` (动态 patch + spatial_merge); **Gemma 3 用 SigLIP-2 + Gemma3MultiModalProjector**, 固定 256 soft tokens per image.
- Position: Qwen2VL 用 mRoPE 3D (temporal+height+width 三轴); **Gemma 3 是标准 1D rotary**, 不需要 `get_rope_index_multiqa`.
- 子模块路径: Qwen2VL 是 `model.visual / model.model`; Gemma 3 是 `model.vision_tower / model.multi_modal_projector / model.language_model`. → `supported_models.py` 的 `MODULE_KEYWORDS` 要为 gemma3 写新的 path keys.
- Token 分隔: Qwen2VL 用 special token id (`<|video_pad|>`) 然后 `masked_scatter` 到 inputs_embeds; Gemma 3 用 `token_type_ids` + 固定的 image token id.
- forward 签名: Qwen2VL 13+ 参数 (含 grid_thw, rope_deltas, combine_t_list, multi_qa, attention_mask_multiqa, feature_inputs); Gemma 3 简洁很多 (没有 grid_thw 等), **要在 wrapper 里加回来**.

### Port plan (5 个新 file, ~800-1100 行)
1. `models/gemma3_vl.py`: `Gemma3VLMRForConditionalGeneration` 继承 `Gemma3ForConditionalGeneration`, override `forward` 加 timestamp interleave + multi-qa attention mask + feature_inputs path. 没有 mRoPE 所以 `get_rope_index_multiqa` 简化成 1D.
2. `collators/gemma_vision_process.py`: Gemma 版的 `process_vision_info`, fetch_video 出来的 feature shape 跟 Qwen 不同 (3D vs 4D), 要适配.
3. `collators/gemma3_vl.py`: `Gemma3DataCollator`, mr_seg multi-window 逻辑跟 Qwen 一致, 但 token vocab 换成 Gemma 3 的 (`<start_of_turn>` 替换 `<|im_start|>`).
4. `loaders/gemma3_vl.py`: `Gemma3ModelLoader`, return Gemma3VLMR + processor + tokenizer + config.
5. `extract_gemma_features.py`: GTEA 28 video → Gemma 3 vision encoder features, 输出格式跟 Qwen 版的 `{feature, frame_idx, sample_fps}` 对齐, 落到 `feature/Gemma3-4B/gtea/`.

外加: `supported_models.py` 注册 `gemma3-4b-it` model_id + MODULE_KEYWORDS path; `collators/__init__.py` + `loaders/__init__.py` import 新模块; 写新 `experiments/unitime/scripts/train_gemma3.sh`.

### Citations
- UniTime model-agnostic claim: `paper_notes/01_unitime.md:55` ("这一招对所有支持动态 frame 的 MLLM 都能 plug-and-play, Section 3.4 在 Qwen2-VL-2B/7B, Qwen2.5-VL-7B, InternVL2.5-2B/8B 上都验证"). **Gemma 3 不在验证列表里**, 我们是第一个试 Gemma 3 + UniTime 方法的, 是 untested territory.
- mr_seg multi-window 实现位置: `experiments/unitime/UniTime/collators/qwen2_vl.py:91-102` (`for t_w in windows: for t_w_i in t_w: ... interval_text = ", ".join([f"{s} seconds" ...])`).

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
