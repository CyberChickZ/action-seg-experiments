# UniTime → Gemma 4 E4B port — preparation plan

> Track B from Tieqiao's Discord ask (`paper_notes/01_unitime.md` line refs in research_journal): "I am not sure if we have official finetuning code for Muse Spark (not necessary use Unitime). also this one https://deepmind.google/models/gemma/ from google".
> Status: model weights downloaded ✅. Env + wrapper code 都未做.

---

## 0. TL;DR

| Item | Status |
|---|---|
| Gemma 4 E4B weights | ✅ downloaded to `/nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it/` (15 GB single safetensors) |
| `transformers >= 5.x` (which has Gemma 4) | ❌ requires `torch >= 2.4`, current UniTime env has 2.1.2 |
| New conda env `UniTime-gemma4` | ❌ TODO (recipe below) |
| `models/gemma4_vl.py` wrapper | ❌ TODO (port plan below) |
| `extract_gemma4_features.py` | ❌ TODO |
| `train_gemma4.sh` | ❌ TODO |

**Why a separate env**: transformers 5.x (which contains `Gemma4ForConditionalGeneration`) requires torch >= 2.4. The current UniTime env has torch 2.1.2 + flash-attn 2.7.2 compiled against torch 2.1.x, which we cannot upgrade in-place without breaking the running Qwen2-VL + Gemma 3 paths. **Cleanest fix is a parallel env**, name `UniTime-gemma4`.

---

## 1. Architecture differences vs Gemma 3 (informs wrapper port)

Read directly from `transformers/models/gemma4/modeling_gemma4.py` (transformers 5.5.0, on HPC at `/tmp/tf550_gemma4_test/`) and from the downloaded `Gemma4-E4B-it/config.json`:

| Aspect | Gemma 3 (4B) | **Gemma 4 (E4B)** | Impact on port |
|---|---|---|---|
| Class hierarchy | `Gemma3ForConditionalGeneration` has `vision_tower` + `multi_modal_projector` + `language_model` directly | `Gemma4ForConditionalGeneration` only has `self.model = Gemma4Model(config)` + `self.lm_head`. **All multimodal merging is inside `Gemma4Model`** | Wrapper port is HARDER — we either override `Gemma4Model.forward` (deeper) or do feature injection via `inputs_embeds` and skip pixel_values entirely |
| Vision tower | SigLIP-2 (frozen pretrained) | Custom **`gemma4_vision`** (jointly trained with text) | Different feature dim; cannot reuse Gemma 3 cached features |
| `text_config.hidden_size` | 2560 | **2560** (same!) | Conveniently the same — pre-extracted features have the same dim, but different vision spaces, still must re-extract |
| `text_config.num_hidden_layers` | 34 | **42** | Bigger LLM, slower per-step |
| `text_config.sliding_window` | 1024 | **512** | **More aggressive sliding window — 32 frames × 256 tokens = 8192 image tokens, sliding window of 512 means a single image's tokens only see neighbors within 512. timestamp interleave attention pattern will be heavily window-bounded, may hurt long-sequence grounding** |
| `mm_tokens_per_image` | 256 (fixed) | **variable** (config has `vision_soft_tokens_per_image`, blog says 70/140/280/560/1120 budgets) | Wrapper / collator sanity check needs to be parameterized, not hardcoded 256 |
| Modalities | image + text | image + **video** + text + **audio** | Has `pixel_values_videos` separate from `pixel_values`; explicit `video_token_id`. We can use the video path (more natural for our task) |
| `image_token_id` | 262144 | **258880** | Different — wrapper must read from config |
| `video_token_id` | n/a | **present** | NEW: native video token, may simplify timestamp-interleave |
| `audio_token_id` | n/a | 258881 | irrelevant for our task, ignore |
| `forward()` signature | clean | adds `pixel_values_videos`, `input_features` (audio), `image_position_ids`, `video_position_ids`, `mm_token_type_ids` | Wrapper signature must include or kwarg-passthrough these |
| Attention mask construction | `_update_causal_mask` with `token_type_ids` | `create_masks_for_generate` static method, dispatches on `use_bidirectional_attention` config flag | Different multi-qa mask plumbing path |

### 1.1 What this means for our wrapper port

**Option A** (recommended, simpler): write `Gemma4VLMRForConditionalGeneration` that **overrides `forward` and uses the `inputs_embeds` path to bypass `pixel_values_videos` entirely**:

```python
class Gemma4VLMRForConditionalGeneration(Gemma4ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, ...,
                feature_inputs=None,  # cached features [N, mm_tokens, hidden]
                ...):
        if feature_inputs is not None:
            # Build inputs_embeds with features pre-merged
            inputs_embeds = self.get_input_embeddings()(input_ids)
            image_token_id = self.config.image_token_id
            # OR: video_token_id depending on how we tokenize the prompt
            mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                mask, feature_inputs.to(inputs_embeds.dtype)
            )
            # Then call super().forward with input_ids=None and inputs_embeds
            return super().forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                ...
                pixel_values=None,
                pixel_values_videos=None,
                input_features=None,
            )
        # else: pass-through
        return super().forward(input_ids=input_ids, ..., pixel_values=pixel_values, ...)
```

This is structurally identical to the Gemma 3 wrapper, just one extra layer of indirection. Should be ~150 LOC.

**Option B** (deeper): override `Gemma4Model.forward` to inject features inside the multimodal merging step. Cleaner but requires more reading of upstream and more risk of API drift.

→ Go with **Option A**.

### 1.2 Tokenization gotcha

Gemma 4 has **two** image-region special tokens: `image_token_id=258880` and `video_token_id` (for native video). Our prompt-build code in the collator should pick **one** convention and stick with it. Recommendation: **use `image_token_id`** path (treat each frame as an image), because:
1. Video processor in transformers might not let us pre-supply features without going through `pixel_values_videos`
2. Image path is closer to what we did for Gemma 3 → less new code
3. Loses the benefit of native video tokens, but that's fine for an initial port

---

## 2. Conda env recipe (`UniTime-gemma4`)

Run on HPC (any node, doesn't need GPU):

```bash
# 1. create env
conda create -n UniTime-gemma4 python=3.10 -y
conda activate UniTime-gemma4

# 2. PyTorch 2.4 + CUDA 12.1 (matches dgxh nvcc 12.1; cu13 driver is backward-compatible)
pip install 'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1' \
    --index-url https://download.pytorch.org/whl/cu121

# 3. transformers 5.x (which has Gemma 4)
# Note: transformers 5.x pulls in httpx, hf_hub >= 0.30, etc. Don't --no-deps here.
pip install 'transformers==5.5.0'

# 4. UniTime training stack (matching upstream pins as closely as possible)
pip install 'accelerate==1.10.0' 'peft==0.18.0' 'deepspeed==0.16.4' \
            'decord==0.6.0' 'numpy==1.26.4' 'pandas==2.2.3' \
            'tensorboard==2.18.0' 'nncore==0.4.5' 'einops==0.8.2'

# 5. Skip flash-attn for now — fall back to sdpa.
# flash-attn for torch 2.4 cu121 wheel is at:
#   https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# But it likely lacks Gemma4 op-builders → segfault, same as Gemma 3 case.
# sdpa is built into PyTorch and works for any architecture. For training on
# 32 frames × 256-1120 image tokens, sdpa on H100 is ~30% slower than flash-attn
# but won't crash. We can revisit after the model is converging.

# 6. Verify
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import transformers
print('transformers', transformers.__version__)
from transformers import Gemma4ForConditionalGeneration, Gemma4Config
print('Gemma4 imports OK')
import peft, accelerate, decord, deepspeed
print('peft', peft.__version__, 'accelerate', accelerate.__version__,
      'decord', decord.__version__, 'deepspeed', deepspeed.__version__)
"
```

Expected output:
```
torch 2.4.1+cu121 cuda 12.1
transformers 5.5.0
Gemma4 imports OK
peft 0.18.0 accelerate 1.10.0 decord 0.6.0 deepspeed 0.16.4
```

---

## 3. Files to write (port plan)

All under `experiments/unitime/UniTime/` (will be a new branch on `CyberChickZ/UniTime` since the changes are env-incompatible with the current main):

| File | Strategy | Estimated LOC |
|---|---|---|
| `models/gemma4_vl.py` | Port from `models/gemma3_vl.py`, swap class names + adapt to `Gemma4Model` indirection (option A above) | ~200 |
| `loaders/gemma4_vl.py` | Direct port from `loaders/gemma3_vl.py`, register `"gemma4"` family | ~30 |
| `collators/gemma4_vision_process.py` | Same as `gemma_vision_process.py`, no changes (just file rename) | ~100 |
| `collators/gemma4_vl.py` | Port from `collators/gemma3_vl.py`, swap special token strings (`<image_soft_token>` → Gemma 4's image token, image_token_id from config) and read `mm_tokens_per_image` from config (variable) | ~220 |
| `extract_gemma4_features.py` | Port from `extract_gemma_features.py`, swap base model class. Note: Gemma 4 vision is **trainable**, so we'd be extracting features from random init unless we use the released checkpoint. Use the released `Gemma4-E4B-it` weights | ~150 |
| `experiments/unitime/scripts/train_gemma4.sh` | Port from `train_gemma3.sh`, swap model_id + paths + use `UniTime-gemma4` env | ~70 |
| `supported_models.py` (modify) | Add `"gemma4": {...}` to MODULE_KEYWORDS, register `gemma4-e4b-it` | +20 |

Total new code: **~800 LOC**, mostly mechanical port from Gemma 3 versions.

### 3.1 Submodule branching strategy

The Gemma 4 code requires `transformers >= 5.0`, which is incompatible with the current Qwen2VL + Gemma 3 paths in the same env. To keep both working:

**Option A** (recommended): create a `gemma4` branch on `CyberChickZ/UniTime`. Gemma 4 code lives there. Parent repo keeps the submodule on `main` (Gemma 3 ready) but documents how to switch to `gemma4` branch when running Gemma 4 training.

**Option B**: keep all code on `main`. Add `try/except ImportError` guards around Gemma 4 imports so the file doesn't break Qwen2VL/Gemma 3 imports under the older transformers. Single branch, more conditional logic.

→ Recommend **Option A**. Cleaner separation.

---

## 4. Open questions before starting Phase 1 of Gemma 4

1. **Variable mm_tokens_per_image**: which budget do we use? 70 (fastest) / 256 (matches Gemma 3) / 1120 (max quality)? Recommendation: start with **256** to match Gemma 3 for fair comparison.
2. **Vision tower trainable in upstream?** Gemma 4 vision is jointly trained, so the released weights have a tuned vision tower. We freeze it for our finetuning anyway (vision_encoder=False, same as Gemma 3), so this should be fine.
3. **Sliding window 512**: with 32 frames × 256 image tokens = 8192 image tokens + text overhead → ~8500 sequence length. Sliding window 512 means timestamp text token at position N can only attend to image tokens within ±512 of N. **Interleave pattern means each timestamp only sees ~2 frames of image tokens worth of context.** This might be a real problem for multi-window grounding. Mitigation: reduce frame count to 16 (4096 image tokens) so the whole sequence fits inside one sliding window pass.
4. **`run_260408` ACL still pending** — task 1 record source for Qwen2VL UniTime baseline.

---

## 5. Action items (in order)

1. [ ] **Decide**: branch strategy (Option A: gemma4 branch / Option B: single branch with guards). Default A.
2. [ ] **Build env**: run section 2 commands, ~10 min.
3. [ ] **Verify**: section 2 verify block.
4. [ ] **Read** more of `transformers/models/gemma4/modeling_gemma4.py:Gemma4Model.forward` to understand how multimodal merging happens (whether `inputs_embeds` path actually skips vision call).
5. [ ] **Write** files in the order from §3.
6. [ ] **Sanity check** — single forward pass with a fake (16 frame, 256 token) feature tensor before launching training.
7. [ ] **Train** + record results (mirror GEMMA3_PORT.md format).

---

## 6. Citations
- Gemma 4 release: https://huggingface.co/blog/gemma4 (April 2, 2026, Apache 2.0)
- Model files: `/nfs/hpc/share/zhanhaoc/MODLE/Gemma4-E4B-it/` (downloaded 2026-04-08)
- transformers 5.5.0 modeling source: `/tmp/tf550_gemma4_test/transformers/models/gemma4/modeling_gemma4.py` (read 2026-04-08)
- Gemma 3 port reference: [`GEMMA3_PORT.md`](./GEMMA3_PORT.md)
