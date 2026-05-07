"""
Train Qwen3-VL-2B for TAS on GTEA.
UniTime-style timestamp-interleaved tokens + LoRA.

Run:
    unitime-gemma4
    cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/streaming_vlm_tas
    python train_qwen3vl_tas.py
"""
import os
import sys
import torch
import json
import time
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import decord

from dataset_qwen3vl import GTEAWindowDataset

# === Config ===
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
HF_CACHE = os.environ.get("HF_HOME", "/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache")
VIDEO_FOLDER = "/nfs/hpc/dgx2-4/data/TAS_videos/gtea"
GT_FOLDER = "/nfs/hpc/dgx2-4/data/gtea/groundTruth"
TRAIN_SPLIT = "/nfs/hpc/dgx2-4/data/gtea/splits/train.split1.bundle"
OUTPUT_DIR = "./checkpoints/qwen3vl_tas_7s_run1"
NUM_EPOCHS = 50
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
GRAD_ACCUM = 4
MAX_NEW_TOKENS_GT = 256

QUERY = "What actions are in this video? List each action with its start and end time."


def build_messages(pil_frames, timestamps, gt_text=None):
    content = []
    for i, (frame, ts) in enumerate(zip(pil_frames, timestamps)):
        content.append({"type": "text", "text": f"{ts}s"})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": QUERY})
    messages = [{"role": "user", "content": content}]
    if gt_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": gt_text}]})
    return messages


def load_frames(video_path, indices):
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=2)
    frames = []
    for idx in indices:
        if idx < len(vr):
            frames.append(Image.fromarray(vr[idx].asnumpy()))
    return frames


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    tokenizer = processor.tokenizer

    # LoRA
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = GTEAWindowDataset(VIDEO_FOLDER, GT_FOLDER, TRAIN_SPLIT, split="train")
    print(f"Train entries: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    total_steps = NUM_EPOCHS * len(dataset) // GRAD_ACCUM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    model.gradient_checkpointing_enable()

    log_path = os.path.join(OUTPUT_DIR, "train.log")
    start_time = time.time()

    step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        indices = list(range(len(dataset)))
        import random
        random.shuffle(indices)

        for i, di in enumerate(indices):
            sample = dataset[di]
            pil_frames = load_frames(sample["video_path"], sample["sample_indices"])
            if len(pil_frames) == 0:
                continue

            messages = build_messages(pil_frames, sample["frame_timestamps"], gt_text=sample["gt_text"])

            text_with_gt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text_no_gt = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

            inputs = processor(text=[text_with_gt], images=pil_frames, padding=True, return_tensors="pt").to(model.device)

            # Build labels: mask everything before assistant response
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            prompt_inputs = processor(text=[text_no_gt], images=pil_frames, padding=True, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[0, :prompt_len] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    avg = epoch_loss / (i + 1)
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    msg = f"epoch={epoch} step={step} loss={avg:.4f} lr={lr_now:.2e} time={elapsed:.0f}s"
                    print(msg)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")

        avg_loss = epoch_loss / len(indices)
        print(f"=== Epoch {epoch} done, avg_loss={avg_loss:.4f} ===")

        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    total_time = time.time() - start_time
    print(f"\nDone. Total time: {total_time:.0f}s")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
