"""
Train Qwen3-VL-2B for TAS on GTEA.
UniTime-style timestamp-interleaved tokens + LoRA.
50 frames/window (5fps × 10s), no overlap, random start per epoch.
Windows are independent → large batch via grad_accum.

Run:
    unitime-gemma4
    cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/streaming_vlm_tas
    python train_qwen3vl_tas.py
"""
import os
import torch
import time
import random
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import decord

from dataset_qwen3vl import GTEAWindowDataset

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
HF_CACHE = os.environ.get("HF_HOME", "/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache")
VIDEO_FOLDER = "/nfs/hpc/dgx2-4/data/TAS_videos/gtea"
GT_FOLDER = "/nfs/hpc/dgx2-4/data/gtea/groundTruth"
TRAIN_SPLIT = "/nfs/hpc/dgx2-4/data/gtea/splits/train.split1.bundle"
OUTPUT_DIR = "./checkpoints/qwen3vl_tas_10s_run1"
NUM_EPOCHS = 50
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 8
QUERY = "What actions are in this video? List each action with its start and end time."


def build_messages(pil_frames, timestamps, gt_text=None):
    content = []
    for frame, ts in zip(pil_frames, timestamps):
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
    return [Image.fromarray(vr[idx].asnumpy()) for idx in indices if idx < len(vr)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = GTEAWindowDataset(VIDEO_FOLDER, GT_FOLDER, TRAIN_SPLIT, split="train")
    print(f"Videos: {len(dataset.videos)}, Windows/epoch: ~{len(dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    steps_per_epoch = max(1, len(dataset) // BATCH_SIZE)
    total_steps = NUM_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    model.gradient_checkpointing_enable()

    log_path = os.path.join(OUTPUT_DIR, "train.log")
    start_time = time.time()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        dataset.reshuffle()
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_samples = 0
        optimizer.zero_grad()

        for bi in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[bi:bi + BATCH_SIZE]
            batch_loss = 0.0

            for di in batch_indices:
                sample = dataset[di]
                pil_frames = load_frames(sample["video_path"], sample["sample_indices"])
                if len(pil_frames) == 0:
                    continue

                messages = build_messages(pil_frames, sample["frame_timestamps"], gt_text=sample["gt_text"])
                text_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                text_prompt = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

                inputs = processor(text=[text_full], images=pil_frames, padding=True, return_tensors="pt").to(model.device)

                labels = inputs["input_ids"].clone()
                prompt_inputs = processor(text=[text_prompt], images=pil_frames, padding=True, return_tensors="pt")
                prompt_len = prompt_inputs["input_ids"].shape[1]
                labels[0, :prompt_len] = -100

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / len(batch_indices)
                loss.backward()

                batch_loss += outputs.loss.item()
                n_samples += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_loss += batch_loss

            if global_step % 5 == 0:
                avg = epoch_loss / max(n_samples, 1)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                msg = f"ep={epoch} step={global_step} loss={avg:.4f} lr={lr_now:.2e} mem={gpu_mem:.1f}GB t={elapsed:.0f}s"
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

        avg_loss = epoch_loss / max(n_samples, 1)
        msg = f"=== Epoch {epoch} done, avg_loss={avg_loss:.4f}, windows={n_samples} ==="
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"Saved: {ckpt_dir}")

    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nDone. Time: {total_time:.0f}s, Peak GPU: {peak_mem:.1f}GB")


if __name__ == "__main__":
    main()
