"""
Zero-shot Qwen3-VL-2B on GTEA: 10s sliding window, 5fps.
Two prompts per window:
  A) No labels: "How many distinct actions? Just a number."
  B) With labels: "Actions: take,open,... How many do you see? Just a number."

Run on HPC:
    unitime-gemma4
    cd /nfs/hpc/share/zhanhaoc/action-seg-experiments/experiments/streaming_vlm_tas
    python zeroshot_qwen3vl_gtea.py
"""
import os
import torch
import json
from pathlib import Path

VIDEO_PATH = "/nfs/hpc/dgx2-4/data/TAS_videos/gtea/S1_Cheese_C1.mp4"
GT_PATH = "/nfs/hpc/dgx2-4/data/gtea/groundTruth/S1_Cheese_C1.txt"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
HF_CACHE = os.environ.get("HF_HOME", "/nfs/hpc/share/zhanhaoc/hpe/dgx2-2/huggingface_cache")

ORIGINAL_FPS = 15
SAMPLE_FPS = 5
STEP = ORIGINAL_FPS // SAMPLE_FPS
WINDOW_SEC = 10
WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC
N_SAMPLE = SAMPLE_FPS * WINDOW_SEC
OVERLAP_SEC = 5
OVERLAP_FRAMES = ORIGINAL_FPS * OVERLAP_SEC

ACTION_CLASSES = ["take", "open", "pour", "close", "shake", "scoop", "stir", "put", "fold", "spread"]

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = HF_CACHE


def load_gt(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_gt_segments(labels, s_frame, window_frames, fps):
    e_frame = min(s_frame + window_frames, len(labels))
    segs = []
    cur = labels[s_frame]
    start = s_frame
    for i in range(s_frame + 1, e_frame):
        if labels[i] != cur:
            if cur != "background":
                segs.append({"action": cur, "start": round((start - s_frame) / fps, 1), "end": round((i - 1 - s_frame) / fps, 1)})
            cur = labels[i]
            start = i
    if cur != "background":
        segs.append({"action": cur, "start": round((start - s_frame) / fps, 1), "end": round((e_frame - 1 - s_frame) / fps, 1)})
    return segs


def ask(model, processor, pil_frames, prompt_text):
    image_content = [{"type": "image", "image": img} for img in pil_frames]
    messages = [{"role": "user", "content": image_content + [{"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_frames, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, temperature=None, top_p=None)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    print(f"Video: {VIDEO_PATH}")
    print(f"Window: {WINDOW_SEC}s, FPS: {SAMPLE_FPS}, Overlap: {OVERLAP_SEC}s")

    labels = load_gt(GT_PATH)
    total_frames = len(labels)
    print(f"Total: {total_frames} frames = {total_frames / ORIGINAL_FPS:.1f}s\n")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading {MODEL_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=HF_CACHE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    print("Model loaded.\n")

    windows = []
    s = 0
    while s + WINDOW_FRAMES <= total_frames:
        windows.append(s)
        s += WINDOW_FRAMES - OVERLAP_FRAMES
    if s < total_frames and total_frames - s >= ORIGINAL_FPS * 2:
        windows.append(s)

    action_list_str = ", ".join(ACTION_CLASSES)

    prompt_a = (
        "These are consecutive frames from a 10-second cooking video. "
        "How many distinct actions do you see? Just give me the number."
    )
    prompt_b = (
        "These are consecutive frames from a 10-second cooking video. "
        f"The possible action labels are: {action_list_str}. "
        "How many of these actions do you see in this clip? Just give me the number."
    )

    print(f"{'Win':>3} | {'Time':>12} | {'GT#':>3} | {'GT actions':<40} | {'A(no label)':>11} | {'B(w/ label)':>11}")
    print("-" * 100)

    import decord
    decord.bridge.set_bridge("native")
    from PIL import Image

    all_results = []
    for wi, s_frame in enumerate(windows):
        e_frame = min(s_frame + WINDOW_FRAMES, total_frames)
        t_start = round(s_frame / ORIGINAL_FPS, 1)
        t_end = round(e_frame / ORIGINAL_FPS, 1)
        indices = [s_frame + i * STEP for i in range(N_SAMPLE) if s_frame + i * STEP < e_frame]
        gt_segs = get_gt_segments(labels, s_frame, WINDOW_FRAMES, ORIGINAL_FPS)
        gt_count = len(gt_segs)
        gt_actions = [s["action"] for s in gt_segs]

        vr = decord.VideoReader(VIDEO_PATH, num_threads=2)
        pil_frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices if idx < len(vr)]

        resp_a = ask(model, processor, pil_frames, prompt_a)
        resp_b = ask(model, processor, pil_frames, prompt_b)

        print(f"{wi:>3} | {t_start:>5.1f}-{t_end:>5.1f}s | {gt_count:>3} | {str(gt_actions):<40} | {resp_a:>11} | {resp_b:>11}")

        all_results.append({
            "window": wi, "start": t_start, "end": t_end,
            "gt_count": gt_count, "gt_actions": gt_actions,
            "pred_no_label": resp_a, "pred_with_label": resp_b,
        })

    out_path = Path(__file__).parent / "zeroshot_count_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
