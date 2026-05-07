"""
Zero-shot Qwen3-VL-2B on GTEA: 10s sliding window, 5fps, ask how many actions per window.
Hardcoded single video test.

Run on HPC:
    conda activate UniTime-gemma4  # or any env with transformers>=5.0
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
STEP = ORIGINAL_FPS // SAMPLE_FPS  # 3
WINDOW_SEC = 10
WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # 150
N_SAMPLE = SAMPLE_FPS * WINDOW_SEC  # 50 frames per window
OVERLAP_SEC = 5
OVERLAP_FRAMES = ORIGINAL_FPS * OVERLAP_SEC  # 75

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
                segs.append({
                    "action": cur,
                    "start": round((start - s_frame) / fps, 1),
                    "end": round((i - 1 - s_frame) / fps, 1),
                })
            cur = labels[i]
            start = i
    if cur != "background":
        segs.append({
            "action": cur,
            "start": round((start - s_frame) / fps, 1),
            "end": round((e_frame - 1 - s_frame) / fps, 1),
        })
    return segs


def extract_frames(video_path, indices):
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=2)
    valid = [i for i in indices if i < len(vr)]
    frames = vr.get_batch(valid)  # [N, H, W, 3]
    return frames.numpy()  # uint8 HWC


def main():
    print(f"Video: {VIDEO_PATH}")
    print(f"GT: {GT_PATH}")
    print(f"Window: {WINDOW_SEC}s, FPS: {SAMPLE_FPS}, Overlap: {OVERLAP_SEC}s")
    print()

    labels = load_gt(GT_PATH)
    total_frames = len(labels)
    total_sec = total_frames / ORIGINAL_FPS
    print(f"Total: {total_frames} frames = {total_sec:.1f}s")
    print()

    # Load model
    print(f"Loading {MODEL_ID}...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=HF_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    print("Model loaded.")
    print()

    # Sliding windows
    windows = []
    s = 0
    while s + WINDOW_FRAMES <= total_frames:
        windows.append(s)
        s += WINDOW_FRAMES - OVERLAP_FRAMES
    if s < total_frames and total_frames - s >= ORIGINAL_FPS * 2:  # at least 2s
        windows.append(s)

    print(f"Windows: {len(windows)}")
    print()

    action_list_str = ", ".join(ACTION_CLASSES)

    all_results = []
    for wi, s_frame in enumerate(windows):
        e_frame = min(s_frame + WINDOW_FRAMES, total_frames)
        t_start = round(s_frame / ORIGINAL_FPS, 1)
        t_end = round(e_frame / ORIGINAL_FPS, 1)

        # Sample frames
        indices = [s_frame + i * STEP for i in range(N_SAMPLE) if s_frame + i * STEP < e_frame]

        # GT for this window
        gt_segs = get_gt_segments(labels, s_frame, WINDOW_FRAMES, ORIGINAL_FPS)

        print(f"--- Window {wi}: {t_start}s - {t_end}s ({len(indices)} frames) ---")
        print(f"GT: {json.dumps(gt_segs)}")

        # Extract frames as PIL images
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(VIDEO_PATH, num_threads=2)
        from PIL import Image
        import numpy as np
        pil_frames = []
        for idx in indices:
            if idx < len(vr):
                frame = vr[idx].asnumpy()  # HWC uint8
                pil_frames.append(Image.fromarray(frame))

        # Build prompt with inline images
        image_content = [{"type": "image", "image": img} for img in pil_frames]

        prompt_text = (
            f"These are {len(pil_frames)} frames sampled at {SAMPLE_FPS}fps from a {WINDOW_SEC}-second cooking video clip.\n"
            f"The possible actions are: {action_list_str}.\n"
            f"List ALL action segments in this clip. For each segment, give the action name, "
            f"approximate start time (seconds from 0), and end time.\n"
            f"Output format: action start_time end_time (one per line).\n"
            f"Example:\ntake 0.0 2.0\nopen 2.0 4.5\n"
            f"If no action is visible, output: none"
        )

        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt_text}],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=pil_frames,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        print(f"Input tokens: {inputs['input_ids'].shape[1]}")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"Pred: {response}")
        print()

        all_results.append({
            "window": wi,
            "start_sec": t_start,
            "end_sec": t_end,
            "n_frames": len(indices),
            "n_input_tokens": inputs["input_ids"].shape[1],
            "gt": gt_segs,
            "pred": response,
        })

    # Save
    out_path = Path(__file__).parent / "zeroshot_results_S1_Cheese_C1.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== Summary ===")
    for r in all_results:
        gt_actions = [s["action"] for s in r["gt"]]
        print(f"Window {r['window']} ({r['start_sec']}-{r['end_sec']}s): GT={gt_actions}")
        print(f"  Pred: {r['pred'][:120]}")


if __name__ == "__main__":
    main()
