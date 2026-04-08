"""
Extract Qwen2-VL visual features for a directory of videos, save as
UniTime-compatible .pt files.

Source: ported from Tieqiao Wang 2026/4/6/extract_qwen_embeddings.py with
debug `exit()` calls removed and `print()`s removed for clean runs.

Output per video (`{feat_root}/{dataset_name}/{vid}.pt`):
    {
        "feature":     bf16 tensor [T, H_patch//2, W_patch//2, hidden_dim]
        "frame_idx":   int64 tensor [T]   (frame indices in original video)
        "sample_fps":  float              (effective sampling fps)
    }

UniTime's collator (`collators/qwen_vision_process.py`) reads these .pt files
when `feat_folder` is supplied, bypassing on-the-fly Qwen vision encoding.
"""
import argparse
import os

import decord
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# UniTime modules — paths must be importable (run from UniTime/ root or PYTHONPATH set)
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from collators.qwen_vision_process import (
    FPS,
    FRAME_FACTOR,
    IMAGE_FACTOR,
    VIDEO_MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    floor_by_factor,
    generate_clip_lengths,
    round_by_factor,
    smart_resize,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--feat_root", type=str, required=True)
    ap.add_argument("--model_local_path", type=str, required=True,
                    help="Path to Qwen2-VL-7B-Instruct directory")
    ap.add_argument("--dataset_name", type=str, default="gtea",
                    help="Subdir under feat_root (e.g., 'gtea')")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--num_parts", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    return ap.parse_args()


def resize_feature(feature, resize_h, resize_w):
    """Bilinear-interpolate spatial dims of [T,H,W,C] feature."""
    feature = feature.permute(0, 3, 1, 2)
    feature = F.interpolate(feature, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    return feature.permute(0, 2, 3, 1)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    feature_path = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(feature_path, exist_ok=True)

    print(f"Loading Qwen2-VL from {args.model_local_path}...")
    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = Qwen2VLMRProcessor.from_pretrained(args.model_local_path)

    valid_ext = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    all_videos = sorted(f for f in os.listdir(args.video_root) if f.lower().endswith(valid_ext))

    total = len(all_videos)
    part_size = total // args.num_parts
    s = args.part * part_size
    e = (args.part + 1) * part_size if args.part != args.num_parts - 1 else total
    subset = all_videos[s:e]
    print(f"part {args.part}/{args.num_parts}: processing {len(subset)} videos")

    with torch.no_grad():
        for filename in tqdm(subset):
            vid = os.path.splitext(filename)[0]
            video_path = os.path.join(args.video_root, filename)
            out_path = os.path.join(feature_path, f"{vid}.pt")
            if os.path.exists(out_path):
                continue

            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except Exception as ex:
                print(f"decord open failed for {video_path}: {ex}")
                continue

            total_frames, video_fps = len(vr), vr.get_avg_fps()
            sample0 = vr.get_batch([0]).asnumpy()
            _, height, width, _ = sample0.shape

            nframes_2fps = round_by_factor(int(total_frames / video_fps * FPS), FRAME_FACTOR)
            video_total_pixels = 1024 * 16 * 28 * 28
            video_min_pixels = 32 * 28 * 28
            video_max_pixels = 768 * 28 * 28

            max_pixels = max(
                min(video_max_pixels, video_total_pixels / nframes_2fps * FRAME_FACTOR),
                int(video_min_pixels),
            )

            resized_height, resized_width = smart_resize(
                height, width, factor=IMAGE_FACTOR,
                min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS,
            )
            new_resized_height, new_resized_width = smart_resize(
                height, width, factor=IMAGE_FACTOR,
                min_pixels=video_min_pixels, max_pixels=max_pixels,
            )

            if max_pixels == video_min_pixels:
                nframes = video_total_pixels // max_pixels * FRAME_FACTOR
            else:
                nframes = video_total_pixels // (new_resized_height * new_resized_width) * FRAME_FACTOR
            nframes = floor_by_factor(nframes, FRAME_FACTOR)

            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps
            fps_sample_feature_frame_idx = [
                int((x + y) / 2) for x, y in zip(frame_idx[::2], frame_idx[1::2])
            ]

            try:
                frames = vr.get_batch(frame_idx).asnumpy()
            except Exception as ex:
                print(f"decord read failed for {vid}: {ex}")
                continue

            video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)
            video_inputs = [
                transforms.functional.resize(
                    video_tensor,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
            ]

            inputs = processor(
                text=["hello"],
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            pixel_values_videos = inputs["pixel_values_videos"].type(model.visual.get_dtype()).to(model.device)
            video_grid_thw = inputs["video_grid_thw"]

            combine_t_list = [generate_clip_lengths(len(frame_idx) // 2, 1)]
            video_embeds = model.encode_video_chunk(
                pixel_values_videos, video_grid_thw, combine_t_list
            ).cpu()

            video_embeds = video_embeds.reshape(
                len(combine_t_list[0]),
                video_grid_thw[0][1] // 2,
                video_grid_thw[0][2] // 2,
                video_embeds.shape[-1],
            )
            video_embeds = resize_feature(
                video_embeds,
                resize_h=new_resized_height // 28,
                resize_w=new_resized_width // 28,
            )

            torch.save(
                {
                    "feature": video_embeds,
                    "frame_idx": torch.tensor(fps_sample_feature_frame_idx),
                    "sample_fps": sample_fps,
                },
                out_path,
            )


if __name__ == "__main__":
    main()
