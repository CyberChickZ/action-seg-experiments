
import os
import json
import torch
import torch.nn.functional as F
import argparse
import decord
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Qwen-specific imports (Assumes these are in your local path)
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from collators.qwen_vision_process import (
    generate_clip_lengths, smart_resize, round_by_factor, 
    floor_by_factor, IMAGE_FACTOR, FRAME_FACTOR, 
    VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, FPS
)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Qwen2-VL features from a directory of videos.")
    parser.add_argument('--video_root', type=str, required=True, help='Path to the directory containing videos')
    parser.add_argument('--feat_root', type=str, default='./tmp_feature', help='Directory to save .pt features')
    parser.add_argument('--model_local_path', type=str, required=True, help='Path to the local Qwen2-VL model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='custom_dataset', help='Subfolder name for features')
    parser.add_argument('--part', default=0, type=int, help='Part index for parallel processing')
    parser.add_argument('--num_parts', default=1, type=int, help='Total number of parts for parallel processing')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use')
    return parser.parse_args()

def resize_feature(feature, resize_h, resize_w):
    """
    Resizes the model embeddings spatially using bilinear interpolation.
    """
    # feature shape: [T, H, W, C] -> permute to [T, C, H, W] for interpolate
    feature = feature.permute(0, 3, 1, 2)
    feature_resized = F.interpolate(feature, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
    # permute back to [T, H_new, W_new, C]
    feature_resized = feature_resized.permute(0, 2, 3, 1)
    return feature_resized

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    
    # Setup Output Directory
    feature_path = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(feature_path, exist_ok=True)

    # 1. Load Model and Processor
    print(f"Loading model from {args.model_local_path}...")
    compute_dtype = torch.bfloat16
    loading_kwargs = dict(
        torch_dtype=compute_dtype,
        quantization_config=None,
        device_map=None,
    )

    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        attn_implementation="flash_attention_2",
        **loading_kwargs,
    ).to(device).eval()
    
    processor = Qwen2VLMRProcessor.from_pretrained(args.model_local_path)

    # 2. Gather Video Files
    valid_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    all_video_files = sorted([
        f for f in os.listdir(args.video_root) 
        if f.lower().endswith(valid_extensions)
    ])
    
    # 3. Handle Parallel Splitting
    total_data = len(all_video_files)
    part_size = total_data // args.num_parts
    start_idx = args.part * part_size
    end_idx = (args.part + 1) * part_size if args.part != args.num_parts - 1 else total_data
    
    video_files_subset = all_video_files[start_idx:end_idx]
    print(f"Processing part {args.part}/{args.num_parts}: {len(video_files_subset)} videos.")

    # 4. Processing Loop
    with torch.no_grad():
        for filename in tqdm(video_files_subset):
            vid = os.path.splitext(filename)[0]
            video_path = os.path.join(args.video_root, filename)
            visual_feature_path = os.path.join(feature_path, f"{vid}.pt")

            # Skip if already exists and valid
            if os.path.exists(visual_feature_path):
                try:
                    # Optional: Add logic here to check if feature is corrupted
                    continue 
                except:
                    print(f"Feature {vid} corrupted, re-extracting...")

            # Load Video
            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue

            total_frames, video_fps = len(vr), vr.get_avg_fps()
            
            # Determine Frame Dimensions
            video_sample = vr.get_batch([0]).asnumpy()
            _, height, width, _ = video_sample.shape # [B, H, W, C]

            # Calculate Resizing Parameters (Qwen2-VL Logic)
            print(total_frames)
            nframes_2fps = round_by_factor(int(total_frames / video_fps * FPS), FRAME_FACTOR)
            video_total_pixels = 1024 * 16 * 28 * 28
            video_min_pixels = 32 * 28 * 28
            video_max_pixels = 768 * 28 * 28

            max_pixels = max(min(video_max_pixels, video_total_pixels / nframes_2fps * FRAME_FACTOR), int(video_min_pixels))

            resized_height, resized_width = smart_resize(
                height, width, factor=IMAGE_FACTOR, min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS
            )

            new_resized_height, new_resized_width = smart_resize(
                height, width, factor=IMAGE_FACTOR, min_pixels=video_min_pixels, max_pixels=max_pixels
            )

            # Sampling Strategy
            if max_pixels == video_min_pixels:
                nframes = video_total_pixels // max_pixels * FRAME_FACTOR
            else:
                nframes = video_total_pixels // (new_resized_height * new_resized_width) * FRAME_FACTOR

            nframes = floor_by_factor(nframes, FRAME_FACTOR)
            print(nframes)
            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps
            
            # Calculate indices for saved metadata
            fps_sample_feature_frame_idx = [int((x + y)/2) for x, y in zip(frame_idx[::2], frame_idx[1::2])]

            # Extract Frames
            try:
                frames = vr.get_batch(frame_idx).asnumpy()
            except:
                print(f"Failed to get frames for {vid}")
                continue

            # Pre-process frames (Physical Resize)
            video_tensor = torch.tensor(frames).permute(0, 3, 1, 2) # [T, C, H, W]
            print(video_tensor.shape)
            # exit()
            video_inputs = [transforms.functional.resize(
                video_tensor,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()]
            print(video_inputs[0].shape)
            exit()

            # Model Inference
            inputs = processor(
                text=['hello'],
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            pixel_values_videos = inputs['pixel_values_videos'].type(model.visual.get_dtype()).to(model.device)
            video_grid_thw = inputs['video_grid_thw']

            # Generate Embeddings
            combine_t_list = [generate_clip_lengths(len(frame_idx) // 2, 1)]
            video_embeds = model.encode_video_chunk(pixel_values_videos, video_grid_thw, combine_t_list).cpu()
            
            # Reshape and Secondary Resize (Logical/Feature Resize)
            # Qwen2-VL visual tokens are usually pooled/reduced by factor of 2 in T, H, and W
            video_embeds = video_embeds.reshape(
                len(combine_t_list[0]), 
                video_grid_thw[0][1] // 2, 
                video_grid_thw[0][2] // 2, 
                video_embeds.shape[-1]
            )
            
            video_embeds_resize = resize_feature(
                video_embeds, 
                resize_h=new_resized_height // 28, 
                resize_w=new_resized_width // 28
            )

            # Save Results
            torch.save({
                "feature": video_embeds_resize, 
                "frame_idx": torch.tensor(fps_sample_feature_frame_idx), 
                "sample_fps": sample_fps
            }, visual_feature_path)

if __name__ == "__main__":
    main()