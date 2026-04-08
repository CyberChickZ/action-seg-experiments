
import os
import json
import torch
import torch.nn.functional as F
import argparse
import decord
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Qwen-specific imports
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from collators.qwen_vision_process import (
    generate_clip_lengths, round_by_factor, 
    floor_by_factor, FRAME_FACTOR, FPS
)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Qwen2-VL features at fixed 224x224.")
    parser.add_argument('--video_root', type=str, required=True)
    parser.add_argument('--feat_root', type=str, default='./tmp_feature')
    parser.add_argument('--model_local_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default='custom_dataset')
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--num_parts', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args()

def resize_feature(feature, resize_h, resize_w):
    feature = feature.permute(0, 3, 1, 2)
    feature_resized = F.interpolate(feature, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
    feature_resized = feature_resized.permute(0, 2, 3, 1)
    return feature_resized

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    
    # Fixed Config
    FIXED_H, FIXED_W = 224, 224
    # Qwen2-VL factor is 28. 224/28 = 8 patches. 
    # After model pooling (factor 2), spatial dim becomes 8/2 = 4.
    FEATURE_H, FEATURE_W = FIXED_H // 28 // 2, FIXED_W // 28 // 2 
    
    feature_path = os.path.join(args.feat_root, args.dataset_name)
    os.makedirs(feature_path, exist_ok=True)

    print(f"Loading model...")
    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        args.model_local_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    
    processor = Qwen2VLMRProcessor.from_pretrained(args.model_local_path)

    valid_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    all_video_files = sorted([f for f in os.listdir(args.video_root) if f.lower().endswith(valid_extensions)])
    
    # Parallel Splitting
    total_data = len(all_video_files)
    part_size = total_data // args.num_parts
    start_idx = args.part * part_size
    end_idx = (args.part + 1) * part_size if args.part != args.num_parts - 1 else total_data
    video_files_subset = all_video_files[start_idx:end_idx]

    with torch.no_grad():
        for filename in tqdm(video_files_subset):
            vid = os.path.splitext(filename)[0]
            video_path = os.path.join(args.video_root, filename)
            visual_feature_path = os.path.join(feature_path, f"{vid}.pt")

            if os.path.exists(visual_feature_path): continue

            try:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            except: continue

            total_frames, video_fps = len(vr), vr.get_avg_fps()
            
            # --- Updated Sampling Logic for Fixed Res ---
            # Using a standard pixel budget (approx 16k tokens)
            video_total_pixels = 1024 * 16 * 28 * 28 
            nframes = video_total_pixels // (FIXED_H * FIXED_W) * FRAME_FACTOR
            nframes = floor_by_factor(nframes, FRAME_FACTOR)
            
            frame_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            sample_fps = nframes / total_frames * video_fps
            fps_sample_feature_frame_idx = [int((x + y)/2) for x, y in zip(frame_idx[::2], frame_idx[1::2])]

            try:
                frames = vr.get_batch(frame_idx).asnumpy()
            except: continue

            # Physical Resize to 224x224
            video_tensor = torch.tensor(frames).permute(0, 3, 1, 2) # [T, C, H, W]
            print(video_tensor.shape)
            video_inputs = [transforms.functional.resize(
                video_tensor,
                [FIXED_H, FIXED_W],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()]
            print(video_inputs[0].shape)


            inputs = processor(
                text=['hello'],
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            pixel_values_videos = inputs['pixel_values_videos'].type(model.visual.get_dtype()).to(model.device)
            print(pixel_values_videos.shape)
            # exit()
            video_grid_thw = inputs['video_grid_thw']

            # Generate Embeddings
            combine_t_list = [generate_clip_lengths(len(frame_idx) // 2, 1)]
            video_embeds = model.encode_video_chunk(pixel_values_videos, video_grid_thw, combine_t_list).cpu()
            
            # DYNAMIC RESHAPE LOGIC
            # video_grid_thw[0] contains [T, H, W] of the patches
            target_t = video_grid_thw[0][0] // 1  # Time is already halved by the processor usually
            target_h = video_grid_thw[0][1] // 2  # Spatial pooling factor
            target_w = video_grid_thw[0][2] // 2  # Spatial pooling factor
            
            # Calculate how many tokens we actually have
            num_tokens = video_embeds.shape[0]
            embedding_dim = video_embeds.shape[-1]
            
            # Reshape based on what the model actually produced
            # We use -1 for the temporal dimension to absorb any discrepancies
            video_embeds = video_embeds.reshape(-1, target_h, target_w, embedding_dim)
            
            print(f"Final Feature Shape: {video_embeds.shape}") 
            # This should now print [256, 8, 8, 3584] or [256, 4, 4, 3584] correctly
            
            # We skip the second resize_feature because we are already at the target 224 fixed res
            print(video_embeds.shape)
            print(len(fps_sample_feature_frame_idx))
            print(sample_fps)
            # exit()
            torch.save({
                "feature": video_embeds, 
                "frame_idx": torch.tensor(fps_sample_feature_frame_idx), 
                "sample_fps": sample_fps
            }, visual_feature_path)
            # exit()

if __name__ == "__main__":
    main()