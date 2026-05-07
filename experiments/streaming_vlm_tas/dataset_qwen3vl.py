"""GTEA sliding window dataset for Qwen3-VL TAS training.

7s window, 5fps, 35 frames per window.
GT from per-frame labels → merged segments → "start-ends action" text.
"""
import os
import random
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class GTEAWindowDataset(Dataset):

    ORIGINAL_FPS = 15
    SAMPLE_FPS = 5
    STEP = ORIGINAL_FPS // SAMPLE_FPS  # 3
    WINDOW_SEC = 7
    WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # 105
    N_SAMPLE_FRAMES = SAMPLE_FPS * WINDOW_SEC  # 35

    def __init__(self, video_folder: str, gt_folder: str, split_file: str, split: str = "train"):
        super().__init__()
        self.video_folder = video_folder
        self.gt_folder = gt_folder
        self.split = split

        video_ids = self._load_split(split_file)
        self.entries = []
        for vid in video_ids:
            gt_path = os.path.join(gt_folder, f"{vid}.txt")
            labels = self._load_gt(gt_path)
            if len(labels) < self.WINDOW_FRAMES:
                continue
            video_path = None
            for ext in (".mp4", ".avi"):
                p = os.path.join(video_folder, f"{vid}{ext}")
                if os.path.exists(p):
                    video_path = p
                    break
            if video_path is None:
                continue
            self.entries.append({
                "id": vid,
                "video_path": video_path,
                "labels": labels,
                "total_frames": len(labels),
            })

    def _load_split(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip().replace(".txt", "") for line in f if line.strip()]

    def _load_gt(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        labels = entry["labels"]
        total_frames = entry["total_frames"]

        max_s = total_frames - self.WINDOW_FRAMES
        s_frame = random.randint(0, max_s)

        sample_indices = [s_frame + i * self.STEP for i in range(self.N_SAMPLE_FRAMES)]
        sampled_labels = [labels[idx] for idx in sample_indices]

        # Merge consecutive same labels → segments
        segments = []
        current = sampled_labels[0]
        start_i = 0
        for i in range(1, self.N_SAMPLE_FRAMES):
            if sampled_labels[i] != current:
                if current != "background":
                    t_start = round(start_i * self.STEP / self.ORIGINAL_FPS, 1)
                    t_end = round((i - 1) * self.STEP / self.ORIGINAL_FPS, 1)
                    segments.append(f"{t_start}-{t_end}s {current}")
                current = sampled_labels[i]
                start_i = i
        if current != "background":
            t_start = round(start_i * self.STEP / self.ORIGINAL_FPS, 1)
            t_end = round((self.N_SAMPLE_FRAMES - 1) * self.STEP / self.ORIGINAL_FPS, 1)
            segments.append(f"{t_start}-{t_end}s {current}")

        gt_text = "\n".join(segments) if segments else "none"

        # Per-frame timestamps for interleaving
        frame_timestamps = [round(i * self.STEP / self.ORIGINAL_FPS, 1) for i in range(self.N_SAMPLE_FRAMES)]

        return {
            "id": entry["id"],
            "video_path": entry["video_path"],
            "sample_indices": sample_indices,
            "frame_timestamps": frame_timestamps,
            "gt_text": gt_text,
            "split": self.split,
        }
