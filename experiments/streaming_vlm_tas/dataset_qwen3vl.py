"""GTEA sliding window dataset for Qwen3-VL TAS training.

10s window, 5fps, 50 frames per window.
Each epoch: random start offset per video → deterministic sliding → all windows.
Windows are independent (no context from previous), so batch can be large.
"""
import os
import random
from typing import Dict, List
from torch.utils.data import Dataset


class GTEAWindowDataset(Dataset):

    ORIGINAL_FPS = 15
    SAMPLE_FPS = 5
    STEP = ORIGINAL_FPS // SAMPLE_FPS  # 3
    WINDOW_SEC = 10
    WINDOW_FRAMES = ORIGINAL_FPS * WINDOW_SEC  # 150
    N_SAMPLE_FRAMES = SAMPLE_FPS * WINDOW_SEC  # 50

    def __init__(self, video_folder: str, gt_folder: str, split_file: str, split: str = "train"):
        super().__init__()
        self.video_folder = video_folder
        self.gt_folder = gt_folder
        self.split = split

        video_ids = self._load_split(split_file)
        self.videos = []
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
            self.videos.append({
                "id": vid,
                "video_path": video_path,
                "labels": labels,
                "total_frames": len(labels),
            })

        self._build_windows()

    def _build_windows(self):
        """Build window list. Call at start of each epoch for new random offsets."""
        self.windows = []
        for v in self.videos:
            total = v["total_frames"]
            max_offset = min(self.WINDOW_FRAMES, total - self.WINDOW_FRAMES)
            start_offset = random.randint(0, max_offset) if max_offset > 0 else 0
            s = start_offset
            while s + self.WINDOW_FRAMES <= total:
                self.windows.append({"video_idx": len(self.windows) // 999999, "video": v, "s_frame": s})
                s += self.WINDOW_FRAMES  # no overlap
            # Keep reference to video_idx properly
        # Fix video_idx
        self.windows = []
        for vi, v in enumerate(self.videos):
            total = v["total_frames"]
            max_offset = min(self.WINDOW_FRAMES, total - self.WINDOW_FRAMES)
            start_offset = random.randint(0, max_offset) if max_offset > 0 else 0
            s = start_offset
            while s + self.WINDOW_FRAMES <= total:
                self.windows.append({"video_idx": vi, "s_frame": s})
                s += self.WINDOW_FRAMES

    def reshuffle(self):
        """Call at the start of each epoch to re-randomize window start offsets."""
        self._build_windows()

    def _load_split(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip().replace(".txt", "") for line in f if line.strip()]

    def _load_gt(self, path: str) -> List[str]:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        w = self.windows[idx]
        v = self.videos[w["video_idx"]]
        s_frame = w["s_frame"]
        labels = v["labels"]

        sample_indices = [s_frame + i * self.STEP for i in range(self.N_SAMPLE_FRAMES)]
        sampled_labels = [labels[si] for si in sample_indices]

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
        frame_timestamps = [round(i * self.STEP / self.ORIGINAL_FPS, 1) for i in range(self.N_SAMPLE_FRAMES)]

        return {
            "id": v["id"],
            "video_path": v["video_path"],
            "sample_indices": sample_indices,
            "frame_timestamps": frame_timestamps,
            "gt_text": gt_text,
            "split": self.split,
        }
