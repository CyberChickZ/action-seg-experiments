"""
Convert MS-TCN-style GTEA groundTruth CSVs to UniTime mr_seg JSON format.

Source: ported from Tieqiao Wang 2026/4/6 working dir
        (/nfs/hpc/dgx2-4/tmp/2026/4/6/gtea_csv_to_json.py)

Input:
  - csv_dir/<video_id>.csv with columns: start_time, end_time, label
  - splits/{train,test}.split{N}.bundle listing video_ids one per line

Output:
  - {save_dir}/{train,test}.json in UniTime mr_seg format:
      [{"qid": int, "id": str, "annos": [{"query": str, "window": [[s,e],...]}],
        "duration": float, "mode": "mr_seg"}]

Each (video, action_class) pair becomes ONE entry whose `window` field is the
list of all intervals where that action occurs. UniTime's collator
(`collators/qwen2_vl.py:96`) natively handles this multi-window mr_seg target.
"""
import argparse
import json
import os

import pandas as pd


def load_split_ids(split_path):
    """Read .bundle file, strip .txt extension, return list of video ids."""
    with open(split_path, "r") as f:
        return [line.strip().replace(".txt", "") for line in f if line.strip()]


def process_csvs(ids, csv_dir, start_qid=0):
    data = []
    qid = start_qid
    for video_id in ids:
        csv_path = os.path.join(csv_dir, f"{video_id}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        duration = float(df["end_time"].max())
        for label, group in df.groupby("label"):
            windows = group[["start_time", "end_time"]].values.tolist()
            data.append({
                "qid": qid,
                "id": video_id,
                "annos": [{"query": label, "window": windows}],
                "duration": duration,
                "mode": "mr_seg",
            })
            qid += 1
    return data, qid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True,
                   help="Directory of per-video <video_id>.csv files (MS-TCN groundTruth)")
    p.add_argument("--train_split", required=True,
                   help="Path to train.splitN.bundle")
    p.add_argument("--test_split", required=True,
                   help="Path to test.splitN.bundle")
    p.add_argument("--save_dir", required=True,
                   help="Output directory for train.json / test.json")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_ids = load_split_ids(args.train_split)
    test_ids = load_split_ids(args.test_split)

    print(f"Processing {len(train_ids)} train videos...")
    train_data, next_qid = process_csvs(train_ids, args.csv_dir, start_qid=0)
    print(f"Processing {len(test_ids)} test videos...")
    test_data, _ = process_csvs(test_ids, args.csv_dir, start_qid=next_qid)

    with open(os.path.join(args.save_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(args.save_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved {len(train_data)} train + {len(test_data)} test entries to {args.save_dir}")


if __name__ == "__main__":
    main()
