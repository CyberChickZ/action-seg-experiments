
import pandas as pd
import json
import os
import argparse

def load_split_ids(split_path):
    """Reads the IDs from the bundle file and converts .txt to .csv."""
    with open(split_path, 'r') as f:
        # Strip whitespace and replace extension
        ids = [line.strip().replace('.txt', '') for line in f if line.strip()]
    return ids

def process_csvs(ids, csv_dir, start_qid=0):
    """Processes a list of IDs into the required JSON format."""
    data = []
    qid_counter = start_qid

    for video_id in ids:
        csv_path = os.path.join(csv_dir, f"{video_id}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        
        # Determine total duration (assuming end_time of last row is max duration)
        duration = float(df['end_time'].max())

        # Group by label to collect all windows for the same query
        # This creates a dictionary: { 'label': [[start1, end1], [start2, end2]] }
        grouped = df.groupby('label')

        for label, group in grouped:
            windows = group[['start_time', 'end_time']].values.tolist()
            
            entry = {
                "qid": qid_counter,
                "id": video_id,
                "annos": [
                    {
                        "query": label,
                        "window": windows
                    }
                ],
                "duration": duration,
                "mode": "mr_seg"
            }
            data.append(entry)
            qid_counter += 1

    return data, qid_counter

def main():
    parser = argparse.ArgumentParser(description="Convert GTEA CSVs to JSON annotations.")
    
    # Path Arguments
    parser.add_argument("--csv_dir", type=str, 
                        default="/nfs/stak/users/wangtie/2026/1/18/ms-tcn/data/gtea/groundTruth1",
                        help="Directory containing the CSV files.")
    parser.add_argument("--train_split", type=str, 
                        default="/nfs/stak/users/wangtie/2026/1/18/ms-tcn/data/gtea/splits/train.split1.bundle",
                        help="Path to training split bundle.")
    parser.add_argument("--test_split", type=str, 
                        default="/nfs/stak/users/wangtie/2026/1/18/ms-tcn/data/gtea/splits/test.split1.bundle",
                        help="Path to testing split bundle.")
    parser.add_argument("--save_dir", type=str, 
                        default="/nfs/stak/users/wangtie/2026/4/6/data/gtea/annot",
                        help="Directory to save the generated JSONs.")

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load IDs
    train_ids = load_split_ids(args.train_split)
    test_ids = load_split_ids(args.test_split)

    # 2. Process Train Data
    print(f"Processing {len(train_ids)} training files...")
    train_data, next_qid = process_csvs(train_ids, args.csv_dir, start_qid=0)
    
    # 3. Process Test Data (continuing qid counter)
    print(f"Processing {len(test_ids)} testing files...")
    test_data, _ = process_csvs(test_ids, args.csv_dir, start_qid=next_qid)

    # 4. Save JSONs
    with open(os.path.join(args.save_dir, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(args.save_dir, "test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Success! Files saved to {args.save_dir}")

if __name__ == "__main__":
    main()