"""
Generate v2 GTEA annotations with descriptive action labels.

UniTime's training data uses natural language queries:
  "person turn a light on."  (upstream sample)
  "He took out a pan"       (TaCoS)
  "A person opens a door."  (Charades-STA)

Our v1 GTEA annotations use single words: "take", "put", "open", etc.
This script maps them to descriptive sentences and writes v2 annotations.

Usage:
    python generate_v2_annot.py
"""
import json
import os

# Descriptive label mapping: single word → natural language sentence
# Matches UniTime's training data distribution (subject + verb + object context)
LABEL_MAP = {
    "take": "The person takes an object or ingredient from the workspace.",
    "put": "The person puts down an object or places an ingredient on the surface.",
    "open": "The person opens a container, jar, or package.",
    "close": "The person closes a container, jar, or package.",
    "pour": "The person pours liquid or contents from one container into another.",
    "spread": "The person spreads a substance onto bread or a surface with a knife.",
    "stir": "The person stirs the contents inside a bowl, cup, or pan.",
    "scoop": "The person scoops contents out of a container using a spoon or utensil.",
    "shake": "The person shakes a container to mix its contents.",
    "fold": "The person folds or wraps food items together.",
    "background": "No specific cooking action is being performed.",
}


def convert(input_path, output_path):
    data = json.load(open(input_path))
    for entry in data:
        for anno in entry["annos"]:
            old_query = anno["query"]
            anno["query"] = LABEL_MAP.get(old_query, old_query)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{input_path} → {output_path} ({len(data)} entries)")


if __name__ == "__main__":
    annot_dir = os.path.dirname(__file__) + "/annot"
    convert(f"{annot_dir}/train.json", f"{annot_dir}/train_v2.json")
    convert(f"{annot_dir}/test.json", f"{annot_dir}/test_v2.json")
