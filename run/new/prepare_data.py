#!/usr/bin/env python3
"""
Download Prot2Text-Data from Hugging Face and save as CSV files.
Saves train, validation, and test splits to data/csv/.
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from datasets import load_dataset


def main():
    print("=" * 60)
    print("Download Prot2Text-Data → CSV")
    print("=" * 60)
    print(f"Dataset: https://huggingface.co/datasets/habdine/Prot2Text-Data")

    # Output directory: project_root/data/csv
    output_dir = root_dir / "data" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load dataset (all splits in one call to avoid re-downloading)
    print("\n📥 Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset("habdine/Prot2Text-Data")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1

    splits = ["train", "validation", "test"]
    for split in splits:
        if split not in dataset:
            print(f"⚠️ Split '{split}' not found, skipping.")
            continue
        out_path = output_dir / f"{split}.csv"
        print(f"\n📄 Saving {split} ({len(dataset[split])} rows) → {out_path}")
        df = dataset[split].to_pandas()
        df.to_csv(out_path, index=False)
        print(f"   ✅ Saved {out_path}")

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
