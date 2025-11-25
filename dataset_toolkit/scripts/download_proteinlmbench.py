#!/usr/bin/env python3
"""
Download ProteinLMBench from Hugging Face
Source: https://huggingface.co/datasets/tsynbio/ProteinLMBench
License: Apache 2.0
Total: 895,007 samples (1.21 GB)
"""

from datasets import load_dataset
from pathlib import Path
import json
from datetime import datetime

def download_proteinlmbench():
    print("="*70)
    print("ProteinLMBench Dataset Downloader")
    print("="*70)
    print("Source: https://huggingface.co/datasets/tsynbio/ProteinLMBench")
    print("License: Apache 2.0")
    print("Size: ~1.21 GB")
    print("="*70)
    
    output_dir = Path("data/proteinlmbench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # All available subsets
    subsets = [
        ("UniProt_Function", "Function prediction (PFUD)"),
        ("UniProt_Subunit structure", "Structure analysis (PSAD)"),
        ("Enzyme_CoT", "Enzyme reactions with Chain of Thought"),
        ("UniProt_Induction", "Induction conditions"),
        ("UniProt_Involvement in disease", "Disease associations"),
        ("UniProt_Post-translational modification", "PTM information"),
        ("UniProt_Tissue specificity", "Tissue expression"),
        ("evaluation", "Evaluation set")
    ]
    
    total_samples = 0
    downloaded = {}
    
    for subset_name, description in subsets:
        print(f"\n{'='*70}")
        print(f"Downloading: {subset_name}")
        print(f"Description: {description}")
        print(f"{'='*70}")
        
        try:
            # Load dataset from Hugging Face
            print(f"Loading from Hugging Face...")
            dataset = load_dataset(
                "tsynbio/ProteinLMBench",
                subset_name,
                split="train",
                trust_remote_code=True
            )
            
            # Create safe filename
            safe_name = subset_name.replace(" ", "_").replace("/", "_")
            output_file = output_dir / f"{safe_name}.json"
            
            # Convert to list
            print(f"Converting to JSON...")
            data = []
            for i, item in enumerate(dataset):
                data.append(dict(item))
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1:,} samples...")
            
            # Save to JSON
            print(f"Saving to file...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            samples = len(data)
            total_samples += samples
            downloaded[subset_name] = {
                "samples": samples,
                "file": str(output_file),
                "description": description
            }
            
            print(f"✓ Successfully downloaded {subset_name}")
            print(f"  Samples: {samples:,}")
            print(f"  Saved to: {output_file}")
            print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            print(f"✗ Error downloading {subset_name}: {e}")
            print(f"  You can try downloading it separately later")
            continue
    
    # Create summary
    print(f"\n{'='*70}")
    print("Download Summary")
    print(f"{'='*70}")
    
    summary = {
        "download_date": datetime.now().isoformat(),
        "source": "https://huggingface.co/datasets/tsynbio/ProteinLMBench",
        "license": "Apache 2.0",
        "total_samples": total_samples,
        "subsets": downloaded,
        "location": str(output_dir.absolute())
    }
    
    summary_file = output_dir / "download_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ DOWNLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"\nTotal samples downloaded: {total_samples:,}")
    print(f"Location: {output_dir.absolute()}")
    print(f"Summary: {summary_file}")
    
    # Show breakdown
    print(f"\n📊 Dataset Breakdown:")
    print(f"{'='*70}")
    for name, info in downloaded.items():
        print(f"{name:45s} {info['samples']:>8,} samples")
    print(f"{'='*70}")
    print(f"{'TOTAL':45s} {total_samples:>8,} samples")
    print(f"{'='*70}")
    
    # Usage recommendations
    print(f"\n💡 Dataset Mapping:")
    print(f"  PFUD (Function):     UniProt_Function.json (465K samples)")
    print(f"  PSAD (Structure):    UniProt_Subunit_structure.json (291K samples)")
    print(f"  Additional:          Other JSON files")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Check data/proteinlmbench/ directory")
    print(f"  2. Merge with Mol-Instructions data")
    print(f"  3. Start training your model!")
    
    return output_dir, total_samples

if __name__ == "__main__":
    try:
        output_dir, total = download_proteinlmbench()
        print(f"\n✅ Successfully downloaded {total:,} samples to {output_dir}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Install: pip install datasets huggingface-hub")
        print(f"  2. Check internet connection")
        print(f"  3. Try: huggingface-cli login (if needed)")
        import traceback
        traceback.print_exc()

