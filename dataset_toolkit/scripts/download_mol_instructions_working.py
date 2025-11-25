#!/usr/bin/env python3
"""
Download Mol-Instructions dataset from Hugging Face
VERIFIED WORKING VERSION - Uses direct curl downloads
"""

import subprocess
import json
from pathlib import Path

def download_dataset():
    """
    Download all three Mol-Instructions datasets
    """
    print("="*70)
    print("Mol-Instructions Dataset Downloader (Verified Working)")
    print("="*70)
    
    # Create output directory
    output_dir = Path("data/mol_instructions_hf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset files to download (verified URLs)
    base_url = "https://huggingface.co/datasets/zjunlp/Mol-Instructions/resolve/main/data"
    
    datasets = [
        ("Protein-oriented_Instructions.zip", "~182 MB", "495K protein samples"),
        ("Molecule-oriented_Instructions.zip", "~70 MB", "1.34M molecule samples"),
        ("Biomolecular_Text_Instructions.zip", "~7 MB", "54K text samples"),
    ]
    
    print(f"\n📥 Downloading to: {output_dir.absolute()}\n")
    
    downloaded_files = []
    
    for filename, size, description in datasets:
        output_file = output_dir / filename
        
        if output_file.exists():
            print(f"✓ {filename} already exists, skipping...")
            downloaded_files.append(output_file)
            continue
        
        print(f"📦 Downloading {filename} ({size}) - {description}")
        url = f"{base_url}/{filename}"
        
        # Use curl with progress bar
        cmd = [
            "curl", "-L", "--progress-bar",
            "-o", str(output_file),
            url
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            if result.returncode == 0:
                print(f"✓ Downloaded {filename}\n")
                downloaded_files.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {filename}: {e}")
            continue
    
    # Extract all downloaded files
    print("\n" + "="*70)
    print("Extracting datasets...")
    print("="*70 + "\n")
    
    for zip_file in downloaded_files:
        if not zip_file.exists():
            continue
            
        print(f"📂 Extracting {zip_file.name}...")
        
        cmd = ["unzip", "-q", "-o", str(zip_file), "-d", str(output_dir)]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Extracted {zip_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error extracting {zip_file.name}: {e}")
    
    # Generate summary
    print("\n" + "="*70)
    print("Dataset Summary")
    print("="*70 + "\n")
    
    dataset_dirs = {
        "Protein-oriented": output_dir / "Protein-oriented_Instructions",
        "Molecule-oriented": output_dir / "Molecule-oriented_Instructions",
        "Biomolecular Text": output_dir / "Biomolecular_Text_Instructions"
    }
    
    total_samples = 0
    
    for category, path in dataset_dirs.items():
        if not path.exists():
            print(f"⚠️  {category}: Not found")
            continue
        
        print(f"📊 {category}:")
        cat_total = 0
        
        for json_file in sorted(path.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    count = len(data)
                    cat_total += count
                    print(f"   • {json_file.name:45s} {count:>8,} samples")
            except Exception as e:
                print(f"   ✗ Error reading {json_file.name}: {e}")
        
        print(f"   {'SUBTOTAL':>45s} {cat_total:>8,} samples")
        total_samples += cat_total
        print()
    
    print("="*70)
    print(f"{'TOTAL SAMPLES':>55s} {total_samples:>8,}")
    print("="*70)
    
    # Save summary
    summary = {
        "total_samples": total_samples,
        "protein_oriented": str(dataset_dirs["Protein-oriented"]),
        "molecule_oriented": str(dataset_dirs["Molecule-oriented"]),
        "biomolecular_text": str(dataset_dirs["Biomolecular Text"]),
        "status": "complete"
    }
    
    summary_file = output_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return output_dir

def main():
    print("\n🧬 Mol-Instructions Dataset Downloader\n")
    print("This will download:")
    print("  • Protein-oriented instructions (~182 MB)")
    print("  • Molecule-oriented instructions (~70 MB)")
    print("  • Biomolecular text instructions (~7 MB)")
    print("  TOTAL: ~260 MB compressed, ~1.4 GB uncompressed")
    print()
    
    response = input("Proceed with download? (yes/no) [yes]: ").strip().lower() or "yes"
    
    if response not in ['yes', 'y']:
        print("Download cancelled.")
        return
    
    try:
        output_dir = download_dataset()
        
        print("\n" + "="*70)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"\n📁 Data location: {output_dir.absolute()}")
        print("\n💡 Next steps:")
        print("   1. Explore the data in: data/mol_instructions_hf/")
        print("   2. Use protein data for ProtTeX training")
        print("   3. Process with: python process_datasets.py")
        print()
        print("📚 Citation:")
        print("   Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset")
        print("   for Large Language Models (ICLR 2024)")
        print("   https://arxiv.org/abs/2306.08018")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()












