#!/usr/bin/env python3
"""
ProtTeX Dataset Preparation Script
This script helps download and prepare the datasets mentioned in the ProtTeX paper.
"""

import os
import urllib.request
import json
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import requests

# Create directories
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from a URL with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_mol_instructions():
    """
    Download Mol-Instructions dataset
    Source: https://github.com/IDEA-XL/Mol-Instructions
    """
    print("\n=== Downloading Mol-Instructions Dataset ===")
    mol_dir = DATA_DIR / "mol_instructions"
    mol_dir.mkdir(exist_ok=True)
    
    # Mol-Instructions is hosted on GitHub
    base_url = "https://raw.githubusercontent.com/zjunlp/Mol-Instructions/main/data/Protein-oriented_Instructions"
    
    files_to_download = [
        "protein_design.json",
        "protein_function_description.json",
    ]
    
    for file in files_to_download:
        output_path = mol_dir / file
        if output_path.exists():
            print(f"✓ {file} already exists, skipping...")
            continue
        
        try:
            url = f"{base_url}/{file}"
            download_url(url, output_path)
            print(f"✓ Downloaded {file}")
        except Exception as e:
            print(f"✗ Error downloading {file}: {e}")
            print(f"  You may need to download manually from: https://github.com/zjunlp/Mol-Instructions")
    
    return mol_dir

def download_swiss_prot_sample():
    """
    Download Swiss-Prot database sample
    """
    print("\n=== Downloading Swiss-Prot Sample ===")
    swiss_dir = DATA_DIR / "swiss_prot"
    swiss_dir.mkdir(exist_ok=True)
    
    # UniProt Swiss-Prot (small reviewed dataset)
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    output_path = swiss_dir / "uniprot_sprot.fasta.gz"
    
    if output_path.exists():
        print(f"✓ Swiss-Prot already exists, skipping...")
    else:
        try:
            download_url(url, output_path)
            print(f"✓ Downloaded Swiss-Prot database")
            
            # Uncompress
            print("Decompressing...")
            with gzip.open(output_path, 'rb') as f_in:
                with open(swiss_dir / "uniprot_sprot.fasta", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("✓ Decompressed successfully")
        except Exception as e:
            print(f"✗ Error downloading Swiss-Prot: {e}")
    
    return swiss_dir

def download_alphafold_sample():
    """
    Download AlphaFold Database sample
    Note: Full AFDB is very large (23TB+), so we download a small sample
    """
    print("\n=== Downloading AlphaFold Database Sample ===")
    af_dir = DATA_DIR / "alphafold"
    af_dir.mkdir(exist_ok=True)
    
    print("Note: Full AlphaFold DB is 23+ TB. Downloading representative samples...")
    
    # Download a few example structures from AlphaFold
    # Updated URLs - AlphaFold database structure changed
    sample_proteins = [
        ("P00520", "AF-P00520-F1"),  # ABL1_MOUSE
        ("P04637", "AF-P04637-F1"),  # P53_HUMAN
        ("P69905", "AF-P69905-F1"),  # HBA_HUMAN (Hemoglobin)
        ("P01112", "AF-P01112-F1"),  # RASH_HUMAN (HRAS)
    ]
    
    base_url = "https://alphafold.ebi.ac.uk/files"
    
    for uniprot_id, protein_id in sample_proteins:
        output_path = af_dir / f"{protein_id}-model_v4.pdb"
        if output_path.exists():
            print(f"✓ {protein_id} already exists, skipping...")
            continue
        
        try:
            # Try new URL format
            url = f"{base_url}/{protein_id}-model_v4.pdb"
            download_url(url, output_path)
            print(f"✓ Downloaded {protein_id}")
        except Exception as e:
            print(f"✗ Error downloading {protein_id}: {e}")
            print(f"  Note: AlphaFold structures can also be downloaded from:")
            print(f"  https://alphafold.ebi.ac.uk/entry/{uniprot_id}")
    
    return af_dir

def download_pdb_sample():
    """
    Download sample structures from RCSB PDB
    """
    print("\n=== Downloading RCSB PDB Sample ===")
    pdb_dir = DATA_DIR / "pdb"
    pdb_dir.mkdir(exist_ok=True)
    
    # Sample of well-known protein structures
    sample_pdbs = [
        "1crn",  # Crambin
        "1ubq",  # Ubiquitin
        "2lyz",  # Lysozyme
        "1mbn",  # Myoglobin
    ]
    
    base_url = "https://files.rcsb.org/download"
    
    for pdb_id in sample_pdbs:
        output_path = pdb_dir / f"{pdb_id}.pdb"
        if output_path.exists():
            print(f"✓ {pdb_id} already exists, skipping...")
            continue
        
        try:
            url = f"{base_url}/{pdb_id}.pdb"
            download_url(url, output_path)
            print(f"✓ Downloaded {pdb_id}")
        except Exception as e:
            print(f"✗ Error downloading {pdb_id}: {e}")
    
    return pdb_dir

def create_sample_dataset():
    """
    Create a small sample dataset for testing
    """
    print("\n=== Creating Sample Dataset ===")
    sample_dir = DATA_DIR / "sample"
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample protein data
    sample_data = [
        {
            "id": "sample_001",
            "sequence": "MKTIIALSYIFCLVFA",
            "function": "Sample protein for testing",
            "structure_file": "sample_001.pdb"
        },
        {
            "id": "sample_002", 
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "function": "Ubiquitin-like protein",
            "structure_file": "sample_002.pdb"
        }
    ]
    
    output_file = sample_dir / "sample_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✓ Created sample dataset: {output_file}")
    return sample_dir

def download_proteinlmbench():
    """
    Information about ProteinLMBench
    """
    print("\n=== ProteinLMBench Dataset ===")
    print("ProteinLMBench can be found at:")
    print("https://github.com/YuchenShen/ProteinLMBench")
    print("You may need to clone the repository manually.")
    
    bench_dir = DATA_DIR / "proteinlmbench"
    bench_dir.mkdir(exist_ok=True)
    
    # Write instructions
    instructions = """
# ProteinLMBench Instructions

To download ProteinLMBench:

1. Clone the repository:
   git clone https://github.com/YuchenShen/ProteinLMBench.git

2. Follow their instructions to download the datasets

Repository: https://github.com/YuchenShen/ProteinLMBench
"""
    
    with open(bench_dir / "README.md", 'w') as f:
        f.write(instructions)
    
    return bench_dir

def main():
    """Main function to download all datasets"""
    print("="*60)
    print("ProtTeX Dataset Preparation")
    print("="*60)
    
    print(f"\nData will be saved to: {DATA_DIR.absolute()}")
    
    # Create a summary report
    summary = {
        "directories_created": [],
        "files_downloaded": [],
        "notes": []
    }
    
    try:
        # Download datasets
        mol_dir = download_mol_instructions()
        summary["directories_created"].append(str(mol_dir))
        
        swiss_dir = download_swiss_prot_sample()
        summary["directories_created"].append(str(swiss_dir))
        
        af_dir = download_alphafold_sample()
        summary["directories_created"].append(str(af_dir))
        
        pdb_dir = download_pdb_sample()
        summary["directories_created"].append(str(pdb_dir))
        
        sample_dir = create_sample_dataset()
        summary["directories_created"].append(str(sample_dir))
        
        bench_dir = download_proteinlmbench()
        summary["directories_created"].append(str(bench_dir))
        
        # Save summary
        summary_file = DATA_DIR / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("✓ Dataset preparation complete!")
        print("="*60)
        print(f"\nData location: {DATA_DIR.absolute()}")
        print(f"Summary saved to: {summary_file}")
        
        print("\n📊 Dataset Summary:")
        print(f"  • Mol-Instructions: {mol_dir}")
        print(f"  • Swiss-Prot: {swiss_dir}")
        print(f"  • AlphaFold samples: {af_dir}")
        print(f"  • PDB samples: {pdb_dir}")
        print(f"  • Sample dataset: {sample_dir}")
        print(f"  • ProteinLMBench info: {bench_dir}")
        
        print("\n⚠️  Important Notes:")
        print("  1. Full datasets are very large (TB scale)")
        print("  2. This script downloads samples for testing")
        print("  3. For full datasets, visit the official sources:")
        print("     - AlphaFold: https://alphafold.ebi.ac.uk/")
        print("     - RCSB PDB: https://www.rcsb.org/")
        print("     - UniProt: https://www.uniprot.org/")
        print("     - Mol-Instructions: https://github.com/zjunlp/Mol-Instructions")
        
    except Exception as e:
        print(f"\n✗ Error during dataset preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

