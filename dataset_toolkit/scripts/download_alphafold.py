#!/usr/bin/env python3
"""
Download AlphaFold structures using the correct URLs
Based on: https://alphafold.ebi.ac.uk/download
"""

import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

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

def download_individual_structures():
    """
    Download individual AlphaFold structures by UniProt ID
    """
    print("\n=== Downloading Individual AlphaFold Structures ===")
    
    af_dir = Path("data/alphafold")
    af_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample proteins with correct UniProt IDs
    sample_proteins = [
        ("P00520", "ABL1_MOUSE - Tyrosine-protein kinase"),
        ("P04637", "P53_HUMAN - Tumor suppressor p53"),
        ("P69905", "HBA_HUMAN - Hemoglobin subunit alpha"),
        ("P01112", "RASH_HUMAN - GTPase HRas"),
        ("P0DTC2", "SPIKE_SARS2 - Spike protein SARS-CoV-2"),
    ]
    
    # Updated URL format from AlphaFold FTP
    base_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest"
    
    success_count = 0
    
    for uniprot_id, description in sample_proteins:
        # AlphaFold files are organized by UniProt ID prefix
        prefix = uniprot_id[:3]  # First 3 characters
        
        # Construct the correct URL
        pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb.gz"
        url = f"{base_url}/{pdb_filename}"
        
        output_path = af_dir / pdb_filename
        pdb_output = af_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
        
        if pdb_output.exists():
            print(f"✓ {uniprot_id} ({description}) already exists, skipping...")
            success_count += 1
            continue
        
        try:
            # Download compressed file
            download_url(url, output_path)
            
            # Decompress
            print(f"Decompressing {pdb_filename}...")
            with gzip.open(output_path, 'rb') as f_in:
                with open(pdb_output, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file
            output_path.unlink()
            
            print(f"✓ Downloaded {uniprot_id} ({description})")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error downloading {uniprot_id}: {e}")
            print(f"  Alternative: Visit https://alphafold.ebi.ac.uk/entry/{uniprot_id}")
    
    print(f"\n✓ Successfully downloaded {success_count}/{len(sample_proteins)} structures")
    return af_dir

def download_swissprot_structures():
    """
    Download AlphaFold predictions for Swiss-Prot proteins (clustered at 50%)
    WARNING: This is a large file (~20GB compressed)
    """
    print("\n=== AlphaFold Swiss-Prot Dataset ===")
    print("⚠️  Warning: This is a LARGE download (~20GB compressed, ~120GB uncompressed)")
    
    response = input("Do you want to download the full Swiss-Prot structures? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Skipping Swiss-Prot bulk download")
        print("You can download it later from:")
        print("  https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/")
        return None
    
    af_dir = Path("data/alphafold/swissprot")
    af_dir.mkdir(parents=True, exist_ok=True)
    
    # Swiss-Prot clustered at 50% (smaller subset)
    url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v4.tar"
    output_path = af_dir / "swissprot_pdb_v4.tar"
    
    try:
        download_url(url, output_path)
        print("✓ Download complete!")
        print(f"  Saved to: {output_path}")
        print("\nTo extract, run:")
        print(f"  cd {af_dir} && tar -xvf swissprot_pdb_v4.tar")
        return af_dir
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    print("="*60)
    print("AlphaFold Structure Database Downloader")
    print("="*60)
    print("\nOptions:")
    print("1. Download sample structures (5 proteins, ~5MB)")
    print("2. Download full Swiss-Prot dataset (~20GB)")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        download_individual_structures()
    
    if choice in ['2', '3']:
        download_swissprot_structures()
    
    print("\n" + "="*60)
    print("✓ AlphaFold download complete!")
    print("="*60)
    print("\n📚 Resources:")
    print("  • AlphaFold Database: https://alphafold.ebi.ac.uk/")
    print("  • Download page: https://alphafold.ebi.ac.uk/download")
    print("  • FTP: https://ftp.ebi.ac.uk/pub/databases/alphafold/")

if __name__ == "__main__":
    main()












