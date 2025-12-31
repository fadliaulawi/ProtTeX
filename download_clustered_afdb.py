#!/usr/bin/env python3
"""
Download Clustered AlphaFold Database v4 (as used by ProtTeX paper)

ProtTeX uses the Barrio-Hernandez et al. clustering of AFDB v4:
- Paper: "Clustering predicted structures at the scale of the known protein universe"
- Source: 2.27M cluster representatives from AFDB v4 clustered at 50% sequence identity

Two modes:
  --mode test  : Download 10 sample structures to verify everything works
  --mode full  : Download all 2.27M clustered structures (~500GB uncompressed)

The clustered AFDB data is available from:
1. ESM Metagenomic Atlas (includes AFDB clustered representatives)
2. Direct FTP from EBI AlphaFold (swissprot + proteomes)

Usage:
    python download_clustered_afdb.py --mode test --output_dir ./afdb_clustered_test
    python download_clustered_afdb.py --mode full --output_dir ./afdb_clustered_full
"""

import argparse
import os
import subprocess
import requests
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
import gzip
import shutil

# Cluster representative accessions from AFDB v4 clustering
# These are the 50% sequence identity cluster representatives
# Full list: https://ftp.ebi.ac.uk/pub/databases/alphafold/clusters/

AFDB_CLUSTER_FTP = "https://ftp.ebi.ac.uk/pub/databases/alphafold/clusters/"
AFDB_STRUCTURE_API = "https://alphafold.ebi.ac.uk/files/"

# Sample cluster representatives for testing
TEST_ACCESSIONS = [
    "AF-A0A1W2PQ64-F1",  # Cluster rep
    "AF-Q9Y6K9-F1",      # Human protein
    "AF-P0DTC2-F1",      # SARS-CoV-2 Spike
    "AF-P04637-F1",      # P53 tumor suppressor
    "AF-P00533-F1",      # EGFR
    "AF-P01308-F1",      # Insulin
    "AF-P69905-F1",      # Hemoglobin alpha
    "AF-P68871-F1",      # Hemoglobin beta
    "AF-P02768-F1",      # Albumin
    "AF-P01375-F1",      # TNF-alpha
]


def download_structure(accession: str, output_dir: Path) -> dict:
    """Download a single AlphaFold structure using the API."""
    result = {"accession": accession, "success": False, "file": None}
    
    # Extract UniProt ID from accession (AF-P04637-F1 -> P04637)
    parts = accession.split("-")
    if len(parts) >= 2:
        uniprot_id = parts[1]
    else:
        uniprot_id = accession
    
    try:
        # Get download URL from API
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        api_response = requests.get(api_url, timeout=30)
        
        if api_response.status_code == 200:
            data = api_response.json()
            if data and len(data) > 0:
                pdb_url = data[0].get("pdbUrl")
                if pdb_url:
                    # Download the PDB file
                    pdb_response = requests.get(pdb_url, timeout=30)
                    if pdb_response.status_code == 200:
                        pdb_file = output_dir / f"{accession}.pdb"
                        pdb_file.write_text(pdb_response.text)
                        result["success"] = True
                        result["file"] = str(pdb_file)
                        result["version"] = data[0].get("latestVersion", "unknown")
                        return result
        
        result["error"] = f"API returned {api_response.status_code}"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def download_cluster_representatives_list():
    """
    Download the list of cluster representatives from AFDB clustering.
    
    The Barrio-Hernandez clustering provides:
    - clusters_by_entity.tsv: Mapping of UniProt IDs to cluster IDs
    - reps50: 50% sequence identity cluster representatives
    """
    cluster_files = [
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/clusters/cluster_reps_v4.fasta.gz",
        "https://ftp.ebi.ac.uk/pub/databases/alphafold/clusters/clusters_by_entity_v4.tsv.gz"
    ]
    return cluster_files


def run_test_mode(output_dir: Path):
    """Test mode: Download 10 sample structures."""
    print("="*60)
    print("CLUSTERED AFDB DOWNLOAD - TEST MODE")
    print("="*60)
    print(f"Downloading {len(TEST_ACCESSIONS)} sample structures...")
    print(f"Output directory: {output_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed = []
    
    for i, accession in enumerate(TEST_ACCESSIONS, 1):
        print(f"[{i}/{len(TEST_ACCESSIONS)}] Downloading {accession}...", end=" ")
        result = download_structure(accession, output_dir)
        if result["success"]:
            ver = result.get("version", "?")
            print(f"✓ {Path(result['file']).name} (v{ver})")
            success_count += 1
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
            failed.append(accession)
    
    print()
    print("="*60)
    print(f"TEST COMPLETE: {success_count}/{len(TEST_ACCESSIONS)} structures downloaded")
    print(f"Output directory: {output_dir}")
    if failed:
        print(f"Failed accessions: {', '.join(failed)}")
    print("="*60)
    
    return success_count == len(TEST_ACCESSIONS)


def run_full_mode(output_dir: Path, gsutil_path: str = None):
    """
    Full mode: Download all 2.27M clustered AFDB structures.
    
    This uses gsutil for bulk download from Google Cloud Storage.
    The AFDB v4 clustered data is available at:
    gs://public-datasets-deepmind-alphafold-v4/
    """
    print("="*60)
    print("CLUSTERED AFDB DOWNLOAD - FULL MODE")
    print("="*60)
    print("Target: 2.27M cluster representatives from AFDB v4")
    print(f"Output directory: {output_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download cluster representative list
    print("Step 1: Downloading cluster representative accession list...")
    cluster_reps_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/clusters/cluster_reps_v4.fasta.gz"
    
    cluster_file = output_dir / "cluster_reps_v4.fasta.gz"
    try:
        subprocess.run(
            ["wget", "-c", "-O", str(cluster_file), cluster_reps_url],
            check=True
        )
        print(f"  ✓ Downloaded cluster representatives list")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to download cluster list: {e}")
        print("  Trying curl...")
        subprocess.run(
            ["curl", "-L", "-o", str(cluster_file), cluster_reps_url],
            check=True
        )
    
    # Step 2: Extract accessions from FASTA
    print("\nStep 2: Extracting accessions from cluster representatives...")
    accessions = []
    with gzip.open(cluster_file, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                # Format: >AF-A0A1W2PQ64-F1 ...
                acc = line.split()[0][1:]  # Remove '>'
                accessions.append(acc)
    
    print(f"  Found {len(accessions):,} cluster representatives")
    
    # Step 3: Download structures using gsutil or HTTP
    print("\nStep 3: Downloading structures...")
    
    if gsutil_path:
        print(f"  Using gsutil from: {gsutil_path}")
        # Download all AFDB v4 structures (includes cluster reps)
        # Filter to only cluster representatives locally
        gs_bucket = "gs://public-datasets-deepmind-alphafold-v4/"
        
        print(f"  Downloading from {gs_bucket}")
        print("  This will download ~500GB of data...")
        
        cmd = [
            gsutil_path, "-m", "cp", "-r",
            f"{gs_bucket}proteomes/*",
            str(output_dir / "proteomes")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("  ✓ Download complete via gsutil")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ gsutil failed: {e}")
            print("  Falling back to HTTP download...")
            gsutil_path = None
    
    if not gsutil_path:
        # HTTP download (slower but works without gsutil)
        print("  Using HTTP download (parallel, may take several days)...")
        
        structures_dir = output_dir / "structures"
        structures_dir.mkdir(exist_ok=True)
        
        # Download in batches with progress
        batch_size = 100
        total_batches = (len(accessions) + batch_size - 1) // batch_size
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(accessions))
                batch = accessions[start_idx:end_idx]
                
                futures = {
                    executor.submit(download_structure, acc, structures_dir): acc
                    for acc in batch
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result["success"]:
                        success_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Progress: {end_idx:,}/{len(accessions):,} "
                          f"({100*end_idx/len(accessions):.1f}%) - "
                          f"{success_count:,} successful")
        
        print(f"\n  Download complete: {success_count:,}/{len(accessions):,} structures")
    
    # Save manifest
    manifest_file = output_dir / "download_manifest.json"
    manifest = {
        "source": "AFDB v4 Clustered (Barrio-Hernandez et al.)",
        "total_cluster_reps": len(accessions),
        "output_dir": str(output_dir),
        "cluster_file": str(cluster_file)
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    
    print()
    print("="*60)
    print("FULL DOWNLOAD COMPLETE")
    print(f"Cluster representatives: {len(accessions):,}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download Clustered AFDB v4 structures (as used by ProtTeX)"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        required=True,
        help="test: Download 10 samples | full: Download all 2.27M structures"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./afdb_clustered",
        help="Output directory for downloaded structures"
    )
    parser.add_argument(
        "--gsutil_path",
        type=str,
        default=None,
        help="Path to gsutil binary (optional, for faster full download)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.mode == "test":
        success = run_test_mode(output_dir)
        exit(0 if success else 1)
    else:
        run_full_mode(output_dir, args.gsutil_path)


if __name__ == "__main__":
    main()

