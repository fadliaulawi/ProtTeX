#!/usr/bin/env python3
"""
PSPD Dataset Download - AlphaFold Swiss-Prot + RCSB PDB

Usage:
    python download_pspd.py --mode test    # Download 10 samples each (verification)
    python download_pspd.py --mode full    # Download complete datasets
"""
import argparse
import requests
import subprocess
import json
import gzip
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# gsutil path (update if installed elsewhere)
GSUTIL = "/lustrefs/shared/mohammad.sayeed/google-cloud-sdk/bin/gsutil"

def download_rcsb_test(out_dir: Path, limit=10):
    """Download test samples from RCSB PDB (pre-Oct 2021)"""
    print(f"\n{'='*60}")
    print(f"RCSB PDB - Test Mode ({limit} samples)")
    print(f"{'='*60}")
    
    pdb_dir = out_dir / "rcsb_pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json={
        "query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.deposit_date", 
            "operator": "less", 
            "value": "2021-10-13"
        }},
        "return_type": "entry", 
        "request_options": {"paginate": {"start": 0, "rows": limit}}
    })
    
    pdb_ids = [x['identifier'] for x in r.json().get('result_set', [])]
    count = 0
    
    for pid in pdb_ids:
        try:
            resp = requests.get(f"https://files.rcsb.org/download/{pid}.pdb", timeout=30)
            if resp.ok:
                (pdb_dir / f"{pid}.pdb").write_bytes(resp.content)
                count += 1
                print(f"  ✓ {pid}")
            else:
                print(f"  ✗ {pid}: {resp.status_code}")
        except Exception as e:
            print(f"  ✗ {pid}: {e}")
    
    print(f"\n✓ RCSB PDB: {count}/{limit}")
    return count


def download_rcsb_full(out_dir: Path, batch_size=1000):
    """Download ALL RCSB PDB structures deposited before Oct 13, 2021"""
    print(f"\n{'='*60}")
    print("RCSB PDB - Full Download (pre-Oct 2021)")
    print(f"{'='*60}")
    
    pdb_dir = out_dir / "rcsb_pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    # First, get total count
    count_query = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json={
        "query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.deposit_date", 
            "operator": "less", 
            "value": "2021-10-13"
        }},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 0}, "return_counts": True}
    })
    
    total = count_query.json().get('total_count', 0)
    print(f"Total structures to download: {total:,}")
    
    # Download in batches
    downloaded = 0
    start = 0
    
    while start < total:
        print(f"\nBatch {start//batch_size + 1}: {start:,} - {min(start+batch_size, total):,}")
        
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json={
            "query": {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.deposit_date", 
                "operator": "less", 
                "value": "2021-10-13"
            }},
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": batch_size}}
        })
        
        pdb_ids = [x['identifier'] for x in r.json().get('result_set', [])]
        
        def download_one(pid):
            fpath = pdb_dir / f"{pid}.pdb"
            if fpath.exists():
                return True  # Skip existing
            try:
                resp = requests.get(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60)
                if resp.ok:
                    fpath.write_bytes(resp.content)
                    return True
            except:
                pass
            return False
        
        # Parallel download
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_one, pid): pid for pid in pdb_ids}
            batch_ok = 0
            for future in as_completed(futures):
                if future.result():
                    batch_ok += 1
            downloaded += batch_ok
            print(f"  Downloaded: {batch_ok}/{len(pdb_ids)} | Total: {downloaded:,}/{total:,}")
        
        start += batch_size
        time.sleep(0.5)  # Be nice to the server
    
    print(f"\n✓ RCSB PDB Complete: {downloaded:,} structures")
    return downloaded


def download_alphafold_test(out_dir: Path, limit=10):
    """Download test samples from AlphaFold"""
    print(f"\n{'='*60}")
    print(f"AlphaFold Swiss-Prot - Test Mode ({limit} samples)")
    print(f"{'='*60}")
    
    af_dir = out_dir / "alphafold"
    af_dir.mkdir(parents=True, exist_ok=True)
    
    # Well-known UniProt IDs
    ids = ['P12345', 'P00750', 'P01308', 'P68871', 'P69905', 
           'P04637', 'P62258', 'P31946', 'P04083', 'P63104',
           'P0A8P8', 'P0AES4', 'P69441', 'P0A6F5', 'P0A7B8'][:limit]
    
    count = 0
    for uid in ids:
        try:
            api = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}", timeout=15)
            if api.ok and api.json():
                pdb_url = api.json()[0].get('pdbUrl')
                if pdb_url:
                    resp = requests.get(pdb_url, timeout=30)
                    if resp.ok:
                        fname = pdb_url.split('/')[-1]
                        (af_dir / fname).write_bytes(resp.content)
                        count += 1
                        print(f"  ✓ {fname}")
                        continue
            print(f"  ✗ AF-{uid}: no data")
        except Exception as ex:
            print(f"  ✗ AF-{uid}: {ex}")
    
    print(f"\n✓ AlphaFold: {count}/{limit}")
    return count


def download_alphafold_full(out_dir: Path):
    """Download full AlphaFold Swiss-Prot dataset using gsutil"""
    print(f"\n{'='*60}")
    print("AlphaFold Swiss-Prot - Full Download")
    print(f"{'='*60}")
    
    af_dir = out_dir / "alphafold"
    af_dir.mkdir(parents=True, exist_ok=True)
    
    # Check gsutil
    gsutil_exists = Path(GSUTIL).exists()
    if not gsutil_exists:
        print(f"  gsutil not found at {GSUTIL}")
        print("  Falling back to HTTP method (slower)...")
        return download_alphafold_full_http(out_dir)
    
    print(f"  Using gsutil: {GSUTIL}")
    print("  Downloading Swiss-Prot proteomes from Google Cloud...")
    
    # List available Swiss-Prot proteome files
    try:
        result = subprocess.run(
            [GSUTIL, 'ls', 'gs://public-datasets-deepmind-alphafold-v4/proteomes/'],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            print(f"  Error listing: {result.stderr}")
            return download_alphafold_full_http(out_dir)
        
        # Get Swiss-Prot proteome files (model organisms)
        proteome_files = [
            'proteome-tax_id-9606-0_v4.tar',    # Human
            'proteome-tax_id-10090-0_v4.tar',   # Mouse
            'proteome-tax_id-10116-0_v4.tar',   # Rat
            'proteome-tax_id-7955-0_v4.tar',    # Zebrafish
            'proteome-tax_id-7227-0_v4.tar',    # Drosophila
            'proteome-tax_id-6239-0_v4.tar',    # C. elegans
            'proteome-tax_id-559292-0_v4.tar',  # Yeast
            'proteome-tax_id-83333-0_v4.tar',   # E. coli
            'proteome-tax_id-3702-0_v4.tar',    # Arabidopsis
        ]
        
        total_downloaded = 0
        for pfile in proteome_files:
            print(f"\n  Downloading {pfile}...")
            gs_path = f"gs://public-datasets-deepmind-alphafold-v4/proteomes/{pfile}"
            local_path = af_dir / pfile
            
            result = subprocess.run(
                [GSUTIL, '-m', 'cp', gs_path, str(local_path)],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and local_path.exists():
                print(f"    ✓ Downloaded {pfile}")
                # Extract
                print(f"    Extracting...")
                import tarfile
                try:
                    with tarfile.open(local_path, 'r') as tar:
                        tar.extractall(af_dir / pfile.replace('.tar', ''))
                    total_downloaded += 1
                    print(f"    ✓ Extracted")
                except Exception as e:
                    print(f"    ✗ Extract error: {e}")
            else:
                print(f"    ✗ Failed: {result.stderr[:100] if result.stderr else 'unknown'}")
        
        print(f"\n✓ AlphaFold: {total_downloaded} proteomes downloaded")
        return total_downloaded
        
    except Exception as e:
        print(f"  Error: {e}")
        return download_alphafold_full_http(out_dir)


def download_alphafold_full_http(out_dir: Path):
    """Fallback: Download AlphaFold structures via HTTP using Swiss-Prot ID list"""
    print("\n  HTTP Fallback Mode")
    print("  Downloading Swiss-Prot ID list...")
    
    af_dir = out_dir / "alphafold"
    af_dir.mkdir(parents=True, exist_ok=True)
    
    # Get Swiss-Prot IDs from UniProt
    # Download reviewed (Swiss-Prot) accession list
    try:
        print("  Fetching Swiss-Prot accession list...")
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=list&size=500",
            timeout=60
        )
        if not r.ok:
            print(f"  ✗ Failed to get ID list: {r.status_code}")
            return 0
        
        ids = r.text.strip().split('\n')[:500]  # Limit for HTTP method
        print(f"  Got {len(ids)} IDs (limited to 500 for HTTP method)")
        
        count = 0
        for i, uid in enumerate(ids):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(ids)}")
            try:
                api = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}", timeout=10)
                if api.ok and api.json():
                    pdb_url = api.json()[0].get('pdbUrl')
                    if pdb_url:
                        resp = requests.get(pdb_url, timeout=30)
                        if resp.ok:
                            fname = pdb_url.split('/')[-1]
                            (af_dir / fname).write_bytes(resp.content)
                            count += 1
            except:
                pass
        
        print(f"\n✓ AlphaFold (HTTP): {count} structures")
        return count
        
    except Exception as e:
        print(f"  Error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download PSPD Dataset")
    parser.add_argument('--mode', choices=['test', 'full'], required=True,
                        help='test: 10 samples each | full: complete datasets')
    parser.add_argument('--output', type=str, default='data/pspd',
                        help='Output directory')
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"PSPD DATASET DOWNLOAD - {args.mode.upper()} MODE")
    print("="*60)
    print(f"Output: {out_dir.absolute()}")
    
    if args.mode == 'test':
        rcsb = download_rcsb_test(out_dir, limit=10)
        af = download_alphafold_test(out_dir, limit=10)
    else:
        rcsb = download_rcsb_full(out_dir)
        af = download_alphafold_full(out_dir)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"RCSB PDB:  {rcsb:,} structures")
    print(f"AlphaFold: {af:,} structures/proteomes")
    print(f"Location:  {out_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()

