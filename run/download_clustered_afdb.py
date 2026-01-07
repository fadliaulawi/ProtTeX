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
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
import gzip
import shutil
import asyncio
import aiohttp
import aiofiles

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


def download_structure_test(accession: str, output_dir: Path) -> dict:
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
                    print(pdb_url)
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


async def download_structure_full(session: aiohttp.ClientSession, accession: str, temp_dir: Path, structures_dir: Path, metrics_dir: Path) -> dict:
    """Download a single AlphaFold structure and fetch its prediction metrics to temp directory."""
    result = {"accession": accession, "success": False, "pdb_content": None, "metrics": None}
    
    # Extract UniProt ID from accession (AF-A0A7Y8APW1-F1 -> A0A7Y8APW1)
    parts = accession.split("-")
    if len(parts) >= 2:
        uniprot_id = parts[1]
    else:
        uniprot_id = accession
    
    # Check cache: look for existing individual files from previous runs
    cached_pdb = structures_dir / f"{accession}.pdb"
    cached_metrics = metrics_dir / f"{accession}.json"
    pdb_file = temp_dir / f"{accession}.pdb"
    
    # Check and load PDB from cache if available
    pdb_cached = False
    if cached_pdb.exists():
        try:
            pdb_content = cached_pdb.read_bytes()
            async with aiofiles.open(pdb_file, 'wb') as f:
                await f.write(pdb_content)
            result["pdb_content"] = pdb_content
            result["success"] = len(pdb_content) > 0  # Success if file is not empty
            pdb_cached = True
        except Exception:
            # If loading from cache fails, proceed with download
            pass
    
    # Check and load metrics from cache if available
    metrics_cached = False
    if cached_metrics.exists():
        try:
            async with aiofiles.open(cached_metrics, 'r') as f:
                metrics_content = await f.read()
                metrics_data = json.loads(metrics_content)
                # Handle both dict and list formats
                if isinstance(metrics_data, dict):
                    entry = metrics_data
                elif isinstance(metrics_data, list) and len(metrics_data) > 0:
                    entry = metrics_data[0]
                else:
                    entry = {}
                
                # Extract only specified fields (same as download)
                result["metrics"] = {
                    "uniprot_id": entry.get("uniprot_id", uniprot_id),
                    "pTM": entry.get("pTM") or entry.get("globalMetricValue"),
                    "pLDDT_very_high": entry.get("pLDDT_very_high") or entry.get("fractionPlddtVeryHigh"),
                    "pLDDT_confident": entry.get("pLDDT_confident") or entry.get("fractionPlddtConfident"),
                    "pLDDT_low": entry.get("pLDDT_low") or entry.get("fractionPlddtLow"),
                    "pLDDT_very_low": entry.get("pLDDT_very_low") or entry.get("fractionPlddtVeryLow"),
                    "sequence_length": entry.get("sequence_length") or entry.get("uniprotEnd")
                }
                metrics_cached = True
        except Exception:
            # If loading from cache fails, proceed with download
            pass
    
    # If both are cached, return early
    if pdb_cached and metrics_cached:
        return result
    
    # Direct URL pattern: https://alphafold.ebi.ac.uk/files/{accession}-model_v6.pdb
    pdb_url = f"https://alphafold.ebi.ac.uk/files/{accession}-model_v6.pdb"
    
    try:
        # Download PDB file if not cached
        if not pdb_cached:
            async with session.get(pdb_url) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(pdb_file, 'wb') as f:
                        await f.write(content)
                    result["pdb_content"] = content
                    result["success"] = True
                else:
                    # Failed - create empty file to preserve order
                    async with aiofiles.open(pdb_file, 'wb') as f:
                        await f.write(b'')
                    result["error"] = f"PDB HTTP {response.status}"
        
        # Fetch prediction API metrics if not cached
        if not metrics_cached:
            api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
            async with session.get(api_url) as api_response:
                if api_response.status == 200:
                    metrics_data = await api_response.json()
                    if metrics_data and len(metrics_data) > 0:
                        entry = metrics_data[0]
                        # Extract only specified fields
                        result["metrics"] = {
                            "uniprot_id": uniprot_id,
                            "pTM": entry.get("globalMetricValue"),
                            "pLDDT_very_high": entry.get("fractionPlddtVeryHigh"),
                            "pLDDT_confident": entry.get("fractionPlddtConfident"),
                            "pLDDT_low": entry.get("fractionPlddtLow"),
                            "pLDDT_very_low": entry.get("fractionPlddtVeryLow"),
                            "sequence_length": entry.get("uniprotEnd")
                        }
                else:
                    # Failed - create empty JSON to preserve order
                    result["metrics"] = {}
            
    except asyncio.TimeoutError:
        # Create empty files for failed downloads
        async with aiofiles.open(pdb_file, 'wb') as f:
            await f.write(b'')
        result["error"] = "Timeout"
        result["metrics"] = {}
    except aiohttp.ClientError as e:
        async with aiofiles.open(pdb_file, 'wb') as f:
            await f.write(b'')
        result["error"] = f"Client error: {str(e)}"
        result["metrics"] = {}
    except Exception as e:
        async with aiofiles.open(pdb_file, 'wb') as f:
            await f.write(b'')
        result["error"] = str(e)
        result["metrics"] = {}
    
    return result

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
        result = download_structure_test(accession, output_dir)
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
    
    # Step 1: Download accession IDs list
    print("Step 1: Downloading accession IDs list...")
    accession_ids_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/updated_entries/v4_updated_entries-0.tar"
    
    if not os.path.exists(output_dir / "v4_updated_accessions.txt"):
        print(f"  Downloading accession IDs list...")
        subprocess.run(
            ["wget", "-c", "-O", str(output_dir / "v4_updated_accessions.txt"), accession_ids_url],
            check=True
        )
    else:
        print(f"  Accession IDs list already downloaded")
    
    # Step 2 & 3: Stream accessions from text file and download immediately
    print("\nStep 2-3: Streaming and downloading structures from text file...")
    
    # Quick count of total lines
    txt_path = output_dir / "v4_updated_accessions.txt"
    try:
        result = subprocess.run(
            ["wc", "-l", str(txt_path)],
            capture_output=True,
            text=True,
            check=True
        )
        total_lines = int(result.stdout.split()[0])
        print(f"  Estimated total accessions: {total_lines:,}")
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        # Fallback: wc not available or failed, use None (progress bar won't show total)
        total_lines = None
        print("  Could not determine total count (progress bar will show count only)")
   
    # HTTP download using async/await (faster than ThreadPoolExecutor)
    print("  Using async HTTP download with batching (100k files per batch)...")
    
    structures_batch_dir = output_dir / "structures_batched"
    metrics_batch_dir = output_dir / "metrics_batched"
    # Also check old individual file locations for cache (from previous runs)
    structures_cache_dir = output_dir / "structures"
    metrics_cache_dir = output_dir / "metrics"
    temp_dir = output_dir / "temp"
    structures_batch_dir.mkdir(exist_ok=True)
    metrics_batch_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    # Check for existing batches to resume from
    batch_size = 100000  # 100k files per batch
    existing_batches = list(metrics_batch_dir.glob("metrics_batch_*.json"))
    if existing_batches:
        # Extract batch numbers and find the highest
        batch_numbers = []
        for batch_file in existing_batches:
            try:
                # Extract number from filename like "metrics_batch_00042.json"
                batch_num = int(batch_file.stem.split('_')[-1])
                batch_numbers.append(batch_num)
            except (ValueError, IndexError):
                continue
        
        if batch_numbers:
            last_batch_num = max(batch_numbers)
            skip_count = (last_batch_num + 1) * batch_size  # Next batch to start
            print(f"  Found existing batches up to batch {last_batch_num:05d}")
            print(f"  Resuming from accession {skip_count + 1:,} (skipping first {skip_count:,} accessions)")
        else:
            skip_count = 0
            last_batch_num = -1
    else:
        skip_count = 0
        last_batch_num = -1
    
    async def async_download_structures():
        """Async function to download structures streaming from text file and batch into archives."""
        def accession_generator(txt_path, skip_lines=0):
            """Generator that yields accessions from text file one at a time (one per line)."""
            with open(txt_path, 'r', encoding='utf-8') as f:
                # Skip lines that were already processed
                for _ in range(skip_lines):
                    next(f, None)
                
                for line in f:
                    accession = line.strip()
                    if accession:  # Skip empty lines
                        yield accession
        
        def create_batch_archive(batch_num, batch_results, batch_dir):
            """Create tar archive and metrics JSON for a batch."""
            # Create tar archive for PDBs
            tar_file = structures_batch_dir / f"structures_batch_{batch_num:05d}.tar"
            with tarfile.open(tar_file, 'w') as tar:
                for result in batch_results:
                    accession = result["accession"]
                    pdb_file = batch_dir / f"{accession}.pdb"
                    if pdb_file.exists():
                        tar.add(pdb_file, arcname=f"{accession}.pdb")
                        pdb_file.unlink()  # Remove temp file after adding to tar
            
            # Create metrics JSON array
            metrics_file = metrics_batch_dir / f"metrics_batch_{batch_num:05d}.json"
            metrics_list = [result.get("metrics", {}) for result in batch_results]
            with open(metrics_file, 'w') as f:
                json.dump(metrics_list, f, indent=2)
            
            return len([r for r in batch_results if r["success"]])
        
        success_count = 0
        total_processed = 0
        max_concurrent = 1024  # Higher concurrency with async
        batch_results = []
        batch_num = last_batch_num + 1  # Start from next batch after last completed
        
        from tqdm import tqdm
        from collections import OrderedDict
        
        accession_gen = accession_generator(txt_path, skip_count)
        tasks = OrderedDict()  # Track tasks by accession to preserve order
        results_dict = OrderedDict()  # Store results by accession
        accession_order = []  # Track order of accessions
        generator_exhausted = False
        next_index = 0  # Next expected result index
        
        connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Adjust progress bar total to account for skipped lines
            progress_total = total_lines - skip_count if total_lines else None
            with tqdm(total=progress_total, initial=0, desc="Downloading structures", unit=" structures") as pbar:
                while True:
                    # Fill up to max_concurrent tasks
                    while len(tasks) < max_concurrent and not generator_exhausted:
                        try:
                            accession = next(accession_gen)
                            accession_order.append(accession)
                            task = asyncio.create_task(
                                download_structure_full(session, accession, temp_dir, structures_cache_dir, metrics_cache_dir)
                            )
                            tasks[accession] = task
                        except StopIteration:
                            generator_exhausted = True
                            break
                    
                    if not tasks and next_index >= len(accession_order):
                        break
                    
                    # Wait for at least one task to complete
                    done, pending = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
                    
                    # Process completed tasks
                    for task in done:
                        # Find which accession this task belongs to
                        accession = None
                        for acc, t in tasks.items():
                            if t == task:
                                accession = acc
                                break
                        
                        if accession:
                            result = await task
                            results_dict[accession] = result
                            del tasks[accession]
                    
                    # Process results in order (as they appear in input file)
                    while next_index < len(accession_order):
                        accession = accession_order[next_index]
                        if accession in results_dict:
                            result = results_dict[accession]
                            del results_dict[accession]
                            total_processed += 1
                            batch_results.append(result)
                            
                    if result["success"]:
                        success_count += 1
                
                        # Create batch archive when batch is full
                        if len(batch_results) >= batch_size:
                            batch_success = create_batch_archive(batch_num, batch_results, temp_dir)
                            batch_num += 1
                            batch_results = []
                            print(f"\n  Created batch {batch_num}: {batch_success:,} successful structures")
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            "success": success_count,
                            "total": total_processed,
                            "batch": batch_num
                        })
                        next_index += 1
                    else:
                        # Haven't received this result yet, wait for more
                        break
                    
                    tasks = OrderedDict((acc, t) for acc, t in tasks.items() if acc not in results_dict)
        
        # Create final batch if there are remaining files
        if batch_results:
            batch_success = create_batch_archive(batch_num, batch_results, temp_dir)
            batch_num += 1
            print(f"\n  Created final batch {batch_num}: {batch_success:,} successful structures")
        
        # Clean up temp directory
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        return success_count, total_processed
    
    # Run the async download
    success_count, total_accessions = asyncio.run(async_download_structures())
    print(f"\n  Download complete: {success_count:,}/{total_accessions:,} structures")
    
    # Count batch files
    batch_count = len(list(structures_batch_dir.glob("structures_batch_*.tar")))
    print(f"  Created {batch_count} structure batch archives")
    print(f"  Created {batch_count} metrics batch JSON files")
    
    # Save manifest
    manifest_file = output_dir / "metadata.json"
    batch_count = len(list(structures_batch_dir.glob("structures_batch_*.tar")))
    manifest = {
        "source": "AFDB v4 Clustered (Barrio-Hernandez et al.)",
        "success_count": success_count,
        "total_cluster_reps": total_accessions,
        "batch_size": 100000,
        "batch_count": batch_count,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    
    print()
    print("="*60)
    print("FULL DOWNLOAD COMPLETE")
    print(f"Cluster representatives: {total_accessions:,}")
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
        default="data/datasets/afdb_clustered",
        help="Output directory for downloaded structures"
    )
    parser.add_argument(
        "--gsutil_path",
        type=str,
        default="gsutil/google-cloud-sdk/bin/gsutil",
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