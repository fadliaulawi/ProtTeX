#!/usr/bin/env python3
"""Prepare PFUD (Protein Function Understanding Dataset) for ProtTeX model."""

import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os
import sys
import numpy as np

# Add ProToken to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add workspace root
sys.path.insert(0, str(Path(__file__).parent.parent / 'ProToken'))  # Add ProToken directly
sys.path.insert(0, './ProToken')

from data.pipeline import from_pdb_string

class PFUDPreparer:
    """Prepare PFUD dataset by collecting protein-QA pairs from UniProt/SwissProt
    
    QA pairs are sourced from Mol-Instructions and ProteinLMBench datasets.
    These are matched to proteins by UniProt accession.
    """
    
    def __init__(self, output_dir: str, cache_dir: str, 
                mol_instructions_path: Optional[str] = "../Mol-Instructions/data/Protein-oriented_Instructions",
                protein_lmbench_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to real QA datasets
        self.mol_instructions_path = mol_instructions_path
        self.protein_lmbench_path = protein_lmbench_path
        
        # Loaded QA data indexed by accession
        self.mol_instruction_qa = {}
        self.protein_lmbench_qa = {}
        
        # Cache for UniProt stream results to avoid re-fetching
        self._uniprot_stream_cache = None
        
        # Load datasets if provided
        self._load_qa_datasets()
    
    def _load_qa_datasets(self):
        """Load QA pairs from Mol-Instructions and ProteinLMBench datasets."""
        print("Loading QA datasets...")
        
        # Load Mol-Instructions dataset (can be a file or directory)
        if self.mol_instructions_path and os.path.exists(self.mol_instructions_path):
            print(f"Loading Mol-Instructions from {self.mol_instructions_path}...")
            try:
                mol_files = []
                if os.path.isdir(self.mol_instructions_path):
                    # If directory, load all JSON files
                    mol_files = [os.path.join(self.mol_instructions_path, f) 
                                for f in os.listdir(self.mol_instructions_path) 
                                if f.endswith('.json')]
                    print(f"Found {len(mol_files)} JSON files in directory")
                else:
                    # If single file
                    mol_files = [self.mol_instructions_path]
                
                for mol_file in mol_files:
                    print(f"  Loading {os.path.basename(mol_file)}...")
                    with open(mol_file, 'r') as f:
                        mol_data = json.load(f)
                        # Index by UniProt accession if present
                        items = mol_data if isinstance(mol_data, list) else mol_data.get('data', [])
                        for item in items:
                            # Get accession from different possible locations
                            accession = (item.get('accession') or 
                                       item.get('uniprot_id') or 
                                       (item.get('metadata', {}).get('protein_accession') if isinstance(item.get('metadata'), dict) else None))
                            if accession:
                                if accession not in self.mol_instruction_qa:
                                    self.mol_instruction_qa[accession] = []
                                self.mol_instruction_qa[accession].append(item)
                
                print(f"Loaded {len(self.mol_instruction_qa)} proteins from Mol-Instructions")
            except Exception as e:
                print(f"Warning: Could not load Mol-Instructions: {e}")
        else:
            print("Mol-Instructions path not provided or not found")
        
        # Load ProteinLMBench dataset (can be a file or directory)
        if self.protein_lmbench_path and os.path.exists(self.protein_lmbench_path):
            print(f"Loading ProteinLMBench from {self.protein_lmbench_path}...")
            try:
                lmbench_files = []
                if os.path.isdir(self.protein_lmbench_path):
                    # If directory, load all JSON files
                    lmbench_files = [os.path.join(self.protein_lmbench_path, f) 
                                   for f in os.listdir(self.protein_lmbench_path) 
                                   if f.endswith('.json')]
                    print(f"Found {len(lmbench_files)} JSON files in directory")
                else:
                    # If single file
                    lmbench_files = [self.protein_lmbench_path]
                
                for lmbench_file in lmbench_files:
                    print(f"  Loading {os.path.basename(lmbench_file)}...")
                    with open(lmbench_file, 'r') as f:
                        lmbench_data = json.load(f)
                        # Index by UniProt accession if present
                        for item in lmbench_data if isinstance(lmbench_data, list) else lmbench_data.get('data', []):
                            accession = item.get('accession') or item.get('uniprot_id')
                            if accession:
                                if accession not in self.protein_lmbench_qa:
                                    self.protein_lmbench_qa[accession] = []
                                self.protein_lmbench_qa[accession].append(item)
                
                print(f"Loaded {len(self.protein_lmbench_qa)} proteins from ProteinLMBench")
            except Exception as e:
                print(f"Warning: Could not load ProteinLMBench: {e}")
        else:
            print("ProteinLMBench path not provided or not found")
    
    def prepare_proteins(self, num_samples: int) -> List[Dict]:
        """Complete pipeline to prepare and enrich proteins with structures.
        
        This combines fetching from UniProt, creating QA pairs, and fetching structures
        into a single unified process where num_samples represents the final count of
        fully prepared proteins (with structures and QA pairs).
        
        Supports pagination: automatically fetches different batches across multiple requests.
        
        Args:
            num_samples: Target number of fully prepared proteins (with structures and QA pairs)
        
        Returns:
            List of prepared protein dictionaries with structures
        """
        print("=" * 60)
        print(f"Preparing {num_samples} complete proteins with structures")
        print("=" * 60)
        
        prepared_proteins = []
        proteins_fetched = 0
        current_offset = 0  # Track where we are in UniProt results
        
        # We need to fetch more proteins to account for skipping
        # Estimate: ~60-70% will have QA pairs, ~80-90% will have structures
        # So fetch ~2x the target to ensure we get enough
        fetch_batch_size = max(num_samples * 2, 100)
        
        while len(prepared_proteins) < num_samples:
            print(f"\n[Progress] {len(prepared_proteins)}/{num_samples} proteins prepared")
            
            # Fetch batch of proteins with pagination
            print(f"Fetching next batch of {fetch_batch_size} proteins from UniProt (offset={current_offset})...")
            proteins = self.fetch_uniprot_proteins(fetch_batch_size, offset=current_offset)
            
            if not proteins:
                print("⚠ No more proteins available from UniProt")
                break
            
            proteins_fetched += len(proteins)
            current_offset += len(proteins)  # Move offset forward for next batch
            
            # Create QA pairs for this batch
            print(f"Creating QA pairs for {len(proteins)} proteins...")
            batch_with_qa = self.create_qa_pairs(proteins)
            
            if not batch_with_qa:
                print("⚠ No QA pairs found in this batch, continuing...")
                continue
            
            # Fetch structures for this batch
            print(f"Fetching PDB structures for {len(batch_with_qa)} proteins...")
            batch_with_structures, skipped = self.add_structure_to_dataset(batch_with_qa)
            
            # Add to prepared list
            prepared_proteins.extend(batch_with_structures)
            
            # If we have enough, stop
            if len(prepared_proteins) >= num_samples:
                prepared_proteins = prepared_proteins[:num_samples]
                break
        
        print("\n" + "=" * 60)
        print(f"Preparation complete!")
        print(f"  Proteins fetched: {proteins_fetched}")
        print(f"  Proteins prepared: {len(prepared_proteins)}")
        print(f"  Next fetch should start at offset: {current_offset}")
        print("=" * 60)
        
        return prepared_proteins
    
    def fetch_uniprot_proteins(self, num_samples: int, offset: int = 0) -> List[Dict]:
        """Fetch protein data from UniProt REST API using stream endpoint (reviewed entries with PDB xrefs).
        
        Uses the /stream endpoint to get ALL matching results in one request (up to 10M).
        Results are cached in memory and saved to disk to avoid re-fetching on subsequent runs.
        Skips the first `offset` entries, then returns `num_samples` entries.
        
        Args:
            num_samples: Number of proteins to return in this batch
            offset: Number of proteins to skip from the beginning
        
        Returns:
            List of protein dictionaries (num_samples entries after skipping offset)
        """
        cache_file = self.cache_dir / "uniprot_stream_cache.json"
        
        # Check if we already have cached results (in memory or on disk)
        if self._uniprot_stream_cache is None:
            # Try to load from disk first
            if cache_file.exists():
                print(f"Loading cached proteins from {cache_file}...")
                try:
                    with open(cache_file, 'r') as f:
                        self._uniprot_stream_cache = json.load(f)
                    print(f"✓ Loaded {len(self._uniprot_stream_cache)} proteins from cache file")
                except Exception as e:
                    print(f"⚠ Failed to load cache file: {e}, will re-fetch from UniProt")
                    self._uniprot_stream_cache = None
            
            # If still no cache, fetch from UniProt
            if self._uniprot_stream_cache is None:
                print(f"Fetching ALL proteins from UniProt stream endpoint (first time)...")
                
                url = "https://rest.uniprot.org/uniprotkb/stream"

                # Reviewed (Swiss-Prot) entries that have a PDB cross-reference
                query = "reviewed:true AND database:pdb"

                # UniProt stream endpoint fields
                fields = "accession,sequence,organism_name,protein_name"

                params = {
                    "query": query,
                    "format": "json",
                    "fields": fields,
                }

                headers = {
                    "Accept": "application/json",
                    "User-Agent": "uniprot-client/1.0 (+your.email@example.com)"
                }

                print(f"Querying UniProt stream endpoint (this may take a moment)...")
                resp = requests.get(url, params=params, headers=headers, timeout=120)  # Longer timeout
                resp.raise_for_status()

                data = resp.json()
                self._uniprot_stream_cache = []

                for entry in data.get("results", []):
                    # Primary accession
                    accession = entry.get("primaryAccession", "")

                    # Sequence value (full aa sequence) if returned
                    seq = ""
                    seq_obj = entry.get("sequence")
                    if isinstance(seq_obj, dict):
                        seq = seq_obj.get("value", "") or ""

                    # Organism scientific name
                    organism = "Unknown"
                    org_obj = entry.get("organism")
                    if isinstance(org_obj, dict):
                        organism = org_obj.get("scientificName", "Unknown")

                    # Protein name (recommended name if present)
                    protein_name = "Unknown"
                    pd = entry.get("proteinDescription") or {}
                    rec = (pd.get("recommendedName") or {})
                    full = (rec.get("fullName") or {})
                    protein_name = full.get("value") or entry.get("uniProtkbId") or "Unknown"

                    if seq:  # keep only entries with a sequence
                        self._uniprot_stream_cache.append({
                            "accession": accession,
                            "sequence": seq,
                            "organism": organism,
                            "protein_name": protein_name,
                        })

                print(f"✓ Fetched {len(self._uniprot_stream_cache)} proteins from UniProt stream")
                
                # Save to cache file
                print(f"Saving cache to {cache_file}...")
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(self._uniprot_stream_cache, f, indent=2)
                    print(f"✓ Cache saved to disk")
                except Exception as e:
                    print(f"⚠ Failed to save cache file: {e}")
        
        # Now slice from cache
        print(f"Retrieving proteins from cache (skipping first {offset}, then fetching {num_samples})...")
        proteins = self._uniprot_stream_cache[offset:offset + num_samples]
        print(f"✓ Retrieved {len(proteins)} proteins from cache (offset={offset})")
        return proteins
    
    def create_qa_pairs(self, proteins: List[Dict]) -> List[Dict]:
        """Create QA pairs for proteins using real data from Mol-Instructions and ProteinLMBench.
        
        QA pairs are matched to proteins by UniProt accession. If no matching QA pair is found,
        the protein is skipped (as per the paper: "If an accession does not exist in our database,
        the corresponding questions were dropped").
        """
        print("Creating QA pairs from real datasets...")
        
        dataset = []
        skipped_count = 0
        
        for protein in proteins:
            accession = protein.get("accession", "")
            
            # Try to find QA pairs for this accession from either dataset
            qa_list = self.mol_instruction_qa.get(accession, []) + self.protein_lmbench_qa.get(accession, [])
            
            if not qa_list:
                # Paper states: "If an accession does not exist in our database, the corresponding questions were dropped"
                skipped_count += 1
                continue
            
            # Convert real QA pairs to standardized format
            qa_pairs = []
            for qa_item in qa_list:
                # For Mol-Instructions format: instruction -> question, output -> answer
                question = (qa_item.get('question') or 
                           qa_item.get('query') or 
                           qa_item.get('text') or
                           qa_item.get('instruction'))
                answer = (qa_item.get('answer') or 
                         qa_item.get('response') or 
                         qa_item.get('label') or
                         qa_item.get('output'))
                domain = (qa_item.get('domain') or 
                         qa_item.get('task_type') or
                         (qa_item.get('metadata', {}).get('task') if isinstance(qa_item.get('metadata'), dict) else None) or
                         'unknown')
                
                if question and answer:
                    qa_pairs.append({
                        "domain": domain,
                        "question": question,
                        "answer": answer
                    })
            
            if qa_pairs:
                protein_data = {
                    "accession": accession,
                    "sequence": protein.get("sequence", ""),
                    "organism": protein.get("organism", "Unknown"),
                    "protein_name": protein.get("protein_name", "Unknown"),
                    "qa_pairs": qa_pairs
                }
                dataset.append(protein_data)
        
        print(f"Created QA pairs for {len(dataset)} proteins, skipped {skipped_count} without QA data")
        return dataset
    
    def fetch_pdb_structure(self, accession: str) -> Optional[Dict]:
        """Fetch PDB structure from UniProt accession and extract structure features.
        
        Returns a dict with structure features or None if structure not available.
        """
        try:
            # Query UniProt for PDB cross-references
            url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract PDB IDs from cross-references
            pdb_ids = []
            for xref in data.get('uniProtKBCrossReferences', []):
                if xref.get('database') == 'PDB':
                    pdb_id = xref.get('id', '')
                    if pdb_id:
                        pdb_ids.append(pdb_id)
            
            if not pdb_ids:
                return None
            
            # Use first available PDB structure
            pdb_id = pdb_ids[0]
            
            # Fetch structure from PDB
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_resp = requests.get(pdb_url, timeout=10)
            pdb_resp.raise_for_status()
            
            # Return raw PDB string directly (will be tokenized with encoder in preprocess_data)
            return {
                'pdb_string': pdb_resp.text,
                'pdb_id': pdb_id
            }
            
        except Exception as e:
            return None
    
    def add_structure_to_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], int]:
        """Enrich dataset with PDB structure information.
        
        Returns (enriched_dataset, skipped_count)
        """
        print("Fetching PDB structures for proteins...")
        enriched_dataset = []
        skipped_count = 0
        
        for item in tqdm(dataset, desc="Fetching structures"):
            accession = item['accession']
            
            # Fetch structure
            structure = self.fetch_pdb_structure(accession)
            
            if structure is None:
                skipped_count += 1
                continue
            
            # Add structure to item
            item['structure'] = structure
            enriched_dataset.append(item)
        
        print(f"✓ Added structures to {len(enriched_dataset)} proteins")
        print(f"  Skipped {skipped_count} proteins without valid structures")
        return enriched_dataset, skipped_count
    
    def save_dataset(self, dataset: List[Dict], split: str = "train"):
        """Save dataset to file, converting numpy arrays to lists"""
        output_file = self.output_dir / f"pfud_{split}.json"
        print(f"Saving {split} split to {output_file}...")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays to lists"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        # Convert dataset
        serializable_dataset = convert_to_serializable(dataset)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
        
        return output_file
    
    def split_dataset(self, dataset: List[Dict], 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test"""
        total = len(dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def prepare(self, num_samples: int = 100):
        """Main preparation pipeline
        
        Always fetches and includes PDB structures for all proteins.
        
        Args:
            num_samples: Number of fully prepared proteins (with structures and QA pairs) to create
                        Defaults to 100
        """
        print("=" * 60)
        print("Preparing PFUD Dataset")
        print("=" * 60)
        
        # Use the combined pipeline (always fetches structures)
        dataset = self.prepare_proteins(num_samples)
        
        print(f"✓ Prepared {len(dataset)} proteins")
        
        train_data, val_data, test_data = self.split_dataset(dataset)
        print(f"✓ Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        self.save_dataset(train_data, "train")
        self.save_dataset(val_data, "val")
        self.save_dataset(test_data, "test")
        
        metadata = {
            "total_samples": len(dataset),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "output_dir": str(self.output_dir),
            "dataset_type": "PFUD",
            "has_structures": True
        }
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  Output directory: {self.output_dir}")
        
        return self.output_dir