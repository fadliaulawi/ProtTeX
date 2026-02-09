#!/usr/bin/env python3
"""
Precompute per-token protein embeddings (ESM + ESMFold + K-means) for train/val/test.
Always saves .npz (flat arrays + lengths, compressed). Run once, then train_prot3text_v3.py loads them.
"""

import os
import argparse
import glob
import pickle

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, EsmForProteinFolding

# Hardcoded (match train_prot3text_v2 / v3)
ESM_PATH = "facebook/esm2_t36_3B_UR50D"
ESMFOLD_PATH = "facebook/esmfold_v1"
MAX_SEQUENCE_LENGTH = 1021
CODEBOOK_DIR = "data/codebooks"
CODEBOOK_K = 1024

# Chunk size for saving: each split is saved as split_000.npz, split_001.npz, ...
# Smaller = less RAM (out_list held in memory until chunk is full). Avoid node OOM; 2k–5k is safe.
# 240k train -> 120 chunks of 2k, or 48 chunks of 5k.
SAMPLES_PER_CHUNK = 10000

def parse_args():
    p = argparse.ArgumentParser(description="Precompute ESM + ESMFold embeddings for Prot3Text v3")
    p.add_argument("--root_csv_dir", type=str, default="data/csv/", help="Dir with train.csv, validation.csv, test.csv")
    p.add_argument("--output_dir", type=str, default="/nfs-stor/fadli.ghiffari/embeddings/", help="Where to save .npz files")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding")
    p.add_argument("--trim", type=int, default=None, help="If set, only process first N rows per split (debug)")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ESM tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_PATH)
    esm = EsmModel.from_pretrained(
        ESM_PATH,
        add_pooling_layer=False,
        torch_dtype=torch.bfloat16,
    ).to(device)
    esm.eval()

    print("Loading ESMFold...")
    esmfold = EsmForProteinFolding.from_pretrained(
        ESMFOLD_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device)
    esmfold.eval()
    for p in esmfold.parameters():
        p.requires_grad = False

    print("Loading codebook...")
    codebook_path = os.path.join(CODEBOOK_DIR, f"structure_codebook_K{CODEBOOK_K}.pkl")
    with open(codebook_path, "rb") as f:
        codebook_data = pickle.load(f)
    centroids = torch.from_numpy(codebook_data["kmeans"].cluster_centers_).to(
        dtype=torch.bfloat16, device=device
    )

    max_len = MAX_SEQUENCE_LENGTH + 2  # BOS + tokens + EOS

    def encode_split(split_name: str, csv_name: str):
        csv_path = os.path.join(args.root_csv_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"Skip {split_name}: {csv_path} not found")
            return
        df = pd.read_csv(csv_path)
        if args.trim:
            df = df.head(args.trim)
        sequences = df["sequence"].astype(str).tolist()
        for i in range(len(sequences)):
            if len(sequences[i]) > MAX_SEQUENCE_LENGTH:
                sequences[i] = sequences[i][: MAX_SEQUENCE_LENGTH]

        # Resume: detect existing chunks (split_000.npz, split_001.npz, ...) and continue from next
        existing_chunks = sorted(glob.glob(os.path.join(args.output_dir, f"{split_name}_*.npz")))
        if existing_chunks:
            total_saved = sum(len(np.load(p, allow_pickle=False)["lengths"]) for p in existing_chunks)
            chunk_idx = len(existing_chunks)
            sequences = sequences[total_saved:]
            if not sequences:
                print(f"  {split_name}: already complete ({total_saved} samples in {chunk_idx} chunk(s))")
                return
            print(f"  Resuming {split_name}: {total_saved} samples in {chunk_idx} chunk(s), encoding remaining {len(sequences)}")
        else:
            total_saved = 0
            chunk_idx = 0

        out_list = []

        def write_chunk(items):
            if not items:
                return
            path = os.path.join(args.output_dir, f"{split_name}_{chunk_idx:03d}.npz")
            print(f"  Writing {path} ({len(items)} samples, compressing...) ", end="", flush=True)
            lengths = np.array([items[j]["seq"].size(0) for j in range(len(items))], dtype=np.int32)
            seq_flat = np.concatenate([items[j]["seq"].float().numpy().astype(np.float16) for j in range(len(items))], axis=0)
            struct_flat = np.concatenate([items[j]["struct"].float().numpy().astype(np.float16) for j in range(len(items))], axis=0)
            np.savez_compressed(path, seq_flat=seq_flat, struct_flat=struct_flat, lengths=lengths)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"done ({size_mb:.1f} MB)", flush=True)
            return size_mb

        for start in tqdm(range(0, len(sequences), args.batch_size), desc=split_name):
            batch_seqs = sequences[start : start + args.batch_size]
            tokenizer.padding_side = "right"
            tok = tokenizer(
                batch_seqs,
                truncation=True,
                padding="longest",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)

            with torch.no_grad():
                esm_out = esm(input_ids=input_ids, attention_mask=attention_mask)
                seq_h = esm_out.last_hidden_state

                esmfold_out = esmfold.esm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                struct_h = esmfold_out.hidden_states[-1]

                B, L, D = struct_h.shape
                flat = struct_h.reshape(-1, D)
                dists = torch.cdist(flat.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
                ids = torch.argmin(dists, dim=1)
                struct_h = centroids[ids].view(B, L, D)

            for i in range(seq_h.size(0)):
                n = attention_mask[i].sum().item()
                out_list.append({
                    "seq": seq_h[i, :n].cpu().to(torch.bfloat16),
                    "struct": struct_h[i, :n].cpu().to(torch.bfloat16),
                })

            while len(out_list) >= SAMPLES_PER_CHUNK:
                chunk_items = out_list[:SAMPLES_PER_CHUNK]
                out_list = out_list[SAMPLES_PER_CHUNK:]
                size_mb = write_chunk(chunk_items)
                total_saved += len(chunk_items)
                chunk_idx += 1
                print(f"  {split_name}_{chunk_idx - 1:03d}.npz: {len(chunk_items)} samples ({size_mb:.1f} MB)")

        if out_list:
            size_mb = write_chunk(out_list)
            total_saved += len(out_list)
            chunk_idx += 1
            print(f"  {split_name}_{chunk_idx - 1:03d}.npz: {len(out_list)} samples ({size_mb:.1f} MB)")
        print(f"  Total: {total_saved} samples in {chunk_idx} chunk(s)")

    encode_split("validation", "validation.csv")
    encode_split("test", "test.csv")
    encode_split("train", "train.csv")
    print("Done.")


if __name__ == "__main__":
    main()
