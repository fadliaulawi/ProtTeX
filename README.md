# ESMFold/ESM-2 Tokenizer Approach

Alternative tokenizer using pre-trained ESM-2 embeddings instead of hand-crafted 6D features.

## Status: Proof of Concept Phase

**Goal:** Extract ESM-2 embeddings and test if clustering works better than 6D approach.

---

## Quick Start (30 minutes)

### Step 1: Extract Embeddings (10-15 min)
```bash
cd /lustrefs/shared/mohammad.sayeed/Prot2Text/esmfold_tokenizer
python extract_esm_embeddings_sample.py
```

**What it does:**
- Loads 10 proteins from AlphaFold dataset
- Extracts per-residue embeddings using ESM-2 650M
- Saves to `sample_embeddings.npy`

**Requirements:**
- transformers
- torch
- biopython

### Step 2: Test Clustering (5-10 min)
```bash
python test_clustering_esm.py
```

**What it does:**
- Loads embeddings
- Runs k-means with K=64 clusters
- Creates PCA visualization
- Saves codebook

**Outputs:**
- `esm_codebook_K64_sample.pkl`
- `esm_clustering_viz.png`
- Cluster statistics

### Step 3: Review Results (5 min)
```bash
# Check visualization
display esm_clustering_viz.png

# Compare to 6D approach
cat sample_metadata.txt
```

---

## Comparison: 6D vs ESM-2

| Aspect | 6D Hand-Crafted | ESM-2 Embeddings |
|--------|-----------------|------------------|
| **Features** | φ, ψ, ω, SS, RSA, density | 1280-dim learned |
| **Pre-training** | None | 250M sequences |
| **Dimension** | 6 | 1280 |
| **Speed** | ⚡ Very fast | 🐌 GPU needed |
| **Interpretability** | ✅ Clear | ❌ Black box |
| **Expected quality** | Good | Better |

---

## Files

- `30MIN_PLAN.md` - Detailed 30-minute action plan
- `extract_esm_embeddings_sample.py` - Extract embeddings from 10 proteins
- `test_clustering_esm.py` - k-means clustering test
- `README.md` - This file

---

## Next Steps After 30-Min Test

### If Successful:
1. Scale to full dataset (8,360 proteins)
2. Train K=512 codebook
3. Compare performance on downstream tasks

### If Issues:
1. Debug environment
2. Try smaller model (ESM-2 150M)
3. Fall back to 6D for now

---

## Full Dataset Pipeline (Future)

```bash
# Extract embeddings from all proteins (~2 hours on A100)
python extract_esm_embeddings_full.py --num_proteins 8360

# Train full codebook with K=512
python train_esm_codebook.py --num_clusters 512

# Tokenize proteins
python tokenize_with_esm.py --input proteins.fasta --output tokens.json

# Fine-tune LLM
python train_llm.py --tokenizer esm_codebook.pkl
```

---

## Model Options

### ESM-2 Models (Recommended)
- `facebook/esm2_t6_8M_UR50D` - 8M params, very fast
- `facebook/esm2_t12_35M_UR50D` - 35M params, fast
- `facebook/esm2_t30_150M_UR50D` - 150M params, good balance
- `facebook/esm2_t33_650M_UR50D` - 650M params, best quality ⭐ (using this)
- `facebook/esm2_t36_3B_UR50D` - 3B params, highest quality (slow)

### ESMFold (Not using for now)
- Slower (includes structure prediction)
- Overkill for just embeddings
- Use ESM-2 instead

---

## Resources Needed

### Sample Test (10 proteins):
- Time: 10-15 minutes
- GPU: Any CUDA GPU (8GB+ VRAM)
- Storage: ~50 MB

### Full Dataset (8,360 proteins):
- Time: 2-3 hours
- GPU: A100 or similar (40GB+ VRAM)
- Storage: ~30 GB




