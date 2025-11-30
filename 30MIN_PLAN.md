# 30-Minute Action Plan: ESMFold Tokenizer

## Status: Starting Fresh
**Time Available:** 30 minutes
**Goal:** Get ESMFold embeddings extraction working on sample data

---

## ✅ Minute 0-5: Environment Check

```bash
# Check if transformers is installed
python -c "from transformers import EsmModel; print('✅ ESM available')"

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Check available models
python -c "from transformers import AutoTokenizer; print('✅ Ready')"
```

**Action:** Verify dependencies, install if needed

---

## ✅ Minute 5-15: Extract Embeddings (Sample)

**Script:** `extract_esm_embeddings_sample.py`

```python
# Extract embeddings from 10 proteins only (PROOF OF CONCEPT)
# Use ESM-2 (faster than ESMFold for testing)
# Save embeddings to .npy file
```

**Input:** 10 random PDB files from your dataset
**Output:** `sample_embeddings.npy` (shape: [~3000 residues, 1280])

**Why ESM-2 instead of ESMFold?**
- Faster (no structure prediction)
- Smaller model (650M vs 15B)
- Still gives rich embeddings
- Good for testing

---

## ✅ Minute 15-25: Quick k-means Test

**Script:** `test_clustering_esm.py`

```python
# Load sample embeddings
# Run k-means with K=64 (small for speed)
# Visualize with PCA
# Compare to 6D approach
```

**Output:** 
- `esm_codebook_K64_sample.pkl`
- `esm_clustering_viz.png`
- Cluster statistics

---

## ✅ Minute 25-30: Compare Results

**Quick Analysis:**
- How many clusters are used? (vs 96.9% in 6D)
- Do embeddings separate better?
- PCA visualization: ESM vs 6D

**Decision Point:**
- Worth scaling up to full dataset?
- Or stick with 6D for now?

---

## Files to Create (in priority order):

1. **extract_esm_embeddings_sample.py** - Extract from 10 proteins
2. **test_clustering_esm.py** - Quick k-means test
3. **compare_6d_vs_esm.py** - Side-by-side comparison
4. **README.md** - What this directory contains

---

## What We WON'T Do (Not enough time):

❌ Full dataset extraction (2 hours on GPU)
❌ Train 512-cluster codebook
❌ LLM integration
❌ Benchmark evaluation

---

## Success Criteria (30 min):

✅ ESM-2 embeddings extracted for 10 proteins
✅ k-means clustering works on embeddings
✅ Visual comparison shows embedding quality
✅ Decision: proceed with full run or not

---

## Next Steps After 30 Min:

**If successful:**
- Scale to full 8,360 proteins (submit as batch job)
- Train K=512 codebook
- Compare performance

**If issues:**
- Debug environment
- Fall back to 6D approach for now
- Plan ESM upgrade for later




