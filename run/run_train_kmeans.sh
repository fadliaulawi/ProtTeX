#!/bin/bash
# Train k-means codebooks with different K values in parallel on 4 GPUs

cd /adialab/usr/shadabk/prottex

# Create logs directory
mkdir -p logs

echo "Starting parallel k-means training on 4 GPUs..."
echo "Training K values: 64, 128, 256, 512"
echo "Monitor progress with: tail -f logs/kmeans_*.log"

# GPU 0: K=64
(
  echo "GPU 0: Training K=64" | tee logs/kmeans_k64.log
  CUDA_VISIBLE_DEVICES=0 python -u run/02_train_kmeans_codebook.py --k 64 >> logs/kmeans_k64.log 2>&1
  echo "GPU 0: K=64 DONE!" >> logs/kmeans_k64.log
) &

# GPU 1: K=1024
(
  echo "GPU 1: Training K=1024" | tee logs/kmeans_k1024.log
  CUDA_VISIBLE_DEVICES=1 python -u run/02_train_kmeans_codebook.py --k 1024 >> logs/kmeans_k1024.log 2>&1
  echo "GPU 1: K=1024 DONE!" >> logs/kmeans_k1024.log
) &

# GPU 2: K=512
(
  echo "GPU 2: Training K=512" | tee logs/kmeans_k512.log
  CUDA_VISIBLE_DEVICES=2 python -u run/02_train_kmeans_codebook.py --k 512 >> logs/kmeans_k512.log 2>&1
  echo "GPU 2: K=512 DONE!" >> logs/kmeans_k512.log
) &

# GPU 3: K=256
(
  echo "GPU 3: Training K=256" | tee logs/kmeans_k256.log
  CUDA_VISIBLE_DEVICES=3 python -u run/02_train_kmeans_codebook.py --k 256 >> logs/kmeans_k256.log 2>&1
  echo "GPU 3: K=256 DONE!" >> logs/kmeans_k256.log
) &

# Wait for all to finish
wait

echo "✅ All k-means training completed!"
echo "Output files:"
echo "  - data/structure_codebook_K64.pkl"
echo "  - data/structure_codebook_K128.pkl"
echo "  - data/structure_codebook_K512.pkl"
echo "  - data/structure_codebook_K256.pkl"
echo ""
echo "Compare results with:"
echo "  python run/02b_compare_codebooks.py --all"
