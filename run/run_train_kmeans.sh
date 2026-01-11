#!/bin/bash
# Train k-means codebooks with different K values in parallel on 4 GPUs

cd /adialab/usr/shadabk/prottex

# Create logs directory
mkdir -p logs

rm -f logs/kmeans/gpu_*.log

echo "Starting parallel k-means training on 4 GPUs..."
echo "Training K values: 32 to 2048"
echo "Monitor progress with: tail -f logs/kmeans/gpu_*.log"

# # GPU 0: K=32
# (
#   echo "GPU 0: Training K=32" | tee logs/kmeans/gpu_32.log
#   CUDA_VISIBLE_DEVICES=0 python -u run/02_train_kmeans_codebook.py --k 32 >> logs/kmeans/gpu_32.log 2>&1
#   echo "GPU 0: K=32 DONE!" >> logs/kmeans/gpu_32.log
# ) &

# # GPU 1: K=64
# (
#   echo "GPU 1: Training K=64" | tee logs/kmeans/gpu_64.log
#   CUDA_VISIBLE_DEVICES=1 python -u run/02_train_kmeans_codebook.py --k 64 >> logs/kmeans/gpu_64.log 2>&1
#   echo "GPU 1: K=64 DONE!" >> logs/kmeans/gpu_64.log
# ) &

# GPU 2: K=128
(
  echo "GPU 2: Training K=128" | tee logs/kmeans/gpu_128.log
  CUDA_VISIBLE_DEVICES=2 python -u run/02_train_kmeans_codebook.py --k 128 >> logs/kmeans/gpu_128.log 2>&1
  echo "GPU 2: K=128 DONE!" >> logs/kmeans/gpu_128.log
) &

# GPU 3: K=256
(
  echo "GPU 3: Training K=256" | tee logs/kmeans/gpu_256.log
  CUDA_VISIBLE_DEVICES=3 python -u run/02_train_kmeans_codebook.py --k 256 >> logs/kmeans/gpu_256.log 2>&1
  echo "GPU 3: K=256 DONE!" >> logs/kmeans/gpu_256.log
) &

# GPU 4: K=512
(
  echo "GPU 4: Training K=512" | tee logs/kmeans/gpu_512.log
  CUDA_VISIBLE_DEVICES=4 python -u run/02_train_kmeans_codebook.py --k 512 >> logs/kmeans/gpu_512.log 2>&1
  echo "GPU 4: K=512 DONE!" >> logs/kmeans/gpu_512.log
) &

# GPU 5: K=1024
(
  echo "GPU 5: Training K=1024" | tee logs/kmeans/gpu_1024.log
  CUDA_VISIBLE_DEVICES=5 python -u run/02_train_kmeans_codebook.py --k 1024 >> logs/kmeans/gpu_1024.log 2>&1
  echo "GPU 5: K=1024 DONE!" >> logs/kmeans/gpu_1024.log
) &

# GPU 6: K=2048
(
  echo "GPU 6: Training K=2048" | tee logs/kmeans/gpu_2048.log
  CUDA_VISIBLE_DEVICES=6 python -u run/02_train_kmeans_codebook.py --k 2048 >> logs/kmeans/gpu_2048.log 2>&1
  echo "GPU 6: K=2048 DONE!" >> logs/kmeans/gpu_2048.log
) &

# Wait for all to finish
wait

echo "✅ All k-means training completed!"
echo "Output files:"
echo "  - data/codebooks/structure_codebook_K*.pkl"
echo ""
echo "Compare results with:"
echo "  python run/02b_compare_codebooks.py --all"
