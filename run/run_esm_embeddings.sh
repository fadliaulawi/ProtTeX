#!/bin/bash
# Run 72 batches across 8 GPUs in parallel

# Create logs directory
mkdir -p logs

echo "Starting parallel processing on 8 GPUs..."
echo "Monitor progress with: tail -f logs/gpu*.log"

# GPU 0: batches 0-8
(
  for i in {0..8}; do
    echo "GPU 0: Processing batch $i" >> logs/gpu0.log
    CUDA_VISIBLE_DEVICES=0 python run/01_extract_esm_embeddings.py $i >> logs/gpu0.log 2>&1
  done
  echo "GPU 0: DONE!" >> logs/gpu0.log
) &

# GPU 1: batches 9-17
(
  for i in {9..17}; do
    echo "GPU 1: Processing batch $i" >> logs/gpu1.log
    CUDA_VISIBLE_DEVICES=1 python run/01_extract_esm_embeddings.py $i >> logs/gpu1.log 2>&1
  done
  echo "GPU 1: DONE!" >> logs/gpu1.log
) &

# GPU 2: batches 18-26
(
  for i in {18..26}; do
    echo "GPU 2: Processing batch $i" >> logs/gpu2.log
    CUDA_VISIBLE_DEVICES=2 python run/01_extract_esm_embeddings.py $i >> logs/gpu2.log 2>&1
  done
  echo "GPU 2: DONE!" >> logs/gpu2.log
) &

# GPU 3: batches 27-35
(
  for i in {27..35}; do
    echo "GPU 3: Processing batch $i" >> logs/gpu3.log
    CUDA_VISIBLE_DEVICES=3 python run/01_extract_esm_embeddings.py $i >> logs/gpu3.log 2>&1
  done
  echo "GPU 3: DONE!" >> logs/gpu3.log
) &

# GPU 4: batches 36-44
(
  for i in {36..44}; do
    echo "GPU 4: Processing batch $i" >> logs/gpu4.log
    CUDA_VISIBLE_DEVICES=4 python run/01_extract_esm_embeddings.py $i >> logs/gpu4.log 2>&1
  done
  echo "GPU 4: DONE!" >> logs/gpu4.log
) &

# GPU 5: batches 45-53
(
  for i in {45..53}; do
    echo "GPU 5: Processing batch $i" >> logs/gpu5.log
    CUDA_VISIBLE_DEVICES=5 python run/01_extract_esm_embeddings.py $i >> logs/gpu5.log 2>&1
  done
  echo "GPU 5: DONE!" >> logs/gpu5.log
) &

# GPU 6: batches 54-62
(
  for i in {54..62}; do
    echo "GPU 6: Processing batch $i" >> logs/gpu6.log
    CUDA_VISIBLE_DEVICES=6 python run/01_extract_esm_embeddings.py $i >> logs/gpu6.log 2>&1
  done
  echo "GPU 6: DONE!" >> logs/gpu6.log
) &

# GPU 7: batches 63-71
(
  for i in {63..71}; do
    echo "GPU 7: Processing batch $i" >> logs/gpu7.log
    CUDA_VISIBLE_DEVICES=7 python run/01_extract_esm_embeddings.py $i >> logs/gpu7.log 2>&1
  done
  echo "GPU 7: DONE!" >> logs/gpu7.log
) &

# Wait for all GPUs to finish
wait

echo "✅ All 72 batches completed!"
echo "Check logs in: logs/gpu*.log"
