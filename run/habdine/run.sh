bash run/habdine/run_train_clip.sh --model llama --k 1024
bash run/habdine/run_train_lora.sh --model llama --k 1024
bash run/habdine/run_inference.sh --model llama --k 1024 --input data/habdine/evaluation/llama/K1024/test_data.json