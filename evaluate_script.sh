for i in $(seq 0 37)
do
    echo "Evaluating batch $i"
    start_time=$(date +%s)
    
    # Run the evaluation script for each batch
    python evaluate_codebook.py --codebook esmfold_tokenizer/data/UniProt_Function/structure_codebook_K128.pkl --embeddings esmfold_tokenizer/data/UniProt_Function/raw_esm_embeddings/esm_embeddings_batch_$i.npy --proteins esmfold_tokenizer/data/UniProt_Function/sample_proteins/sample_proteins_batch_$i.json
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "✅ Batch $i completed in ${elapsed}s"
    echo "---"
done