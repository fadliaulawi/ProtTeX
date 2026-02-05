#!/usr/bin/env python3
"""
Evaluation Script for ESM-LLAMA Inference Results
Computes BLEU, ROUGE, and embedding similarity metrics.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
import argparse
import warnings
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# ROUGE
from rouge_score import rouge_scorer

# BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available for BLEU computation. Install with: pip install nltk")
    NLTK_AVAILABLE = False

# Embedding models
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


def compute_bleu(reference: str, prediction: str, n: int = 4) -> float:
    """Compute BLEU-n score"""
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            return 0.0
        
        # Create n-gram references
        if n == 2:
            weights = (0.5, 0.5)
        elif n == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        else:
            weights = (1.0 / n,) * n
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smoothing)
        return score
    except Exception as e:
        return 0.0


def compute_rouge(reference: str, prediction: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def load_embedding_model(model_name: str, device='cuda'):
    """Load embedding model for semantic similarity"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"⚠️  Warning: Failed to load {model_name}: {e}")
        return None, None


def compute_embedding_similarity(text1: str, text2: str, tokenizer, model, device='cuda') -> float:
    """Compute cosine similarity between embeddings of two texts"""
    if tokenizer is None or model is None:
        return 0.0
    
    try:
        # Tokenize and encode
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # Get embeddings (mean pool over sequence)
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
            
            # Mean pool
            emb1 = outputs1.last_hidden_state.mean(dim=1).cpu().numpy()
            emb2 = outputs2.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    except Exception as e:
        return 0.0


def evaluate_predictions(predictions_dict: Dict,
                        roberta_tokenizer=None, roberta_model=None,
                        biobert_tokenizer=None, biobert_model=None,
                        device='cuda') -> Dict:
    """Evaluate all predictions and return annotated predictions with metrics"""
    
    results = []
    
    metric_lists = {
        'bleu2': [],
        'bleu4': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'roberta_similarity': [],
        'biobert_similarity': []
    }
    
    print("Computing metrics...")
    for name, data in tqdm(predictions_dict.items(), desc="Evaluating"):
        reference = data.get('true', '').strip()
        prediction = data.get('pred', '').strip()
        
        # Create result item
        result_item = {
            'name': name,
            'ground_truth': reference,
            'prediction': prediction
        }
        
        # Skip empty predictions or ground truth
        if not reference or not prediction:
            result_item['metrics'] = None
            results.append(result_item)
            continue
        
        # Compute all metrics for this sample
        metrics = {}
        
        # BLEU scores
        if NLTK_AVAILABLE:
            metrics['bleu2'] = compute_bleu(reference, prediction, n=2)
            metrics['bleu4'] = compute_bleu(reference, prediction, n=4)
            metric_lists['bleu2'].append(metrics['bleu2'])
            metric_lists['bleu4'].append(metrics['bleu4'])
        
        # ROUGE scores
        rouge_scores = compute_rouge(reference, prediction)
        metrics['rouge1'] = rouge_scores['rouge1']
        metrics['rouge2'] = rouge_scores['rouge2']
        metrics['rougeL'] = rouge_scores['rougeL']
        metric_lists['rouge1'].append(metrics['rouge1'])
        metric_lists['rouge2'].append(metrics['rouge2'])
        metric_lists['rougeL'].append(metrics['rougeL'])
        
        # Embedding similarities
        if roberta_tokenizer and roberta_model:
            metrics['roberta_similarity'] = compute_embedding_similarity(
                reference, prediction, roberta_tokenizer, roberta_model, device
            )
            metric_lists['roberta_similarity'].append(metrics['roberta_similarity'])
        
        if biobert_tokenizer and biobert_model:
            metrics['biobert_similarity'] = compute_embedding_similarity(
                reference, prediction, biobert_tokenizer, biobert_model, device
            )
            metric_lists['biobert_similarity'].append(metrics['biobert_similarity'])
        
        # Append metrics to item
        result_item['metrics'] = metrics
        results.append(result_item)
    
    # Compute summary statistics
    summary = {}
    for metric, values in metric_lists.items():
        if values:
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_min'] = float(np.min(values))
            summary[f'{metric}_max'] = float(np.max(values))
        else:
            summary[f'{metric}_mean'] = 0.0
            summary[f'{metric}_std'] = 0.0
            summary[f'{metric}_min'] = 0.0
            summary[f'{metric}_max'] = 0.0
    
    summary['total_samples'] = len(predictions_dict)
    summary['valid_samples'] = len([r for r in results if r.get('metrics') is not None])
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate ESM-LLAMA inference results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with predictions (combined_predictions.json)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for evaluation results')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  CUDA not available, using CPU")
        raise RuntimeError("This script requires a CUDA-capable GPU to run efficiently.")
    
    # Load predictions
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    print(f"📥 Loading predictions from {input_path}...")
    with open(input_path, 'r') as f:
        predictions_dict = json.load(f)
    
    print(f"✅ Loaded {len(predictions_dict)} predictions")
    
    # Load embedding models
    print("\n📥 Loading embedding models...")
    
    roberta_tokenizer, roberta_model = load_embedding_model('roberta-base', device)
    if roberta_tokenizer:
        print("✅ RoBERTa loaded")
    else:
        print("⚠️  RoBERTa not available")
    
    biobert_tokenizer, biobert_model = load_embedding_model('dmis-lab/biobert-base-cased-v1.2', device)
    if biobert_tokenizer:
        print("✅ BioBERT loaded")
    else:
        print("⚠️  BioBERT not available (trying alternative)")
        # Try alternative BioBERT
        biobert_tokenizer, biobert_model = load_embedding_model('monologg/biobert_v1.1_pubmed', device)
        if biobert_tokenizer:
            print("✅ BioBERT (alternative) loaded")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    results, summary = evaluate_predictions(
        predictions_dict,
        roberta_tokenizer=roberta_tokenizer,
        roberta_model=roberta_model,
        biobert_tokenizer=biobert_tokenizer,
        biobert_model=biobert_model,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTotal samples: {summary['total_samples']}")
    print(f"Valid samples: {summary['valid_samples']}")
    
    if NLTK_AVAILABLE:
        print(f"\n📊 BLEU Scores:")
        print(f"   BLEU-2: {summary['bleu2_mean']:.4f} ± {summary['bleu2_std']:.4f}")
        print(f"   BLEU-4: {summary['bleu4_mean']:.4f} ± {summary['bleu4_std']:.4f}")
    
    print(f"\n📊 ROUGE Scores:")
    print(f"   ROUGE-1: {summary['rouge1_mean']:.4f} ± {summary['rouge1_std']:.4f}")
    print(f"   ROUGE-2: {summary['rouge2_mean']:.4f} ± {summary['rouge2_std']:.4f}")
    print(f"   ROUGE-L: {summary['rougeL_mean']:.4f} ± {summary['rougeL_std']:.4f}")
    
    if roberta_tokenizer:
        print(f"\n📊 Embedding Similarities:")
        print(f"   RoBERTa: {summary['roberta_similarity_mean']:.4f} ± {summary['roberta_similarity_std']:.4f}")
        if biobert_tokenizer:
            print(f"   BioBERT: {summary['biobert_similarity_mean']:.4f} ± {summary['biobert_similarity_std']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'results': results,
        'summary': summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
