#!/usr/bin/env python3
"""Inference script for ProtTeX model on PFUD task."""

import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProtTeXInference:
    """Run inference with ProtTeX model"""
    
    def __init__(self, model_path: str, output_dir: str, device: str = 'cpu'):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained ProtTeX model"""
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map=self.device if self.device == 'cuda' else None,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("✓ Model loaded successfully")
        
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response using model"""
        
        try:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding for deterministic results
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=1.0,
                    top_p=1.0
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            print(f"⚠ Error during generation: {e}")
            raise
    
    def run_inference(self, data_path: Path, split: str, max_new_tokens: int, 
                      num_samples: int = -1) -> List[Dict]:
        """Run inference on a dataset split"""
        
        # Load tokenized data
        input_file = data_path / f"pfud_{split}_tokenized.pkl"
        
        if not input_file.exists():
            print(f"⚠ File not found: {input_file}")
            return []
        
        print(f"\nLoading tokenized data from {input_file}...")
        with open(input_file, 'rb') as f:
            data = pkl.load(f)
        
        if num_samples > 0:
            data = data[:num_samples]
        
        print(f"Running inference on {len(data)} samples...")
        
        results = []
        for example in tqdm(data, desc=f"Inference ({split})"):
            prompt = example['prompt']
            answer = example['answer']
            accession = example['accession']
            
            # Generate prediction
            prediction = self.generate_response(prompt, max_new_tokens)
            
            result = {
                'accession': accession,
                'prompt': prompt,
                'ground_truth': answer,
                'prediction': prediction,
                'split': split
            }
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], split: str) -> Path:
        """Save inference results to JSON"""
        
        json_file = self.output_dir / f"predictions_{split}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved {len(results)} predictions to {json_file}")
        return json_file
