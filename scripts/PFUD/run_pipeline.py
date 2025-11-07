#!/usr/bin/env python3
"""
End-to-end pipeline runner for PFUD dataset preparation, preprocessing, and inference.
This script orchestrates the complete workflow from raw data to predictions.
Uses direct function calls instead of subprocess for better performance and debugging.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List

# Import pipeline components directly
from prepare_data import PFUDPreparer
from preprocess_data import PFUDPreprocessor
from inference_model import ProtTeXInference

import warnings
warnings.filterwarnings("ignore")

def arg_parse():
    parser = argparse.ArgumentParser(description='Run complete PFUD pipeline')
    parser.add_argument('--step', default='all', 
                        choices=['prepare', 'preprocess', 'inference', 'all'],
                        help='Which step(s) to run')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process')
    parser.add_argument('--model_path', default='./model/ProtTeX',
                        help='Path to ProtTeX model')
    parser.add_argument('--data_root', default='./data',
                        help='Root data directory')
    parser.add_argument('--mol_instructions_path', default=None,
                        help='Path to Mol-Instructions dataset JSON file')
    parser.add_argument('--protein_lmbench_path', default=None,
                        help='Path to ProteinLMBench dataset JSON file')
    args = parser.parse_args()
    return args


class PipelineRunner:
    """Orchestrates the complete PFUD pipeline"""
    
    def __init__(self, data_root: str, model_path: str, num_samples: int = 100,
                 mol_instructions_path: str = None, protein_lmbench_path: str = None):
        self.data_root = Path(data_root)
        self.model_path = Path(model_path)
        self.num_samples = num_samples
        self.mol_instructions_path = mol_instructions_path
        self.protein_lmbench_path = protein_lmbench_path
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        self.dirs = {
            'raw': self.data_root / 'PFUD',
            'tokenized': self.data_root / 'PFUD_tokenized',
            'results': self.data_root / '../results/PFUD_predictions'
        }
    
    def run_step(self, step_name: str) -> bool:
        """Run a pipeline step (placeholder for different steps)"""
        print(f"\n{'='*60}")
        print(f"Running step: {step_name}")
        print(f"{'='*60}\n")
        
        try:
            if step_name == 'Data Preparation':
                self._run_prepare()
            elif step_name == 'Data Preprocessing':
                self._run_preprocess()
            elif step_name == 'Model Inference':
                self._run_inference()
            
            print(f"✓ {step_name} completed successfully")
            return True
        except Exception as e:
            print(f"✗ {step_name} failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_prepare(self):
        """Execute data preparation step"""
        preparer = PFUDPreparer(
            output_dir=str(self.dirs['raw']),
            cache_dir=str(self.data_root / 'cache'),
            mol_instructions_path=self.mol_instructions_path,
            protein_lmbench_path=self.protein_lmbench_path
        )
        preparer.prepare(self.num_samples)
    
    def _run_preprocess(self):
        """Execute data preprocessing step"""
        preprocessor = PFUDPreprocessor(
            output_dir=str(self.dirs['tokenized']),
            aa_dict_path='./tokenizer_metadata/character_aa_dict.pkl',
            protoken_dict_path='./tokenizer_metadata/character.json'
        )
        preprocessor.preprocess(self.dirs['raw'], split='all')
    
    def _run_inference(self):
        """Execute inference step"""
        inference = ProtTeXInference(
            model_path=str(self.model_path),
            output_dir=str(self.dirs['results']),
            device='cuda' if self._cuda_available() else 'cpu'
        )
        split = 'val'
        results = inference.run_inference(
            data_path=self.dirs['tokenized'],
            split=split,
            max_new_tokens=256,
            num_samples=-1
        )
        # Save results to JSON
        inference.save_results(results, split=split)
        return results
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def run_prepare(self, num_samples: int) -> bool:
        """Step 1: Prepare raw PFUD data"""
        return self.run_step('Data Preparation')
    
    def run_preprocess(self) -> bool:
        """Step 2: Preprocess with ProtTeX tokenizer"""
        return self.run_step('Data Preprocessing')
    
    def run_inference(self, num_samples: int = -1) -> bool:
        """Step 3: Run inference"""
        return self.run_step('Model Inference')
    
    def print_summary(self, success: Dict[str, bool]):
        """Print pipeline summary"""
        print(f"\n{'='*60}")
        print("Pipeline Summary")
        print(f"{'='*60}\n")
        
        for step, completed in success.items():
            status = "✓ SUCCESS" if completed else "✗ FAILED"
            print(f"{step:20} {status}")
        
        print(f"\nData locations:")
        print(f"  Raw data:        {self.dirs['raw']}")
        print(f"  Tokenized data:  {self.dirs['tokenized']}")
        print(f"  Results:         {self.dirs['results']}")
    
    def run(self, steps: List[str], num_samples: int):
        """Run selected pipeline steps"""
        print(f"\nStarting PFUD Pipeline")
        print(f"Steps to run: {', '.join(steps)}")
        print(f"Number of samples: {num_samples}")
        
        success = {}
        
        if 'prepare' in steps:
            success['Data Preparation'] = self.run_prepare(num_samples)
            if not success['Data Preparation']:
                print("Stopping pipeline due to prepare failure")
                self.print_summary(success)
                return False
        
        if 'preprocess' in steps:
            success['Data Preprocessing'] = self.run_preprocess()
            if not success['Data Preprocessing']:
                print("Stopping pipeline due to preprocess failure")
                self.print_summary(success)
                return False
        
        if 'inference' in steps:
            success['Model Inference'] = self.run_inference(num_samples)
        
        self.print_summary(success)
        
        all_success = all(success.values())
        return all_success


def main():
    args = arg_parse()
    
    # Determine steps to run
    if args.step == 'all':
        steps = ['prepare', 'preprocess', 'inference']
    else:
        steps = [args.step]
    
    # Run pipeline
    runner = PipelineRunner(
        data_root=args.data_root,
        model_path=args.model_path,
        num_samples=args.num_samples,
        mol_instructions_path="../Mol-Instructions/data/Protein-oriented_Instructions",
        protein_lmbench_path=args.protein_lmbench_path
    )
    
    success = runner.run(steps, args.num_samples)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
