#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
from pathlib import Path
import json
import pandas as pd

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

def find_top_results(num_results=2):
    """Find the top results based on validation accuracy"""
    results = []
    
    # Find all trainer_state.json files in checkpoint directories
    for state_file in OUTPUTS_DIR.glob("**/checkpoint-*/trainer_state.json"):
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                # Get the best validation accuracy
                val_accuracy = data.get('best_metric', 0)
                # Get the model directory (parent of the checkpoint directory)
                model_dir = state_file.parent.parent
                results.append({
                    'model_dir': str(model_dir),
                    'val_accuracy': val_accuracy
                })
        except Exception as e:
            print(f"Error reading {state_file}: {e}")
    
    # Sort by validation accuracy and get top results
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    return results[:num_results]

def run_predictions(model_dir):
    """Run predictions for a given model directory"""
    print(f"\nRunning predictions for model: {model_dir}")
    
    # Create submission filename based on model directory name
    model_name = Path(model_dir).name
    submission_file = SUBMISSIONS_DIR / f"submission_{model_name}.csv"
    
    # Run the predict.py script
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "predict.py"),
        "--model_dir", model_dir,
        "--batch_size", "128"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Predictions completed for {model_dir}")
        print(f"Submission file should be at: {submission_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running predictions for {model_dir}: {e}")

def main():
    print("Finding top results...")
    top_results = find_top_results()
    
    if not top_results:
        print("No results found!")
        return
    
    print("\nTop results found:")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. Model: {result['model_dir']}")
        print(f"   Validation Accuracy: {result['val_accuracy']:.4f}")
    
    print("\nRunning predictions for top models...")
    for result in top_results:
        run_predictions(result['model_dir'])

if __name__ == "__main__":
    main() 