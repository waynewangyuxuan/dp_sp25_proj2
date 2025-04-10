import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel, PeftConfig
from datetime import datetime
from tqdm import tqdm
import time

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_CSV_FILE = PROJECT_ROOT / "data" / "test" / "test_unlabelled.csv"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
DEFAULT_MODEL_DIR = "/scratch/yw5954/dp_sp25_proj2/outputs/run_20250410_123525_r16_a32_l6_lr0.0002"

# Set cache directory to project folder to avoid disk quota issues
cache_dir = PROJECT_ROOT / "cache"
cache_dir.mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate predictions with a trained LoRA model')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, 
                       help='Directory containing the trained model checkpoints')
    parser.add_argument('--test_file', type=str, default=str(TEST_CSV_FILE),
                       help='Path to the test CSV file')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference (larger values use more memory but are faster)')
    return parser.parse_args()

def load_test_data(test_file):
    """Load the test dataset from CSV file"""
    print(f"Loading test data from {test_file}")
    df = pd.read_csv(test_file)
    
    # Verify the CSV has the expected structure
    if 'text' not in df.columns:
        raise ValueError(f"CSV file {test_file} must contain a 'text' column")
        
    print(f"Loaded {len(df)} test examples")
    # Show a few examples
    print("Sample texts:")
    for i in range(min(3, len(df))):
        print(f"[{i}] {df['text'].iloc[i][:100]}...")
    
    return df['text'].tolist()

def generate_predictions(model, tokenizer, texts, device, batch_size=128):
    """Generate predictions for test data"""
    # Ensure model is in eval mode and on correct device
    model.eval()
    model = model.to(device)
    
    # Preallocate the predictions array
    predictions = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Processing {len(texts)} texts in {total_batches} batches (batch size: {batch_size})")
    
    # Make sure GPU is warmed up
    if device.type == 'cuda':
        print("Warming up GPU with a small batch...")
        warmup_inputs = tokenizer(
            ["Warm up the GPU with a sample text input"], 
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            _ = model(**warmup_inputs)
        print("GPU warm-up complete")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Print memory stats
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
    
    # Create timing metrics
    start_time = time.time()
    
    with torch.no_grad():
        # Create a progress bar
        progress_bar = tqdm(total=len(texts), desc="Generating predictions", 
                           unit="texts", ncols=100, position=0, leave=True)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Time this batch
            batch_start = time.time()
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,  # Limit token length for efficiency
                return_tensors="pt"
            ).to(device)
            
            # Get predictions
            outputs = model(**inputs)
            pred_labels = torch.argmax(outputs.logits, dim=-1)
            batch_predictions = pred_labels.cpu().numpy()
            predictions.extend(batch_predictions)
            
            # Update progress bar
            progress_bar.update(len(batch_texts))
            
            # Log performance for larger batches
            if i % (batch_size * 5) == 0 and i > 0:
                batch_time = time.time() - batch_start
                texts_per_second = len(batch_texts) / batch_time
                print(f"\nBatch {i//batch_size}/{total_batches} - "
                      f"Speed: {texts_per_second:.2f} texts/s - "
                      f"Time: {batch_time:.2f}s")
                
                if device.type == 'cuda':
                    print(f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        
        # Close progress bar
        progress_bar.close()
    
    # Total throughput
    total_time = time.time() - start_time
    throughput = len(texts) / total_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Average speed: {throughput:.2f} texts/s")
    
    return predictions

def save_predictions(predictions, output_file):
    """Save predictions in competition format"""
    # Create DataFrame with ID and Labels columns
    df = pd.DataFrame({
        'ID': range(len(predictions)),
        'Labels': predictions
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    # Parse arguments
    args = parse_args()
    model_dir = args.model_dir
    test_file = args.test_file
    batch_size = args.batch_size
    
    print(f"Using model directory: {model_dir}")
    print(f"Using test file: {test_file}")
    print(f"Using batch size: {batch_size}")
    
    # Create submissions directory if it doesn't exist
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Verify the test file exists
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found at {test_file}")
    
    # Force CUDA options for better performance
    if torch.cuda.is_available() and not args.force_cpu:
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Try to enable TF32 precision if available (on Ampere or newer GPUs)
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster matrix multiplications")
            
        # Enable cuDNN benchmarking and deterministic algorithms
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmarking for faster inference")
    
    # Set device - prioritize GPU if available
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Clear any existing allocations
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
    
    # Find the best model checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")
    
    # Sort by checkpoint number and take the latest
    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    best_model_dir = os.path.join(model_dir, last_checkpoint)
    
    print(f"Loading model from {best_model_dir}")
    
    # Load the PEFT config
    peft_config = PeftConfig.from_pretrained(best_model_dir)
    print(f"PEFT Config: {peft_config}")
    
    # Initialize the base model with the correct number of labels (4 for AG News)
    base_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=4,  # AG News has 4 classes
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32  # Use fp16 on GPU
    )
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, best_model_dir)
    
    # Move model to device immediately
    model = model.to(device)
    
    if device.type == 'cuda':
        # Try to use half-precision for faster GPU inference
        model = model.half()
        print("Using half-precision (FP16) for faster inference")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir)
    
    # Load test data
    test_texts = load_test_data(test_file)
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, test_texts, device, batch_size=batch_size)
    
    # Create submission filename with timestamp and include model dir name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = os.path.basename(model_dir)
    submission_file = SUBMISSION_DIR / f"submission_{model_name}_{timestamp}.csv"
    
    # Save predictions
    save_predictions(predictions, submission_file)
    print(f"Submission saved to {submission_file}")

if __name__ == "__main__":
    main() 