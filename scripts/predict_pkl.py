import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel, PeftConfig
from datasets import Dataset
from datetime import datetime
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import time

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_PICKLE_FILE = PROJECT_ROOT / "data" / "test" / "test_unlabelled.pkl"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
DEFAULT_MODEL_DIR = "/scratch/yw5954/dp_sp25_proj2/outputs/run_20250410_123525_r16_a32_l6_lr0.0002"

# Set cache directory to project folder to avoid disk quota issues
cache_dir = PROJECT_ROOT / "cache"
cache_dir.mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate predictions from pickle file')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, 
                       help='Directory containing the trained model checkpoints')
    parser.add_argument('--pickle_file', type=str, default=str(TEST_PICKLE_FILE),
                       help='Path to the test pickle file')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (larger values use more memory but are faster)')
    return parser.parse_args()

def load_pickle_dataset(pickle_file):
    """Try multiple methods to load the pickle file and create a dataset"""
    print(f"Loading pickle file from {pickle_file}")
    
    try:
        # First attempt: Try direct loading with pandas
        data = pd.read_pickle(pickle_file)
        print(f"Successfully loaded pickle with pandas: {type(data)}")
        
        # Check if it's a list of texts
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                print(f"Loaded {len(data)} text samples")
                return Dataset.from_dict({"text": data})
        return Dataset.from_dict({"text": data})
    
    except Exception as e:
        print(f"Error loading with pandas: {e}")
        
        try:
            # Second attempt: Try with pickle directly
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Successfully loaded with pickle: {type(data)}")
            
            # Handle different possible formats
            if isinstance(data, list):
                if all(isinstance(x, str) for x in data):
                    print(f"Loaded {len(data)} text samples")
                    return Dataset.from_dict({"text": data})
            elif isinstance(data, dict) and "text" in data:
                print(f"Loaded dictionary with {len(data['text'])} text samples")
                return Dataset.from_dict({"text": data["text"]})
            
            # If we're here, we need to try to extract text from an unknown format
            print("Unknown data format, attempting extraction from raw bytes")
            return load_from_raw_bytes(pickle_file)
        
        except Exception as e:
            print(f"Error loading with pickle: {e}")
            return load_from_raw_bytes(pickle_file)

def load_from_raw_bytes(pickle_file):
    """Fallback method to load text from raw bytes"""
    try:
        print("Attempting to extract text data by scanning raw content")
        with open(pickle_file, 'rb') as f:
            content = f.read()
        
        # Convert to text with lenient decoding
        text_content = content.decode('utf-8', errors='ignore')
        
        # Look for text patterns that appear to be news articles
        import re
        news_pattern = r'([A-Z][a-zA-Z0-9\s,\'\"\-\(\)]{20,350}[\.!?])'
        matches = re.findall(news_pattern, text_content)
        
        # Filter out obvious non-articles
        filtered = []
        for match in matches:
            if not any(x in match for x in ['function(', 'import ', 'class ', '.py', 'def ', '{', '}', '=>']):
                filtered.append(match.strip())
        
        print(f"Extracted {len(filtered)} potential text samples from raw bytes")
        
        # Use all entries, not limited to 8000
        return Dataset.from_dict({"text": filtered})
    
    except Exception as e:
        print(f"Error extracting from raw bytes: {e}")
        raise ValueError(f"Could not load data from {pickle_file} using any method")

def preprocess_for_prediction(dataset, tokenizer):
    """Preprocess dataset for prediction"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=False,  # Will be handled by DataCollator
            truncation=True,
            max_length=512
        )
    
    # Apply preprocessing
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']  # Remove the original text column
    )
    
    print(f"Tokenized dataset length: {len(tokenized_dataset)}")
    print(f"Dataset column names: {tokenized_dataset.column_names}")
    
    # Print a sample if available
    if len(tokenized_dataset) > 0:
        print(f"Sample item keys: {list(tokenized_dataset[0].keys())}")
    
    return tokenized_dataset

def run_inference(model, dataset, tokenizer, device, batch_size=32):
    """Run inference on a dataset and return predictions"""
    # Use the transformers DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    # Create data loader with the collator
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    # Prepare model
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    # Create timing metrics
    start_time = time.time()
    
    with torch.no_grad():
        # Create a progress bar
        progress_bar = tqdm(total=len(dataset), desc="Generating predictions", 
                           unit="texts", ncols=100, position=0, leave=True)
        
        for batch in dataloader:
            # Start batch timing
            batch_start = time.time()
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Run inference
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            
            # Update progress bar
            progress_bar.update(len(predictions))
            
            # Log batch timing
            batch_time = time.time() - batch_start
            texts_per_second = len(predictions) / batch_time
            
            # Log every few batches
            if len(all_predictions) % (batch_size * 5) == 0:
                print(f"\nProcessed {len(all_predictions)}/{len(dataset)} examples - "
                     f"Speed: {texts_per_second:.2f} texts/s")
        
        # Close progress bar
        progress_bar.close()
    
    # Total throughput
    total_time = time.time() - start_time
    throughput = len(dataset) / total_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Average speed: {throughput:.2f} texts/s")
    
    return all_predictions

def save_predictions(predictions, output_file):
    """Save predictions to a CSV file"""
    df = pd.DataFrame({
        'ID': range(len(predictions)),
        'Labels': predictions
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    # Parse arguments
    args = parse_args()
    model_dir = args.model_dir
    pickle_file = args.pickle_file
    batch_size = args.batch_size
    
    print(f"Using model directory: {model_dir}")
    print(f"Using pickle file: {pickle_file}")
    print(f"Using batch size: {batch_size}")
    
    # Create submissions directory
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Check if pickle file exists
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    # Set up CUDA optimization
    if torch.cuda.is_available() and not args.force_cpu:
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Try to enable TF32 precision if available (on Ampere or newer GPUs)
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster matrix multiplications")
            
        # Enable cuDNN benchmarking for faster inference
        torch.backends.cudnn.benchmark = True
        print("Enabled cuDNN benchmarking for faster inference")
    
    # Set device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Clear cache
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
    
    # Find the best model checkpoint
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_dir}")
    
    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    best_model_dir = os.path.join(model_dir, last_checkpoint)
    
    print(f"Loading model from {best_model_dir}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir)
    
    # Load the PEFT config
    peft_config = PeftConfig.from_pretrained(best_model_dir)
    print(f"PEFT Config: {peft_config}")
    
    # Initialize the base model
    base_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=4,  # AG News has 4 classes
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, best_model_dir)
    
    # Use half precision for GPU
    if device.type == 'cuda':
        model = model.half()
        print("Using half-precision (FP16) for faster inference")
    
    # Load pickle dataset
    dataset = load_pickle_dataset(pickle_file)
    
    # Preprocess dataset for prediction
    tokenized_dataset = preprocess_for_prediction(dataset, tokenizer)
    
    # Run inference with the tokenizer for proper batching
    predictions = run_inference(model, tokenized_dataset, tokenizer, device, batch_size)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = os.path.basename(model_dir)
    submission_file = SUBMISSION_DIR / f"submission_pkl_{model_name}_{timestamp}.csv"
    save_predictions(predictions, submission_file)

if __name__ == "__main__":
    main() 