import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import torch
from transformers import RobertaTokenizer
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lora_model import LoRAModel
from src.training.config import TrainingConfig

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
TEST_FILE = TEST_DATA_DIR / "test_unlabelled.pkl"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

def load_test_data(test_file: str):
    """Load the test dataset from pickle file"""
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

def generate_predictions(model, tokenizer, test_data, config, device):
    """Generate predictions for test data"""
    model.eval()
    predictions = []
    
    # Process in batches
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_texts = test_data[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            ).to(device)
            
            # Get predictions
            outputs = model(**inputs)
            pred_labels = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(pred_labels.cpu().numpy())
    
    return predictions

def save_predictions(predictions, output_file: str):
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
    # Create submissions directory if it doesn't exist
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Check if test file exists
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found at {TEST_FILE}. "
            f"Please place test_unlabelled.pkl in {TEST_DATA_DIR}"
        )
    
    # Load configuration
    config = TrainingConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the best model
    best_model_dir = None
    best_acc = 0.0
    
    # Find the best model directory
    for dirname in os.listdir(config.output_dir):
        if dirname.startswith('best-acc'):
            acc = float(dirname.split('acc')[-1])
            if acc > best_acc:
                best_acc = acc
                best_model_dir = os.path.join(config.output_dir, dirname)
    
    if best_model_dir is None:
        raise ValueError("No best model found!")
    
    print(f"Loading best model from {best_model_dir}")
    
    # Initialize model and tokenizer
    model = LoRAModel(
        model_name=config.model_name,
        num_labels=config.num_labels,
        lora_config=config.lora_config
    ).get_model()
    
    # Load the best weights
    model.load_state_dict(torch.load(os.path.join(best_model_dir, "pytorch_model.bin")))
    model.to(device)
    
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    
    # Load test data
    print(f"Loading test data from {TEST_FILE}")
    test_data = load_test_data(TEST_FILE)
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, test_data, config, device)
    
    # Create submission filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_file = SUBMISSION_DIR / f"submission_{timestamp}.csv"
    
    # Save predictions
    save_predictions(predictions, submission_file)
    print(f"Submission saved to {submission_file}")

if __name__ == "__main__":
    main() 