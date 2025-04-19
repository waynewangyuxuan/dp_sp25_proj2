import numpy as np
import os
from datetime import datetime
import sys

def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def format_value(value):
    if value == "N/A":
        return value
    return f"{float(value):.4f}"

def generate_training_log(metrics_dir):
    # Load all metrics
    steps = load_npy_file(os.path.join(metrics_dir, 'metrics/steps.npy'))
    train_loss = load_npy_file(os.path.join(metrics_dir, 'metrics/train_loss.npy'))
    eval_loss = load_npy_file(os.path.join(metrics_dir, 'metrics/eval_loss.npy'))
    eval_accuracy = load_npy_file(os.path.join(metrics_dir, 'metrics/eval_accuracy.npy'))

    # Check if any of the files failed to load
    if any(x is None for x in [steps, train_loss, eval_loss, eval_accuracy]):
        print("Error: Could not load all required metrics files")
        return

    # Create log content
    log_content = []
    log_content.append("Training Log for Latent LoRA Model (Layers 9-11)")
    log_content.append("=" * 50)
    log_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append("=" * 50)
    log_content.append("\nModel Configuration:")
    log_content.append("- Base Model: RoBERTa-base")
    log_content.append("- LoRA Rank: 8")
    log_content.append("- LoRA Alpha: 16")
    log_content.append("- LoRA Dropout: 0.1")
    log_content.append("- Target Modules: ['query', 'value']")
    log_content.append("- Layers Transformed: [9, 10, 11]")
    log_content.append("\nTraining Progress:")
    log_content.append("-" * 50)
    log_content.append("Step\tTrain Loss\tEval Loss\tEval Accuracy")
    log_content.append("-" * 50)

    # Add training progress
    for i in range(len(steps)):
        step = int(steps[i])
        t_loss = train_loss[i] if i < len(train_loss) else "N/A"
        e_loss = eval_loss[i] if i < len(eval_loss) else "N/A"
        e_acc = eval_accuracy[i] if i < len(eval_accuracy) else "N/A"
        
        log_content.append(f"{step}\t{format_value(t_loss)}\t{format_value(e_loss)}\t{format_value(e_acc)}")

    # Add final summary
    log_content.append("\nFinal Results:")
    log_content.append("-" * 50)
    log_content.append(f"Best Evaluation Accuracy: {format_value(np.max(eval_accuracy))}")
    log_content.append(f"Best Evaluation Loss: {format_value(np.min(eval_loss))}")
    log_content.append(f"Final Training Loss: {format_value(train_loss[-1])}")
    
    # Write to file
    output_path = os.path.join(metrics_dir, 'training_log.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(log_content))
    
    print(f"Training log generated at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_training_log.py <metrics_dir>")
        sys.exit(1)
    metrics_dir = sys.argv[1]
    generate_training_log(metrics_dir) 