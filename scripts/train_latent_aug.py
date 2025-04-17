import os
import sys
import logging
import argparse
import random
import re
from pathlib import Path

# Set cache directory to project folder
cache_dir = Path(__file__).parent.parent / "cache"
cache_dir.mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir)

# Set matplotlib cache directory
matplotlib_cache_dir = Path(__file__).parent.parent / "matplotlib_cache"
matplotlib_cache_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data for tokenization
try:
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Download all required NLTK resources
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print(f"Using fallback tokenization method")
    
    # Fallback tokenization function if NLTK fails
    def word_tokenize(text):
        return text.split()
    
    # Override NLTK's word_tokenize with our fallback
    nltk.word_tokenize = word_tokenize

class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.steps = []
        self.current_step = 0
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Metrics will be saved to {self.metrics_dir}")
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called when initialization ends"""
        self.logger.info("Metrics callback initialized")
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.logger.info("Starting training with metrics tracking")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.logger.info("Training ended, generating final plots")
        self.plot_metrics()
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        
        # Track training loss
        if "loss" in logs:
            self.train_loss.append(logs["loss"])
            self.steps.append(state.global_step)
            self.current_step = state.global_step
        
        # Track evaluation metrics
        if "eval_loss" in logs:
            self.eval_loss.append(logs["eval_loss"])
            self.eval_accuracy.append(logs["eval_accuracy"])
            
            # Save metrics to files
            np.save(os.path.join(self.metrics_dir, "train_loss.npy"), np.array(self.train_loss))
            np.save(os.path.join(self.metrics_dir, "eval_loss.npy"), np.array(self.eval_loss))
            np.save(os.path.join(self.metrics_dir, "eval_accuracy.npy"), np.array(self.eval_accuracy))
            np.save(os.path.join(self.metrics_dir, "steps.npy"), np.array(self.steps))
            
            # Plot metrics
            self.plot_metrics()
        
        return control
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        if len(self.train_loss) < 2 or len(self.eval_loss) < 1:
            return  # Not enough data points to plot
        
        # Set style
        plt.style.use('ggplot')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Latent LoRA Training Metrics (with Augmentation)', fontsize=16)
        
        # Plot 1: Training Loss
        axes[0, 0].plot(self.steps, self.train_loss, 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: Validation Loss
        eval_steps = self.steps[::len(self.steps)//len(self.eval_loss)][:len(self.eval_loss)]
        axes[0, 1].plot(eval_steps, self.eval_loss, 'r-', label='Validation Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Plot 3: Validation Accuracy
        axes[1, 0].plot(eval_steps, self.eval_accuracy, 'g-', label='Validation Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot 4: Combined Plot
        axes[1, 1].plot(self.steps, self.train_loss, 'b-', label='Training Loss')
        axes[1, 1].plot(eval_steps, self.eval_loss, 'r-', label='Validation Loss')
        axes[1, 1].plot(eval_steps, self.eval_accuracy, 'g-', label='Validation Accuracy')
        axes[1, 1].set_title('Combined Metrics')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.metrics_dir, "training_metrics.png"), dpi=300)
        plt.close()
        
        # Create additional plots
        
        # Learning curve (accuracy vs. epochs)
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, self.eval_accuracy, 'g-', label='Validation Accuracy')
        plt.title('Learning Curve')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.metrics_dir, "learning_curve.png"), dpi=300)
        plt.close()
        
        # Loss comparison
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_loss, 'b-', label='Training Loss')
        plt.plot(eval_steps, self.eval_loss, 'r-', label='Validation Loss')
        plt.title('Loss Comparison')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.metrics_dir, "loss_comparison.png"), dpi=300)
        plt.close()
        
        # Accuracy trend
        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, self.eval_accuracy, 'g-', marker='o')
        plt.title('Accuracy Trend')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(self.metrics_dir, "accuracy_trend.png"), dpi=300)
        plt.close()
        
        self.logger.info(f"Metrics plots saved to {self.metrics_dir}")

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Data augmentation functions
def random_swap_words(text, prob=0.1):
    """Randomly swap adjacent words with probability prob"""
    words = word_tokenize(text)
    if len(words) < 2:
        return text
    
    for i in range(len(words) - 1):
        if random.random() < prob:
            words[i], words[i+1] = words[i+1], words[i]
    
    return ' '.join(words)

def random_delete_words(text, prob=0.1):
    """Randomly delete words with probability prob"""
    words = word_tokenize(text)
    if len(words) < 2:
        return text
    
    words = [word for word in words if random.random() > prob]
    return ' '.join(words)

def random_insert_synonyms(text, prob=0.1):
    """Randomly insert synonyms with probability prob"""
    # Simple implementation - in a real scenario, you would use a thesaurus
    # For now, we'll just duplicate some words
    words = word_tokenize(text)
    if len(words) < 2:
        return text
    
    new_words = []
    for word in words:
        new_words.append(word)
        if random.random() < prob and len(word) > 3:  # Only duplicate words longer than 3 chars
            new_words.append(word)
    
    return ' '.join(new_words)

def back_translate(text, prob=0.1):
    """Simulate back translation by adding some noise"""
    if random.random() > prob:
        return text
    
    # Simple implementation - in a real scenario, you would use a translation API
    # For now, we'll just add some noise
    words = word_tokenize(text)
    if len(words) < 2:
        return text
    
    # Add some noise by replacing some characters
    noisy_words = []
    for word in words:
        if len(word) > 3 and random.random() < 0.2:
            # Replace a random character with a similar one
            pos = random.randint(0, len(word) - 1)
            char = word[pos]
            if char.isalpha():
                # Replace with a similar character
                similar_chars = {
                    'a': 'e', 'e': 'a', 'i': 'y', 'o': 'u', 'u': 'o',
                    's': 'z', 'z': 's', 'c': 'k', 'k': 'c', 't': 'd', 'd': 't'
                }
                if char.lower() in similar_chars:
                    new_char = similar_chars[char.lower()]
                    if char.isupper():
                        new_char = new_char.upper()
                    noisy_word = word[:pos] + new_char + word[pos+1:]
                    noisy_words.append(noisy_word)
                else:
                    noisy_words.append(word)
            else:
                noisy_words.append(word)
        else:
            noisy_words.append(word)
    
    return ' '.join(noisy_words)

def augment_text(text, augmentation_prob=0.5):
    """Apply a random augmentation to the text with probability augmentation_prob"""
    if random.random() > augmentation_prob:
        return text
    
    # Choose a random augmentation method
    augmentation_methods = [
        random_swap_words,
        random_delete_words,
        random_insert_synonyms,
        back_translate
    ]
    
    method = random.choice(augmentation_methods)
    return method(text)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Latent LoRA model on AG News dataset with data augmentation')
    
    # Basic training parameters
    parser.add_argument('--exp_name', type=str, default=None, 
                       help='Experiment name (default: auto-generated based on config)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    # Model architecture parameters
    parser.add_argument('--start_layer', type=int, default=6,
                       help='First layer to apply LoRA (default: 6)')
    parser.add_argument('--end_layer', type=int, default=11,
                       help='Last layer to apply LoRA (inclusive) (default: 11)')
    parser.add_argument('--lora_r', type=int, default=48,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=96,
                       help='LoRA alpha scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='Dropout probability for LoRA layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of steps for warmup')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    
    # Efficiency parameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients')
    parser.add_argument('--fp16', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing')
    
    # Data augmentation parameters
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                       help='Probability of applying augmentation to a text')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                       help='Ratio of training data to augment')
    parser.add_argument('--disable_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    return parser.parse_args()

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    args = parse_args()
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        import datetime
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"latent_lora_aug_{date_str}_l{args.start_layer}-{args.end_layer}"
    
    # Generate output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", args.exp_name)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting Latent LoRA experiment with augmentation: {args.exp_name}")
    
    # Set random seed
    set_seed(args.seed)
    random.seed(args.seed)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('ag_news', split='train')
    
    # Get label information
    num_labels = dataset.features['label'].num_classes
    id2label = {i: label for i, label in enumerate(dataset.features["label"].names)}
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=num_labels,
        id2label=id2label
    )
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Configure LoRA for specific layers
    target_modules = ["query", "value"]  # Only apply to query and value projections
    layers_to_transform = list(range(args.start_layer, args.end_layer + 1))
    
    logger.info(f"Applying LoRA to layers {layers_to_transform} for modules: {target_modules}")
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Calculate and store trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Save parameter counts to a file
    param_info_file = os.path.join(args.output_dir, "parameter_counts.txt")
    with open(param_info_file, 'w') as f:
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        f.write(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%\n\n")
        f.write(f"LoRA configuration:\n")
        f.write(f"  - LoRA rank (r): {args.lora_r}\n")
        f.write(f"  - LoRA alpha: {args.lora_alpha}\n")
        f.write(f"  - Target modules: {target_modules}\n")
        f.write(f"  - Layers to transform: {layers_to_transform}\n")
        f.write(f"  - Number of layers: {len(layers_to_transform)}\n")
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Apply data augmentation if enabled
    if not args.disable_augmentation:
        logger.info(f"Applying data augmentation with probability {args.augmentation_prob} to {args.augmentation_ratio*100}% of training data")
        
        # Create a function to augment the dataset
        def augment_dataset(examples):
            # Determine which examples to augment based on augmentation_ratio
            num_examples = len(examples['text'])
            num_to_augment = int(num_examples * args.augmentation_ratio)
            
            # Create a mask for examples to augment
            augment_mask = [False] * num_examples
            augment_indices = random.sample(range(num_examples), num_to_augment)
            for idx in augment_indices:
                augment_mask[idx] = True
            
            # Apply augmentation to selected examples
            augmented_texts = []
            for i, text in enumerate(examples['text']):
                if augment_mask[i]:
                    augmented_texts.append(augment_text(text, args.augmentation_prob))
                else:
                    augmented_texts.append(text)
            
            examples['text'] = augmented_texts
            return examples
        
        # Apply augmentation to the dataset
        dataset = dataset.map(augment_dataset, batched=True)
        
        # Log some examples of augmented data
        logger.info("Sample augmented texts:")
        for i in range(min(5, len(dataset))):
            logger.info(f"Original: {dataset['text'][i][:100]}...")
            logger.info(f"Augmented: {augment_text(dataset['text'][i], 1.0)[:100]}...")
            logger.info("---")
    
    # Preprocess data
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Split dataset
    split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=args.seed)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            MetricsCallback(args.output_dir)
        ]
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        
        # Final evaluation
        metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 