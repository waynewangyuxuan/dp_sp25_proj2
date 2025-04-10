import os
import sys
import logging
import argparse
from pathlib import Path

# Set cache directory to project folder
cache_dir = Path(__file__).parent.parent / "cache"
cache_dir.mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir)

import torch
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LoRA model on AG News dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting experiment: {args.exp_name}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('ag_news', split='train')
    
    # Extract label information
    num_labels = dataset.features['label'].num_classes
    class_names = dataset.features["label"].names
    id2label = {i: label for i, label in enumerate(class_names)}
    
    # Load tokenizer
    base_model = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    
    # Preprocess data
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Split dataset
    split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
        base_model,
        id2label=id2label
    )
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Analyze model structure
    logger.info("Model structure:")
    for name, module in model.named_modules():
        if 'query' in name and 'layer.0' in name:
            logger.info(f"Found first layer query: {name}")
    
    # Setup LoRA configuration - targetting only what we need
    logger.info("Setting up LoRA config...")
    peft_config = LoraConfig(
        r=8,  # Increase rank for more parameters
        lora_alpha=16,  # Twice the rank
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],  # Target all attention components
        task_type="SEQ_CLS",
        inference_mode=False,  # Ensure we're in training mode
    )
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    peft_model = get_peft_model(model, peft_config)
    
    # Don't try to access adapters directly
    # Instead, let PEFT handle the active adapter internally
    logger.info("Using default adapter configuration")
    
    # Now freeze all parameters except selected layers' LoRA adapters
    logger.info("Selectively unfreezing LoRA parameters...")
    target_layers = list(range(6))  # First 6 layers
    for name, param in peft_model.named_parameters():
        # Default: freeze all parameters
        param.requires_grad = False
        
        # Only unfreeze LoRA parameters for target layers
        if "lora" in name:
            for i in target_layers:
                if f"layer.{i}." in name:
                    param.requires_grad = True
                    logger.info(f"Keeping trainable: {name}")
                    break
    
    # Print parameter counts
    logger.info("LoRA model created")
    peft_model.print_trainable_parameters()
    
    # Detailed analysis of trainable parameters
    logger.info("Detailed trainable parameter analysis:")
    total_trainable = 0
    trainable_params_by_layer = {}
    
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            logger.info(f"Trainable: {name} - Shape: {param.shape} - Parameters: {param.numel()}")
            
            # Categorize by layer
            if "layer" in name:
                layer_num = name.split("layer.")[1].split(".")[0]
                if layer_num not in trainable_params_by_layer:
                    trainable_params_by_layer[layer_num] = 0
                trainable_params_by_layer[layer_num] += param.numel()
            elif "classifier" in name:
                if "classifier" not in trainable_params_by_layer:
                    trainable_params_by_layer["classifier"] = 0
                trainable_params_by_layer["classifier"] += param.numel()
    
    logger.info(f"Total trainable parameters: {total_trainable}")
    logger.info("Trainable parameters by layer:")
    for layer, count in trainable_params_by_layer.items():
        logger.info(f"  {layer}: {count}")
        
    # Verify we're under the parameter limit
    if total_trainable > 1_000_000:
        raise ValueError(f"Model has {total_trainable} trainable parameters, which exceeds the 1 million limit!")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        
        # Evaluate
        metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 