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
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (default: auto-generated based on config)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--target_layers', type=int, default=6, help='Number of layers to apply LoRA to')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
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
    
    # Generate descriptive run name if not provided
    if args.exp_name is None:
        import datetime
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"run_{date_str}_r{args.lora_r}_a{args.lora_alpha}_l{args.target_layers}_lr{args.learning_rate}"
        print(f"Using auto-generated experiment name: {args.exp_name}")
    
    # Generate output directory if not provided
    if args.output_dir is None:
        # Always create output directory based on experiment name to keep them linked
        args.output_dir = os.path.join("outputs", args.exp_name)
        print(f"Using auto-generated output directory: {args.output_dir}")
        
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Configuration:")
    logger.info(f"  - LoRA rank (r): {args.lora_r}")
    logger.info(f"  - LoRA alpha: {args.lora_alpha}")
    logger.info(f"  - Target layers: {args.target_layers}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Number of epochs: {args.num_epochs}")
    
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
    
    # Setup LoRA configuration
    logger.info("Setting up LoRA config...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],  # Target all attention components
        task_type="SEQ_CLS",
        inference_mode=False,  # Ensure we're in training mode
    )
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    peft_model = get_peft_model(model, peft_config)
    
    # Now selectively unfreeze LoRA adapters for the target layers
    logger.info(f"Unfreezing LoRA adapters for the first {args.target_layers} layers...")
    target_layers = list(range(args.target_layers))
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
    for layer, count in sorted(trainable_params_by_layer.items()):
        logger.info(f"  Layer {layer}: {count}")
    
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
    
    # Initialize trainer with custom handling
    logger.info("Initializing trainer...")
    
    # Override save_model method to handle adapter errors
    original_save_model = Trainer.save_model
    
    def safe_save_model(self, output_dir=None, _internal_call=False):
        try:
            logger.info(f"Saving model to {output_dir}...")
            original_save_model(self, output_dir, _internal_call)
            logger.info("Model saved successfully")
        except AttributeError as e:
            if 'adapters' in str(e) or 'adapter' in str(e):
                logger.warning(f"Caught adapter error during save: {str(e)}")
                logger.info("Attempting alternative save method...")
                # Alternative save approach
                if output_dir is None:
                    output_dir = self.args.output_dir
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info("Model saved using alternative method")
            else:
                raise  # Re-raise if it's not an adapter error
    
    # Patch the save_model method
    Trainer.save_model = safe_save_model
    
    # Also patch the _load_best_model method to handle adapter compatibility
    original_load_best_model = Trainer._load_best_model
    
    def safe_load_best_model(self):
        try:
            logger.info("Attempting to load best model...")
            original_load_best_model(self)
            logger.info("Best model loaded successfully")
        except (AttributeError, TypeError) as e:
            if 'active_adapters' in str(e) or 'subscriptable' in str(e):
                logger.warning(f"Caught adapter compatibility error: {str(e)}")
                logger.info("Continuing with current model state")
                
                # Create symbolic link to best model in best_models directory
                try:
                    best_model_checkpoint = self.state.best_model_checkpoint
                    if best_model_checkpoint:
                        best_models_dir = os.path.join(self.args.output_dir, "best_models")
                        os.makedirs(best_models_dir, exist_ok=True)
                        
                        # Create a text file with information about the best model
                        info_file = os.path.join(best_models_dir, "best_model_info.txt")
                        with open(info_file, 'w') as f:
                            f.write(f"Best model checkpoint: {best_model_checkpoint}\n")
                            f.write(f"Best model global step: {self.state.best_global_step}\n")
                            f.write(f"Best model metric: {self.state.best_metric}\n")
                            f.write("\nTo use this model, load it from the checkpoint directory above.\n")
                        
                        # Create a symbolic link to the best checkpoint
                        symlink_path = os.path.join(best_models_dir, "best_checkpoint")
                        if os.path.exists(symlink_path):
                            os.remove(symlink_path)  # Remove existing symlink if present
                        
                        # Create a relative path for the symlink
                        relative_path = os.path.relpath(best_model_checkpoint, best_models_dir)
                        os.symlink(relative_path, symlink_path)
                        
                        logger.info(f"Created symbolic link to best model at {symlink_path}")
                except Exception as link_error:
                    logger.warning(f"Failed to create symbolic link to best model: {str(link_error)}")
                
                return
            else:
                raise  # Re-raise if it's not the specific error we're handling
    
    # Patch the _load_best_model method
    Trainer._load_best_model = safe_load_best_model
    
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