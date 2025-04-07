import os
import sys
import logging
from pathlib import Path
import torch
from transformers import (
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import AGNewsDataLoader
from src.models.lora_model import LoRAModel
from src.training.config import TrainingConfig
from src.training.trainer import CustomTrainer

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config.output_dir) / "logs"
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

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Setup logging
    logger = setup_logging(config)
    
    # Set random seed for reproducibility
    set_seed(config.random_seed)
    logger.info(f"Set random seed to {config.random_seed}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = AGNewsDataLoader(
        model_name=config.model_name,
        max_length=config.max_length
    )
    train_dataset, eval_dataset = data_loader.load_and_preprocess()
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = LoRAModel(
        model_name=config.model_name,
        num_labels=config.num_labels,
        lora_config=config.lora_config
    ).get_model()
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Verify we're under the parameter limit
    if trainable_params > 1_000_000:
        raise ValueError(
            f"Model has {trainable_params:,} trainable parameters, "
            "which exceeds the 1 million parameter limit!"
        )
    
    # Move model to device
    model.to(device)
    
    # Calculate number of training steps and warmup steps
    num_training_steps = (
        len(train_dataset) 
        * config.num_epochs 
        // config.batch_size
    )
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,  # Use mixed precision training
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        warmup_steps=num_warmup_steps,
        report_to="tensorboard",  # Enable tensorboard logging
        remove_unused_columns=True,
        dataloader_num_workers=4,  # Parallel data loading
        group_by_length=True,  # Reduce padding by grouping similar lengths
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(data_loader.tokenizer),
        config=config,  # Pass our config to the custom trainer
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model
        final_output_dir = os.path.join(config.output_dir, "final_model")
        trainer.save_model(final_output_dir)
        logger.info(f"Final model saved to {final_output_dir}")
        
        # Log final metrics
        final_metrics = trainer.evaluate()
        logger.info("Final evaluation metrics:")
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main() 