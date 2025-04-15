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
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description='Train LoRA models with knowledge distillation on AG News dataset')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (default: auto-generated based on config)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    
    # Teacher model parameters
    parser.add_argument('--teacher_lora_r', type=int, default=64, help='Teacher LoRA rank')
    parser.add_argument('--teacher_lora_alpha', type=int, default=128, help='Teacher LoRA alpha')
    parser.add_argument('--teacher_target_layers', type=int, default=12, help='Number of layers to apply LoRA to in teacher')
    
    # Student model parameters
    parser.add_argument('--student_lora_r', type=int, default=16, help='Student LoRA rank')
    parser.add_argument('--student_lora_alpha', type=int, default=32, help='Student LoRA alpha')
    parser.add_argument('--student_target_layers', type=int, default=8, help='Number of layers to apply LoRA to in student')
    
    # Distillation parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for distillation loss (1-alpha for hard labels)')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--base_model', type=str, default='roberta-base', help='Base model to use for both teacher and student')
    parser.add_argument('--teacher_checkpoint', type=str, default=None, help='Path to pre-trained teacher model checkpoint')
    return parser.parse_args()

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy
    }

def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_trainable = 0
    trainable_params_by_layer = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            
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
    
    return total_trainable, trainable_params_by_layer

def log_model_device(model, model_name, logger):
    """Log the device placement of a model"""
    device = next(model.parameters()).device
    logger.info(f"{model_name} is on device: {device}")
    
    # Check if any parameters are on different devices
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    
    if len(devices) > 1:
        logger.warning(f"{model_name} has parameters on multiple devices: {devices}")
    else:
        logger.info(f"All {model_name} parameters are on device: {device}")
    
    return device

class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation"""
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Freeze teacher model
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
            # Ensure teacher model is on the same device as the student model
            if hasattr(self.model, 'device'):
                self.teacher_model = self.teacher_model.to(self.model.device)
                logger.info(f"Teacher model moved to device: {self.model.device}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for knowledge distillation
        """
        # Ensure inputs are on the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Compute hard label loss
        hard_loss = F.cross_entropy(student_logits, inputs["labels"])
        
        # Compute distillation loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Combine losses
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return (loss, student_outputs) if return_outputs else loss

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Generate descriptive run name if not provided
    if args.exp_name is None:
        import datetime
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"distill_lora_{date_str}_tr{args.teacher_lora_r}_ta{args.teacher_lora_alpha}_tl{args.teacher_target_layers}_sr{args.student_lora_r}_sa{args.student_lora_alpha}_sl{args.student_target_layers}"
        print(f"Using auto-generated experiment name: {args.exp_name}")
    
    # Generate output directory if not provided
    if args.output_dir is None:
        # Always create output directory based on experiment name to keep them linked
        args.output_dir = os.path.join("outputs", args.exp_name)
        print(f"Using auto-generated output directory: {args.output_dir}")
        
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting LoRA distillation experiment: {args.exp_name}")
    logger.info(f"Configuration:")
    logger.info(f"  - Base model: {args.base_model}")
    logger.info(f"  - Teacher LoRA rank (r): {args.teacher_lora_r}")
    logger.info(f"  - Teacher LoRA alpha: {args.teacher_lora_alpha}")
    logger.info(f"  - Teacher target layers: {args.teacher_target_layers}")
    logger.info(f"  - Student LoRA rank (r): {args.student_lora_r}")
    logger.info(f"  - Student LoRA alpha: {args.student_lora_alpha}")
    logger.info(f"  - Student target layers: {args.student_target_layers}")
    logger.info(f"  - Distillation alpha: {args.alpha}")
    logger.info(f"  - Distillation temperature: {args.temperature}")
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
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    
    # Preprocess data
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Split dataset
    split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42)
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']
    
    # Load base model for teacher
    logger.info(f"Loading base model for teacher: {args.base_model}")
    teacher_base_model = RobertaForSequenceClassification.from_pretrained(
        args.base_model,
        id2label=id2label
    )
    
    # Freeze everything in teacher base model
    for param in teacher_base_model.parameters():
        param.requires_grad = False
    
    # Setup teacher LoRA configuration
    logger.info("Setting up teacher LoRA config...")
    teacher_peft_config = LoraConfig(
        r=args.teacher_lora_r,
        lora_alpha=args.teacher_lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],  # Target all attention components
        task_type="SEQ_CLS",
        inference_mode=False,  # Ensure we're in training mode
    )
    
    # Apply LoRA to teacher
    logger.info("Applying LoRA to teacher...")
    teacher_model = get_peft_model(teacher_base_model, teacher_peft_config)
    
    # Now selectively unfreeze LoRA adapters for the target layers in teacher
    logger.info(f"Unfreezing LoRA adapters for the first {args.teacher_target_layers} layers in teacher...")
    teacher_target_layers = list(range(args.teacher_target_layers))
    for name, param in teacher_model.named_parameters():
        # Default: freeze all parameters
        param.requires_grad = False
        
        # Only unfreeze LoRA parameters for target layers
        if "lora" in name:
            for i in teacher_target_layers:
                if f"layer.{i}." in name:
                    param.requires_grad = True
                    logger.info(f"Keeping trainable in teacher: {name}")
                    break
    
    # Count teacher parameters
    teacher_total_trainable, teacher_trainable_params_by_layer = count_trainable_parameters(teacher_model)
    logger.info(f"Teacher total trainable parameters: {teacher_total_trainable}")
    logger.info("Teacher trainable parameters by layer:")
    for layer, count in sorted(teacher_trainable_params_by_layer.items()):
        logger.info(f"  Layer {layer}: {count}")
    
    # Load base model for student
    logger.info(f"Loading base model for student: {args.base_model}")
    student_base_model = RobertaForSequenceClassification.from_pretrained(
        args.base_model,
        id2label=id2label
    )
    
    # Freeze everything in student base model
    for param in student_base_model.parameters():
        param.requires_grad = False
    
    # Setup student LoRA configuration
    logger.info("Setting up student LoRA config...")
    student_peft_config = LoraConfig(
        r=args.student_lora_r,
        lora_alpha=args.student_lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],  # Target all attention components
        task_type="SEQ_CLS",
        inference_mode=False,  # Ensure we're in training mode
    )
    
    # Apply LoRA to student
    logger.info("Applying LoRA to student...")
    student_model = get_peft_model(student_base_model, student_peft_config)
    
    # Now selectively unfreeze LoRA adapters for the target layers in student
    logger.info(f"Unfreezing LoRA adapters for the first {args.student_target_layers} layers in student...")
    student_target_layers = list(range(args.student_target_layers))
    for name, param in student_model.named_parameters():
        # Default: freeze all parameters
        param.requires_grad = False
        
        # Only unfreeze LoRA parameters for target layers
        if "lora" in name:
            for i in student_target_layers:
                if f"layer.{i}." in name:
                    param.requires_grad = True
                    logger.info(f"Keeping trainable in student: {name}")
                    break
    
    # Count student parameters
    student_total_trainable, student_trainable_params_by_layer = count_trainable_parameters(student_model)
    logger.info(f"Student total trainable parameters: {student_total_trainable}")
    logger.info("Student trainable parameters by layer:")
    for layer, count in sorted(student_trainable_params_by_layer.items()):
        logger.info(f"  Layer {layer}: {count}")
    
    # Verify student is under the parameter limit
    if student_total_trainable > 1_000_000:
        raise ValueError(f"Student model has {student_total_trainable} trainable parameters, which exceeds the 1 million limit!")
    
    # Verify teacher has more parameters than student
    if teacher_total_trainable <= student_total_trainable:
        logger.warning(f"Teacher model ({teacher_total_trainable} params) has fewer or equal parameters than student model ({student_total_trainable} params)")
    
    # Load pre-trained teacher checkpoint if provided
    if args.teacher_checkpoint:
        logger.info(f"Loading teacher checkpoint from: {args.teacher_checkpoint}")
        teacher_model = RobertaForSequenceClassification.from_pretrained(
            args.teacher_checkpoint,
            id2label=id2label
        )
        
        # Move teacher model to the same device as student model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model = teacher_model.to(device)
        logger.info(f"Teacher model moved to device: {device}")
        
        # Log device placement
        log_model_device(teacher_model, "Teacher model", logger)
    else:
        # Train teacher model first
        logger.info("Training teacher model first...")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments for teacher
        teacher_training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "teacher"),
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=50,          # Reduced from 100
            eval_steps=200,           # Reduced from 500
            save_steps=200,           # Reduced from 500
            eval_strategy="steps",
            save_strategy="steps",
            fp16=torch.cuda.is_available(),
            report_to="none",
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
            greater_is_better=True,
        )
        
        # Initialize teacher trainer with custom handling
        logger.info("Initializing teacher trainer...")
        
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
                    return
                else:
                    raise  # Re-raise if it's not the specific error we're handling
        
        # Patch the _load_best_model method
        Trainer._load_best_model = safe_load_best_model
        
        teacher_trainer = Trainer(
            model=teacher_model,
            args=teacher_training_args,
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
        
        # Train teacher model
        logger.info("Starting teacher training...")
        try:
            teacher_trainer.train()
            logger.info("Teacher training completed successfully!")
            
            # Save final teacher model
            teacher_trainer.save_model(os.path.join(args.output_dir, "teacher", "final_model"))
            
            # Evaluate teacher
            teacher_metrics = teacher_trainer.evaluate()
            logger.info(f"Teacher final evaluation metrics: {teacher_metrics}")
            
        except Exception as e:
            logger.error(f"Teacher training failed with error: {str(e)}")
            raise
    
    # Data collator for student
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments for student
    student_training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "student"),
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
        fp16=torch.cuda.is_available(),
        report_to="none",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
    )
    
    # Initialize custom distillation trainer
    student_trainer = DistillationTrainer(
        model=student_model,
        args=student_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
    )
    
    # Log device placement of both models
    log_model_device(student_model, "Student model", logger)
    log_model_device(teacher_model, "Teacher model", logger)
    
    # Train student model
    logger.info("Starting student distillation training...")
    try:
        student_trainer.train()
        logger.info("Student distillation training completed successfully!")
        
        # Save final student model
        student_trainer.save_model(os.path.join(args.output_dir, "student", "final_model"))
        
        # Evaluate student
        student_metrics = student_trainer.evaluate()
        logger.info(f"Student final evaluation metrics: {student_metrics}")
        
    except Exception as e:
        logger.error(f"Student training failed with error: {str(e)}")
        raise
    
    logger.info("LoRA distillation training completed!")

if __name__ == "__main__":
    main() 