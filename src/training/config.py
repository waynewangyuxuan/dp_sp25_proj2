from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class TrainingConfig:
    """Configuration for AG News classification with LoRA"""
    
    # Model configuration
    model_name: str = 'roberta-base'
    num_labels: int = 4  # AG News has 4 classes
    max_length: int = 512
    
    # Multi-GPU training configuration
    use_multi_gpu: bool = True  # Enable multi-GPU training if available
    strategy: str = 'ddp'  # Options: 'ddp' (DistributedDataParallel) or 'dp' (DataParallel)
    
    # LoRA configuration
    lora_config: Dict[str, Any] = None
    default_lora_config: Dict[str, Any] = {
        'r': 8,  # LoRA attention dimension
        'lora_alpha': 16,  # Alpha scaling
        'lora_dropout': 0.1,
        'bias': 'none',
        'task_type': 'SEQ_CLS',
        # Target specific layers for LoRA adaptation
        'target_modules': [
            'query',
            'key',
            'value'
        ],
    }
    
    # Training hyperparameters
    learning_rate: float = 2e-4  # Slightly higher LR since we're only training LoRA params
    batch_size: int = 32
    num_epochs: int = 10
    warmup_ratio: float = 0.1  # Warmup for first 10% of steps
    weight_decay: float = 0.01
    
    # Evaluation and logging
    logging_steps: int = 10  # Log every 10 steps
    eval_steps: int = 100  # Evaluate every 100 steps
    save_steps: int = 100  # Save every 100 steps
    
    # Output settings
    output_dir: str = './outputs'
    
    # Data settings
    train_split_size: float = 0.9  # 90% for training, 10% for validation
    random_seed: int = 42
    
    def __post_init__(self):
        """Set LoRA config after initialization if not provided"""
        if self.lora_config is None:
            self.lora_config = self.default_lora_config.copy()
        
        # Adjust batch size for multi-GPU training
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            self.batch_size = self.batch_size * torch.cuda.device_count()
            
    def update_lora_config(self, **kwargs):
        """Update LoRA configuration parameters"""
        self.lora_config.update(kwargs) 