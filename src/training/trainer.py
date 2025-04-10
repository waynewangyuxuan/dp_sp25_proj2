import os
import sys
import time
import json
import csv
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
import numpy as np
import psutil
import pandas as pd
from tqdm.auto import tqdm

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig

def get_gpu_memory_info():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return {
            "allocated": f"{gpu_memory:.1f}MB",
            "reserved": f"{gpu_memory_reserved:.1f}MB",
            "total": f"{torch.cuda.get_device_properties(0).total_memory/1024/1024:.1f}MB"
        }
    return {"allocated": "0MB", "reserved": "0MB", "total": "0MB"}

class ProgressCallback(TrainerCallback):
    """Custom callback for interactive progress display"""
    def __init__(self):
        self.training_bar = None
        self.epoch_bar = None
        self.current_epoch = 0
        self.current_step = 0
        self.train_loss = 0.0
        self.best_metric = 0.0
        self.log_history = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.training_bar = tqdm(total=self.total_steps, desc="Training", position=0)
        
        # Log initial GPU info
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            self._log_gpu_info("Initial", gpu_info)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(desc=f"Epoch {state.epoch}", position=1, leave=False)
        self.current_step = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.training_bar.update(1)
        self.epoch_bar.update(1)
        self.current_step += 1
        
        # Update metrics display
        if state.log_history:
            latest_log = state.log_history[-1]
            if "loss" in latest_log:
                self.train_loss = latest_log["loss"]
                gpu_info = get_gpu_memory_info()
                self.training_bar.set_postfix({
                    "loss": f"{self.train_loss:.4f}",
                    "best_acc": f"{self.best_metric:.4f}",
                    "GPU": gpu_info["allocated"]
                })
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_acc = metrics.get("eval_accuracy", 0)
            if eval_acc > self.best_metric:
                self.best_metric = eval_acc
            
            # Update progress bars with metrics
            gpu_info = get_gpu_memory_info()
            self.training_bar.set_postfix({
                "loss": f"{self.train_loss:.4f}",
                "eval_acc": f"{eval_acc:.4f}",
                "best_acc": f"{self.best_metric:.4f}",
                "GPU": gpu_info["allocated"]
            })
            
            # Store metrics for plotting
            self.log_history.append({
                "step": state.global_step,
                "eval_accuracy": eval_acc,
                "loss": self.train_loss,
                "gpu_memory": float(gpu_info["allocated"].rstrip("MB"))
            })
            
            # Log GPU usage after evaluation
            self._log_gpu_info("Evaluation", gpu_info)
    
    def _log_gpu_info(self, stage: str, gpu_info: Dict[str, str]):
        """Log GPU memory information"""
        if torch.cuda.is_available():
            tqdm.write(
                f"\n[{stage}] GPU Memory: "
                f"Used={gpu_info['allocated']}, "
                f"Reserved={gpu_info['reserved']}, "
                f"Total={gpu_info['total']}"
            )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            # Log GPU usage after epoch
            gpu_info = get_gpu_memory_info()
            self._log_gpu_info(f"Epoch {state.epoch}", gpu_info)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        # Log final GPU usage
        gpu_info = get_gpu_memory_info()
        self._log_gpu_info("Final", gpu_info)

class CustomTrainer(Trainer):
    """Custom trainer with enhanced logging and checkpointing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup output directories
        self._setup_output_dirs()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'learning_rate': [],
            'time_elapsed': [],
            'memory_used': []
        }
        
        # Add progress callback
        if not any(isinstance(callback, ProgressCallback) for callback in self.callback_handler.callbacks):
            self.add_callback(ProgressCallback())
            
        # Setup multi-GPU training if available
        self._setup_multi_gpu()
        
    def _setup_multi_gpu(self):
        """Configure multi-GPU training"""
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            self._log_to_file(f"Found {n_gpus} GPUs. Setting up distributed training.")
            
            # DDP will be handled by the Trainer class automatically
            self.args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
            self.args.distributed = True
            
            # Log GPU information
            for i in range(n_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                self._log_to_file(
                    f"GPU {i}: {gpu_props.name}, "
                    f"Memory: {gpu_props.total_memory/1024/1024:.1f}MB, "
                    f"Compute Capability: {gpu_props.major}.{gpu_props.minor}"
                )
        else:
            self._log_to_file("Single GPU training mode.")
            
    def _setup_output_dirs(self):
        """Setup output directories for the current training run"""
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        
        # Setup directory structure
        run_dir = os.path.join(self.args.output_dir, "training_runs", timestamp)
        self.checkpoints_dir = os.path.join(run_dir, "checkpoints")
        self.logs_dir = os.path.join(run_dir, "logs")
        self.figures_dir = os.path.join(run_dir, "figures")
        self.best_models_dir = os.path.join(self.args.output_dir, "best_models")
        
        # Create directories
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        
        # Setup log files
        self.log_file = os.path.join(self.logs_dir, "training.log")
        self.metrics_file = os.path.join(self.logs_dir, "metrics.csv")
        self.config_file = os.path.join(self.logs_dir, "config.json")
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics CSV file with header
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'train_loss', 'eval_loss', 'eval_accuracy', 
                           'learning_rate', 'time_elapsed', 'memory_used_mb'])
        
        # Log model information
        self._log_model_info()
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def _save_config(self):
        """Save configuration to a JSON file"""
        # Convert training arguments to dict
        config_dict = self.args.to_dict()
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def _log_model_info(self):
        """Log detailed information about the model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self._log_to_file("=" * 80)
        self._log_to_file("MODEL INFORMATION")
        self._log_to_file("=" * 80)
        self._log_to_file(f"Total Parameters: {total_params:,}")
        self._log_to_file(f"Trainable Parameters: {trainable_params:,}")
        self._log_to_file(f"Training Arguments: {self.args}")
        self._log_to_file("\n" + "=" * 80)
    
    def _log_to_file(self, message: str):
        """Log a message to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _log_metrics(self, step: int, epoch: float, metrics: Dict[str, float], 
                    time_elapsed: float, memory_used: float):
        """Log metrics to CSV file and update history"""
        # Update metrics history
        self.metrics_history['train_loss'].append(metrics.get('loss', 0.0))
        self.metrics_history['eval_loss'].append(metrics.get('eval_loss', 0.0))
        self.metrics_history['eval_accuracy'].append(metrics.get('eval_accuracy', 0.0))
        self.metrics_history['learning_rate'].append(self.state.optimizer.param_groups[0]['lr'])
        self.metrics_history['time_elapsed'].append(time_elapsed)
        self.metrics_history['memory_used'].append(memory_used)
        
        # Log to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                epoch,
                metrics.get('loss', 0.0),
                metrics.get('eval_loss', 0.0),
                metrics.get('eval_accuracy', 0.0),
                self.state.optimizer.param_groups[0]['lr'],
                time_elapsed,
                memory_used
            ])
    
    def _generate_training_graphs(self):
        """Generate training visualization graphs"""
        try:
            # Create figure directory if it doesn't exist
            os.makedirs(self.figures_dir, exist_ok=True)
            
            # Convert metrics history to DataFrame for easier plotting
            df = pd.DataFrame(self.metrics_history)
            
            # 1. Loss Curves
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['train_loss'], label='Training Loss')
            plt.plot(df.index, df['eval_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, 'loss_curves.png'))
            plt.close()
            
            # 2. Accuracy Curve
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['eval_accuracy'], label='Validation Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Steps')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, 'accuracy_curve.png'))
            plt.close()
            
            # 3. Learning Rate Schedule
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['learning_rate'], label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, 'learning_rate.png'))
            plt.close()
            
            # 4. Memory Usage
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['memory_used'], label='Memory Usage')
            plt.title('Memory Usage Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, 'memory_usage.png'))
            plt.close()
            
            # 5. Combined Metrics Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Loss subplot
            ax1.plot(df.index, df['train_loss'], label='Training Loss')
            ax1.plot(df.index, df['eval_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Accuracy subplot
            ax2.plot(df.index, df['eval_accuracy'], label='Validation Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'combined_metrics.png'))
            plt.close()
            
            # 6. Correlation Heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Metrics Correlation Heatmap')
            plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.png'))
            plt.close()
            
            self._log_to_file("Generated training visualization graphs")
            
        except Exception as e:
            self._log_to_file(f"Error generating training graphs: {str(e)}")
    
    def train(self, *args, **kwargs):
        """Override train method to add visualization"""
        start_time = time.time()
        
        # Call parent's train method
        result = super().train(*args, **kwargs)
        
        # Generate final visualizations
        self._generate_training_graphs()
        
        # Log final training statistics
        total_time = time.time() - start_time
        self._log_to_file(f"Training completed in {total_time:.2f} seconds")
        self._log_to_file(f"Final validation accuracy: {self.best_accuracy:.4f}")
        
        return result
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save the model with enhanced checkpointing"""
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Create checkpoint name with metrics
        metrics = self.state.log_history[-1] if self.state.log_history else {}
        checkpoint_name = f"checkpoint-{self.state.global_step:05d}"
        if metrics:
            checkpoint_name += f"-acc{metrics.get('eval_accuracy', 0):.4f}"
        
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
            
        # Save training state
        torch.save(self.state, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        # Save metrics
        with open(os.path.join(checkpoint_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # If this is the best model, save it separately
        if metrics.get('eval_accuracy', 0) > getattr(self, 'best_accuracy', 0):
            self.best_accuracy = metrics['eval_accuracy']
            best_model_dir = os.path.join(self.best_models_dir, f"best-acc{self.best_accuracy:.4f}")
            os.makedirs(best_model_dir, exist_ok=True)
            self.model.save_pretrained(best_model_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(best_model_dir)
            
            # Save best model metrics
            with open(os.path.join(best_model_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self._log_to_file(f"New best model saved with accuracy: {self.best_accuracy:.4f}")
            
            # Generate and save visualizations for best model
            self._generate_training_graphs() 