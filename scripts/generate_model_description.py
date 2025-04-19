import os
import sys
import json
from pathlib import Path
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import PeftModel, PeftConfig

# Set cache directory to project folder
project_root = Path(__file__).parent.parent
cache_dir = project_root / "cache"
cache_dir.mkdir(exist_ok=True)

# Set environment variables for cache directories
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
os.environ["XDG_CACHE_HOME"] = str(cache_dir)

# Print cache directory for debugging
print(f"Using cache directory: {cache_dir}")

def convert_sets_to_lists(obj):
    """Convert sets to lists for JSON serialization"""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj

def get_model_info(model_path):
    """Extract model information from a saved model"""
    try:
        # Check if the path is a directory containing a final_model subdirectory
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "final_model")):
            model_path = os.path.join(model_path, "final_model")
            print(f"Using final_model directory: {model_path}")
        
        # Load the PEFT config first
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Load the base model and tokenizer
        base_model = RobertaForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=4,  # For AG News dataset
            id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            cache_dir=str(cache_dir)
        )
        
        # Load the PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Get parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Create model description
        description = {
            "model_type": "RoBERTa with LoRA",
            "base_model": peft_config.base_model_name_or_path,
            "task": "Sequence Classification",
            "parameter_counts": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "percentage_trainable": f"{(100 * trainable_params / total_params):.2f}%"
            },
            "lora_configuration": {
                "rank": peft_config.r,
                "alpha": peft_config.lora_alpha,
                "dropout": peft_config.lora_dropout,
                "target_modules": peft_config.target_modules,
                "layers_to_transform": peft_config.layers_to_transform,
                "number_of_layers": len(peft_config.layers_to_transform) if peft_config.layers_to_transform else "All"
            }
        }
        
        # Convert any sets to lists for JSON serialization
        description = convert_sets_to_lists(description)
        
        return description
    except Exception as e:
        print(f"Error in get_model_info: {str(e)}")
        raise

def main():
    # Get the model directory from command line argument
    if len(sys.argv) != 2:
        print("Usage: python generate_model_description.py <model_directory>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        print(f"Error: Directory {model_dir} does not exist")
        sys.exit(1)
    
    try:
        # Generate model description
        description = get_model_info(model_dir)
        
        # Save as both JSON and readable text
        output_dir = Path(model_dir)
        
        # Save as JSON
        with open(output_dir / "model_description.json", "w") as f:
            json.dump(description, f, indent=2)
        
        # Save as readable text
        with open(output_dir / "model_description.txt", "w") as f:
            f.write("Model Description\n")
            f.write("================\n\n")
            
            f.write("Model Type and Base Model\n")
            f.write("------------------------\n")
            f.write(f"Model Type: {description['model_type']}\n")
            f.write(f"Base Model: {description['base_model']}\n")
            f.write(f"Task: {description['task']}\n\n")
            
            f.write("Parameter Counts\n")
            f.write("----------------\n")
            f.write(f"Total Parameters: {description['parameter_counts']['total_parameters']:,}\n")
            f.write(f"Trainable Parameters: {description['parameter_counts']['trainable_parameters']:,}\n")
            f.write(f"Non-trainable Parameters: {description['parameter_counts']['non_trainable_parameters']:,}\n")
            f.write(f"Percentage Trainable: {description['parameter_counts']['percentage_trainable']}\n\n")
            
            f.write("LoRA Configuration\n")
            f.write("------------------\n")
            f.write(f"LoRA Rank (r): {description['lora_configuration']['rank']}\n")
            f.write(f"LoRA Alpha: {description['lora_configuration']['alpha']}\n")
            f.write(f"LoRA Dropout: {description['lora_configuration']['dropout']}\n")
            f.write(f"Target Modules: {description['lora_configuration']['target_modules']}\n")
            f.write(f"Layers to Transform: {description['lora_configuration']['layers_to_transform']}\n")
            f.write(f"Number of Layers: {description['lora_configuration']['number_of_layers']}\n")
        
        print(f"Model description generated successfully in {model_dir}")
        print("Files created:")
        print("- model_description.json")
        print("- model_description.txt")
        
    except Exception as e:
        print(f"Error generating model description: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 