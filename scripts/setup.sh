#!/bin/bash

# Load required modules
module purge
module load cuda/11.8.0
module load anaconda3/2023.03
module load python3/3.10.12

# Create and activate conda environment
conda create -n dp_sp25_proj2 python=3.10 -y
source /scratch/yw5954/.bashrc
conda activate dp_sp25_proj2

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models outputs/training_runs outputs/best_models

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Print environment info
echo "Environment setup complete!"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi 