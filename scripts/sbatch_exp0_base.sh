#!/bin/bash
#SBATCH --job-name=exp0_base
#SBATCH --output=outputs/slurm_%j_exp0_base.out
#SBATCH --error=outputs/slurm_%j_exp0_base.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000

# Load necessary modules
module purge
module load cuda/11.8.0
module load anaconda3/2023.03
module load python3/3.10.12

# Activate conda environment
source /scratch/yw5954/.bashrc
conda activate dp_sp25_proj2

# Set CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set environment variables for distributed training
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)

# Run training script
if [ 1 -gt 1 ]; then
    # Multi-GPU training using DDP
    python -m torch.distributed.launch --nproc_per_node=1 \
        scripts/train.py \
        --exp_name exp0_base \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --num_epochs 10 \
        --lora_r 8 \
        --output_dir outputs/exp1_base
else
    # Single GPU training
    python scripts/train.py \
        --exp_name exp0_base \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --num_epochs 10 \
        --lora_r 8 \
        --output_dir outputs/exp1_base
fi
