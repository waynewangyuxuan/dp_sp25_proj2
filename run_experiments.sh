#!/bin/bash

# Base directory
BASE_DIR="/scratch/yw5954/dp_sp25_proj2"
cd $BASE_DIR

# Define parameter arrays
EXP_NAMES=("base" "lora16" "lr1e4" "long" "2gpu")
BATCH_SIZES=(32 32 32 32 64)
LEARNING_RATES=(2e-4 2e-4 1e-4 2e-4 2e-4)
NUM_EPOCHS=(10 10 10 20 10)
LORA_R_VALUES=(8 16 8 8 8)
NUM_GPUS=(1 1 1 1 2)
OUTPUT_DIRS=("outputs/exp1_base" "outputs/exp2_lora16" "outputs/exp3_lr1e4" "outputs/exp4_long" "outputs/exp5_2gpu")

# Create output directory for slurm logs
mkdir -p outputs

# Function to create sbatch file
create_sbatch_file() {
    local exp_name=$1
    local num_gpus=$2
    local batch_size=$3
    local learning_rate=$4
    local num_epochs=$5
    local lora_r=$6
    local output_dir=$7
    
    # Create sbatch file
    cat > "scripts/sbatch_${exp_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${exp_name}
#SBATCH --output=outputs/slurm_%j_${exp_name}.out
#SBATCH --error=outputs/slurm_%j_${exp_name}.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:${num_gpus}
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
export MASTER_ADDR=\$(hostname)

# Run training script
if [ ${num_gpus} -gt 1 ]; then
    # Multi-GPU training using DDP
    python -m torch.distributed.launch --nproc_per_node=${num_gpus} \\
        scripts/train.py \\
        --exp_name ${exp_name} \\
        --batch_size ${batch_size} \\
        --learning_rate ${learning_rate} \\
        --num_epochs ${num_epochs} \\
        --lora_r ${lora_r} \\
        --output_dir ${output_dir}
else
    # Single GPU training
    python scripts/train.py \\
        --exp_name ${exp_name} \\
        --batch_size ${batch_size} \\
        --learning_rate ${learning_rate} \\
        --num_epochs ${num_epochs} \\
        --lora_r ${lora_r} \\
        --output_dir ${output_dir}
fi
EOF

    # Make the script executable
    chmod +x "scripts/sbatch_${exp_name}.sh"
}

# Function to submit job
submit_job() {
    local exp_name=$1
    echo "Submitting job for experiment: ${exp_name}"
    sbatch --account=pr_148_general "scripts/sbatch_${exp_name}.sh"
}

# Create and submit jobs for each experiment
for i in "${!EXP_NAMES[@]}"; do
    exp_name="exp${i}_${EXP_NAMES[$i]}"
    create_sbatch_file \
        "$exp_name" \
        "${NUM_GPUS[$i]}" \
        "${BATCH_SIZES[$i]}" \
        "${LEARNING_RATES[$i]}" \
        "${NUM_EPOCHS[$i]}" \
        "${LORA_R_VALUES[$i]}" \
        "${OUTPUT_DIRS[$i]}"
    submit_job "$exp_name"
done

echo "All jobs submitted. Check outputs directory for slurm logs." 