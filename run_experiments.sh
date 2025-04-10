#!/bin/bash

# Set the base directory for the project
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $BASE_DIR

# Create a timestamp for this experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BATCH="lora_experiments_${TIMESTAMP}"

# Create directories for organization
SBATCH_DIR="sbatch_scripts/${EXPERIMENT_BATCH}"
LOGS_DIR="logs/${EXPERIMENT_BATCH}"
mkdir -p $SBATCH_DIR
mkdir -p $LOGS_DIR/outputs
mkdir -p $LOGS_DIR/errors

echo "Starting experiment batch: $EXPERIMENT_BATCH"
echo "===========================================" 

# Function to submit a training job
submit_job() {
    local exp_name=$1
    local lora_r=$2
    local lora_alpha=$3
    local target_layers=$4
    local learning_rate=$5
    local batch_size=$6
    local num_epochs=$7
    
    # Calculate total trainable parameters (rough estimate: r * alpha * target_layers * 3 * 768)
    # This is a rough estimate: each attention component (query, key, value) has a 768x768 weight matrix
    # For each layer we target, we have 3 components, each with 2 LoRA matrices of size 768xr and rx768
    # Total params: target_layers * 3 * (768*r + r*768) = target_layers * 3 * 2 * 768 * r
    local estimated_params=$((target_layers * 3 * 2 * 768 * lora_r))
    
    echo "Submitting job: $exp_name"
    echo "  - LoRA rank (r): $lora_r"
    echo "  - LoRA alpha: $lora_alpha"
    echo "  - Target layers: $target_layers"
    echo "  - Learning rate: $learning_rate"
    echo "  - Batch size: $batch_size"
    echo "  - Num epochs: $num_epochs"
    echo "  - Estimated trainable parameters: $estimated_params"
    
    # Create a unique job name
    local job_name="${exp_name}_r${lora_r}_a${lora_alpha}_l${target_layers}_lr${learning_rate}"
    
    # Create sbatch file
    local sbatch_file="${SBATCH_DIR}/sbatch_${job_name}.sh"
    
    # Create the sbatch file with proper configuration for NYU HPC
    cat <<EOT > $sbatch_file
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=${LOGS_DIR}/outputs/${job_name}.out
#SBATCH --error=${LOGS_DIR}/errors/${job_name}.err
#SBATCH --time=04:00:00
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb

# Change to project directory
cd $BASE_DIR

# Activate virtual environment (if needed)
# source activate.sh

# Run the training script
python scripts/train.py --exp_name $exp_name --lora_r $lora_r --lora_alpha $lora_alpha --target_layers $target_layers --learning_rate $learning_rate --batch_size $batch_size --num_epochs $num_epochs
EOT
    
    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    # Submit the job
    sbatch --account=pr_148_general $sbatch_file
    
    # Wait a bit to avoid overwhelming the scheduler
    sleep 2
}

# === Experiment 1: Baseline ===
# r=8, alpha=16, target_layers=6, lr=2e-4, batch_size=32, epochs=10
# Estimated params: 6 * 3 * 2 * 768 * 8 = 221,184
submit_job "baseline" 8 16 6 2e-4 32 10

# === Experiment 2: Reduced Rank ===
# r=4, alpha=8, target_layers=6, lr=2e-4, batch_size=32, epochs=10
# Estimated params: 6 * 3 * 2 * 768 * 4 = 110,592
submit_job "reduced_rank" 4 8 6 2e-4 32 10

# === Experiment 3: Minimal Configuration ===
# r=2, alpha=4, target_layers=3, lr=2e-4, batch_size=32, epochs=10
# Estimated params: 3 * 3 * 2 * 768 * 2 = 27,648
submit_job "minimal" 2 4 3 2e-4 32 10

# === Experiment 4: Higher Learning Rate ===
# r=4, alpha=8, target_layers=6, lr=5e-4, batch_size=32, epochs=10
# Same params as reduced_rank but with higher learning rate
submit_job "higher_lr" 4 8 6 5e-4 32 10

# === Experiment 5: Lower Learning Rate ===
# r=4, alpha=8, target_layers=6, lr=1e-4, batch_size=32, epochs=10
# Same params as reduced_rank but with lower learning rate
submit_job "lower_lr" 4 8 6 1e-4 32 10

# === Experiment 6: More Layers ===
# r=2, alpha=4, target_layers=12, lr=2e-4, batch_size=32, epochs=10
# Estimated params: 12 * 3 * 2 * 768 * 2 = 110,592
submit_job "more_layers" 2 4 12 2e-4 32 10

# === Experiment 7: Longer Training ===
# r=4, alpha=8, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Same params as reduced_rank but with longer training
submit_job "longer_training" 4 8 6 2e-4 32 20

# === Experiment 8: Larger Batch Size ===
# r=4, alpha=8, target_layers=6, lr=2e-4, batch_size=64, epochs=10
# Same params as reduced_rank but with larger batch size
submit_job "larger_batch" 4 8 6 2e-4 64 10

echo "===========================================" 
echo "All jobs submitted for experiment batch: $EXPERIMENT_BATCH"
echo "Check status with: squeue -u $USER"
echo "Sbatch files are in: $SBATCH_DIR"
echo "Output logs are in: $LOGS_DIR/outputs"
echo "Error logs are in: $LOGS_DIR/errors" 