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
#SBATCH --time=08:00:00
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb

# Change to project directory
cd $BASE_DIR

# Activate virtual environment (if needed)
source activate.sh

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

# === Experiment 1: Baseline (200K parameters) ===
# r=8, alpha=16, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 6 * 3 * 2 * 768 * 8 = 221,184
submit_job "baseline" 8 16 6 2e-4 32 20

# === Experiment 2: Medium Parameters (500K) ===
# r=12, alpha=24, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 6 * 3 * 2 * 768 * 12 = 331,776
submit_job "medium_params" 12 24 6 2e-4 32 20

# === Experiment 3: High Parameters (1M) ===
# r=16, alpha=32, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 6 * 3 * 2 * 768 * 16 = 442,368
submit_job "high_params" 16 32 6 2e-4 32 20

# === Experiment 4: Very High Parameters (1.5M) ===
# r=20, alpha=40, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 6 * 3 * 2 * 768 * 20 = 552,960
submit_job "very_high_params" 20 40 6 2e-4 32 20

# === Experiment 5: Maximum Parameters (2M) ===
# r=24, alpha=48, target_layers=6, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 6 * 3 * 2 * 768 * 24 = 663,552
submit_job "max_params" 24 48 6 2e-4 32 20

# === Experiment 6: More Layers (1M) ===
# r=8, alpha=16, target_layers=12, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 12 * 3 * 2 * 768 * 8 = 442,368
submit_job "more_layers" 8 16 12 2e-4 32 20

# === Experiment 7: More Layers + Higher Rank (1.5M) ===
# r=12, alpha=24, target_layers=12, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 12 * 3 * 2 * 768 * 12 = 663,552
submit_job "more_layers_high_rank" 12 24 12 2e-4 32 20

# === Experiment 8: Maximum Layers (2M) ===
# r=16, alpha=32, target_layers=12, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 12 * 3 * 2 * 768 * 16 = 884,736
submit_job "max_layers" 16 32 12 2e-4 32 20

# === Experiment 9: Latent LoRA (500K) ===
# Using train_latent.py with r=16, alpha=32, layers 9-11, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 3 * 2 * 2 * 768 * 16 = 147,456 (only query and value, 3 layers)
submit_job "latent_lora" 16 32 3 2e-4 32 20

# === Experiment 10: Latent LoRA High Rank (1M) ===
# Using train_latent.py with r=24, alpha=48, layers 9-11, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 3 * 2 * 2 * 768 * 24 = 221,184 (only query and value, 3 layers)
submit_job "latent_lora_high_rank" 24 48 3 2e-4 32 20

# === Experiment 11: Latent LoRA More Layers (1.5M) ===
# Using train_latent.py with r=24, alpha=48, layers 8-12, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 5 * 2 * 2 * 768 * 24 = 368,640 (only query and value, 5 layers)
submit_job "latent_lora_more_layers" 24 48 5 2e-4 32 20

# === Experiment 12: Latent LoRA Maximum (2M) ===
# Using train_latent.py with r=32, alpha=64, layers 8-12, lr=2e-4, batch_size=32, epochs=20
# Estimated params: 5 * 2 * 2 * 768 * 32 = 491,520 (only query and value, 5 layers)
submit_job "latent_lora_max" 32 64 5 2e-4 32 20

echo "===========================================" 
echo "All jobs submitted for experiment batch: $EXPERIMENT_BATCH"
echo "Check status with: squeue -u $USER"
echo "Sbatch files are in: $SBATCH_DIR"
echo "Output logs are in: $LOGS_DIR/outputs"
echo "Error logs are in: $LOGS_DIR/errors" 