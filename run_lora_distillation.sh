#!/bin/bash

# Set the base directory for the project
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd $BASE_DIR

# Create a timestamp for this experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BATCH="lora_distillation_${TIMESTAMP}"

# Create directories for organization
SBATCH_DIR="sbatch_scripts/${EXPERIMENT_BATCH}"
LOGS_DIR="logs/${EXPERIMENT_BATCH}"
mkdir -p $SBATCH_DIR
mkdir -p $LOGS_DIR/outputs
mkdir -p $LOGS_DIR/errors

echo "Starting LoRA distillation experiment batch: $EXPERIMENT_BATCH"
echo "===========================================" 

# Function to submit a distillation training job
submit_job() {
    local exp_name=$1
    local base_model=$2
    local teacher_lora_r=$3
    local teacher_lora_alpha=$4
    local teacher_target_layers=$5
    local student_lora_r=$6
    local student_lora_alpha=$7
    local student_target_layers=$8
    local alpha=$9
    local temperature=${10}
    local learning_rate=${11}
    local batch_size=${12}
    local num_epochs=${13}
    local teacher_checkpoint=${14}
    
    # Calculate total trainable parameters for student (rough estimate)
    # For each layer we target, we have 3 components (query, key, value), each with 2 LoRA matrices
    # Total params: student_target_layers * 3 * (768*student_lora_r + student_lora_r*768)
    local estimated_student_params=$((student_target_layers * 3 * 2 * 768 * student_lora_r))
    
    echo "Submitting job: $exp_name"
    echo "  - Base model: $base_model"
    echo "  - Teacher LoRA rank: $teacher_lora_r"
    echo "  - Teacher LoRA alpha: $teacher_lora_alpha"
    echo "  - Teacher target layers: $teacher_target_layers"
    echo "  - Student LoRA rank: $student_lora_r"
    echo "  - Student LoRA alpha: $student_lora_alpha"
    echo "  - Student target layers: $student_target_layers"
    echo "  - Alpha: $alpha"
    echo "  - Temperature: $temperature"
    echo "  - Learning rate: $learning_rate"
    echo "  - Batch size: $batch_size"
    echo "  - Num epochs: $num_epochs"
    echo "  - Estimated student trainable parameters: $estimated_student_params"
    if [ ! -z "$teacher_checkpoint" ]; then
        echo "  - Teacher checkpoint: $teacher_checkpoint"
    fi
    
    # Create a unique job name
    local job_name="${exp_name}_tr${teacher_lora_r}_ta${teacher_lora_alpha}_tl${teacher_target_layers}_sr${student_lora_r}_sa${student_lora_alpha}_sl${student_target_layers}"
    
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

# Run the distillation training script
python scripts/train_distillation.py \
    --exp_name "$exp_name" \
    --base_model "$base_model" \
    --teacher_lora_r "$teacher_lora_r" \
    --teacher_lora_alpha "$teacher_lora_alpha" \
    --teacher_target_layers "$teacher_target_layers" \
    --student_lora_r "$student_lora_r" \
    --student_lora_alpha "$student_lora_alpha" \
    --student_target_layers "$student_target_layers" \
    --alpha "$alpha" \
    --temperature "$temperature" \
    --learning_rate "$learning_rate" \
    --batch_size "$batch_size" \
    --num_epochs "$num_epochs" \
    --output_dir "outputs/lora_distillation/$exp_name" \
    ${teacher_checkpoint:+--teacher_checkpoint "$teacher_checkpoint"}
EOT
    
    # Make the sbatch file executable
    chmod +x $sbatch_file
    
    # Submit the job
    sbatch --account=pr_148_general $sbatch_file
    
    # Wait a bit to avoid overwhelming the scheduler
    sleep 2
}

# === Experiment 1: Basic LoRA distillation with RoBERTa-base ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=8, alpha=16, layers=6
# Estimated student params: 6 * 3 * 2 * 768 * 8 = 221,184
submit_job \
    "lora_distill_roberta_base" \
    "roberta-base" \
    16 \
    32 \
    8 \
    8 \
    16 \
    6 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 2: LoRA distillation with higher alpha (more weight on teacher) ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=8, alpha=16, layers=6
# Alpha: 0.7 (more weight on teacher)
submit_job \
    "lora_distill_roberta_base_alpha_0.7" \
    "roberta-base" \
    16 \
    32 \
    8 \
    8 \
    16 \
    6 \
    0.7 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 3: LoRA distillation with lower temperature (sharper distributions) ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=8, alpha=16, layers=6
# Temperature: 1.0 (sharper distributions)
submit_job \
    "lora_distill_roberta_base_temp_1.0" \
    "roberta-base" \
    16 \
    32 \
    8 \
    8 \
    16 \
    6 \
    0.5 \
    1.0 \
    2e-4 \
    32 \
    10

# === Experiment 4: LoRA distillation with higher teacher LoRA rank ===
# Teacher: r=32, alpha=64, layers=8
# Student: r=8, alpha=16, layers=6
submit_job \
    "lora_distill_roberta_base_teacher_r_32" \
    "roberta-base" \
    32 \
    64 \
    8 \
    8 \
    16 \
    6 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 5: LoRA distillation with more teacher target layers ===
# Teacher: r=16, alpha=32, layers=10
# Student: r=8, alpha=16, layers=6
submit_job \
    "lora_distill_roberta_base_teacher_layers_10" \
    "roberta-base" \
    16 \
    32 \
    10 \
    8 \
    16 \
    6 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 6: LoRA distillation with smaller student model ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=4, alpha=8, layers=4
# Estimated student params: 4 * 3 * 2 * 768 * 4 = 73,728
submit_job \
    "lora_distill_roberta_base_small_student" \
    "roberta-base" \
    16 \
    32 \
    8 \
    4 \
    8 \
    4 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 7: LoRA distillation with BERT-base as base model ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=8, alpha=16, layers=6
submit_job \
    "lora_distill_bert_base" \
    "bert-base-uncased" \
    16 \
    32 \
    8 \
    8 \
    16 \
    6 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 8: LoRA distillation with DistilBERT as base model ===
# Teacher: r=16, alpha=32, layers=8
# Student: r=8, alpha=16, layers=6
submit_job \
    "lora_distill_distilbert" \
    "distilbert-base-uncased" \
    16 \
    32 \
    8 \
    8 \
    16 \
    6 \
    0.5 \
    2.0 \
    2e-4 \
    32 \
    10

# === Experiment 9: LoRA distillation with custom teacher checkpoint ===
# Uncomment and modify the path if you have a pre-trained teacher model
# submit_job \
#     "lora_distill_custom_teacher" \
#     "roberta-base" \
#     16 \
#     32 \
#     8 \
#     8 \
#     16 \
#     6 \
#     0.5 \
#     2.0 \
#     2e-4 \
#     32 \
#     10 \
#     "path/to/your/teacher_checkpoint"

echo "===========================================" 
echo "All LoRA distillation jobs submitted for experiment batch: $EXPERIMENT_BATCH"
echo "Check status with: squeue -u $USER"
echo "Sbatch files are in: $SBATCH_DIR"
echo "Output logs are in: $LOGS_DIR/outputs"
echo "Error logs are in: $LOGS_DIR/errors" 