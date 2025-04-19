#!/bin/bash

# Set the base directory
OUTPUTS_DIR="/scratch/yw5954/dp_sp25_proj2/outputs"
SCRIPT_DIR="/scratch/yw5954/dp_sp25_proj2/scripts"

cd /scratch/yw5954/dp_sp25_proj2
source activate.sh

# Change to the script directory
cd "$SCRIPT_DIR"

# Loop through all directories in the outputs folder
for dir in "$OUTPUTS_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        python generate_training_log.py "$dir"
        echo "----------------------------------------"
    fi
done

echo "All directories processed." 