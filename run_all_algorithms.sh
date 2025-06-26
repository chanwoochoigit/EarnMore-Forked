#!/bin/bash

# Define the learning rates to test
LEARNING_RATES=("2e-5" "3e-5" "4e-5")

# Define the algorithms to run
ALGORITHMS=("dqn" "mask_ddqn" "mask_dqn" "mask_sac" "mask_sync_dqn" "mask_sync_sac" "ppo" "sac")

# Create a log directory
LOG_DIR="training_logs"
mkdir -p $LOG_DIR

# Function to run a specific algorithm with a specific learning rate
run_algorithm() {
    algo=$1
    lr=$2
    
    echo "Running ${algo} with learning rate ${lr}"
    
    # Create a log file for this run
    log_file="${LOG_DIR}/${algo}_lr${lr//e-/em}.log"
    
    # Run the pipeline script with the specified learning rate
    # Note: We're assuming the pipeline scripts follow the naming convention pipeline_${algo}_mcad.sh
    if [ -f "tools/pipeline_${algo}_mcad.sh" ]; then
        bash tools/pipeline_${algo}_mcad.sh $lr > $log_file 2>&1
        echo "Completed ${algo} with learning rate ${lr}. Log saved to ${log_file}"
    else
        echo "Warning: tools/pipeline_${algo}_mcad.sh not found. Skipping."
    fi
}

# Main execution loop
for algo in "${ALGORITHMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        run_algorithm $algo $lr
    done
done

echo "All training runs completed. Logs are available in ${LOG_DIR}/" 