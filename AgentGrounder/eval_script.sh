#!/bin/bash

# Function to run a Python module with an infinite retry mechanism
# $1 = module name (e.g., inference.inference)
# $2+ = arguments (e.g., --config_path ...)
run_script() {
    local module_name=$1
    shift # Remove the first argument so $@ contains only the remaining arguments
    local args="$@"
    
    echo "Running: python -m $module_name $args"
    
    while true; do
        # Run the module with arguments separately
        python -m "$module_name" $args
        
        # $? is the exit code of the last command. 0 means success.
        if [ $? -eq 0 ]; then
            echo "$module_name completed successfully."
            break # Exit the loop to run the next script
        else
            echo "$module_name failed. Retrying in 5 seconds..."
            sleep 5 # Wait 5 seconds to avoid error spam
        fi
    done
}

# Run sequentially
# First argument is the module, the rest are arguments
run_script "inference.inference" --config_path configs/nr3d.yaml --num_workers 8
run_script "inference.inference" --config_path configs/scanrefer.yaml --num_workers 8

echo "All scripts completed."