#!/bin/bash

# Set the directory paths
SLURM_ERR_DIR="slurm-err"
SLURM_OUTPUT_DIR="slurm-output"

# Set the number of days beyond which files will be deleted
DAYS_OLD=30  # Files older than 30 days will be removed

# Function to clean the slurm directories
clean_slurm_dirs() {
    echo "Cleaning Slurm directories..."

    # Clean slurm-err directory
    if [ -d "$SLURM_ERR_DIR" ]; then
        echo "Cleaning $SLURM_ERR_DIR..."
        find "$SLURM_ERR_DIR" -type f -name "*.out" -mtime +$DAYS_OLD -exec rm -v {} \;
    else
        echo "$SLURM_ERR_DIR does not exist."
    fi

    # Clean slurm-output directory
    if [ -d "$SLURM_OUTPUT_DIR" ]; then
        echo "Cleaning $SLURM_OUTPUT_DIR..."
        find "$SLURM_OUTPUT_DIR" -type f -name "*.out" -mtime +$DAYS_OLD -exec rm -v {} \;
    else
        echo "$SLURM_OUTPUT_DIR does not exist."
    fi
}

# Run the clean function
clean_slurm_dirs

echo "Cleaning completed."
