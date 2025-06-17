#!/bin/bash

# Set the directory paths
SLURM_ERR_DIR="slurm-err"
SLURM_OUTPUT_DIR="slurm-output"

# Function to clean the slurm directories by removing all files
clean_slurm_dirs() {
    echo "Cleaning Slurm directories..."

    # Clean slurm-err directory
    if [ -d "$SLURM_ERR_DIR" ]; then
        echo "Removing all files in $SLURM_ERR_DIR..."
        rm -v "$SLURM_ERR_DIR"/*  # Removes all files in the directory
    else
        echo "$SLURM_ERR_DIR does not exist."
    fi

    # Clean slurm-output directory
    if [ -d "$SLURM_OUTPUT_DIR" ]; then
        echo "Removing all files in $SLURM_OUTPUT_DIR..."
        rm -v "$SLURM_OUTPUT_DIR"/*  # Removes all files in the directory
    else
        echo "$SLURM_OUTPUT_DIR does not exist."
    fi
}

# Run the clean function
clean_slurm_dirs

echo "Cleaning completed."
