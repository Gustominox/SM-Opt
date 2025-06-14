#!/bin/bash

SIZE="$1"

# Iterate over each directory starting with 'WA2'
for dir in WA2*/; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        echo "Starting tests in directory: $dir"
        
        if [[ "$dir" == "WA2/" ]]; then
        sbatch --exclusive ./WA2/perf.sh WA-NoFlags ${SIZE}
            
        elif [[ "$dir" == "WA2-Vectorize/" ]]; then
        sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags ${SIZE}
        
        elif [[ "$dir" == "WA2-CSC/" ]]; then
        sbatch --exclusive ./WA2-CSC/perf.sh WA-CSC ${SIZE}
            
        elif [[ "$dir" == "WA2-CSC-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32 48 64 96 128; do
                echo "Running with ${threads} threads..."
                sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        fi
        
    else
        echo "Directory $dir does not exist."
    fi
done
