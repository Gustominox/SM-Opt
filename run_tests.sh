#!/bin/bash

# Verificar se SIZE foi fornecido como argumento
if [ -z "$1" ]; then
    echo "Erro: Tamanho (SIZE) não fornecido."
    echo "Uso: $0 <SIZE>"
    exit 1
fi

SIZE="$1"

dirs=(
    # "WA2/"
    # "WA2-Vectorize/"
    # "WA2-CSC/"
    # "WA2-CSC-OpenMP/"
    "WA2-Vectorize-OpenMP/"
)


# Iterate over each directory starting with 'WA2'
for dir in "${dirs[@]}"; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        echo "Starting tests in directory: $dir"
        
        if [[ "$dir" == "WA2/" ]]; then
        sbatch --exclusive ./WA2/perf.sh WA-NoFlags ${SIZE}
            
        elif [[ "$dir" == "WA2-Vectorize/" ]]; then
        sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags ${SIZE}
        # sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags1 ${SIZE}
        # sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags2 ${SIZE}
        # sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags3 ${SIZE}
        # sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags4 ${SIZE}
        
        elif [[ "$dir" == "WA2-CSC/" ]]; then
        sbatch --exclusive ./WA2-CSC/perf.sh WA-CSC ${SIZE}
            
        elif [[ "$dir" == "WA2-CSC-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32 64 96 128; do
                echo "Running with ${threads} threads..."
                sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        
        elif [[ "$dir" == "WA2-Vectorize-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32 64 96 128; do
                echo "Running with ${threads} threads..."
                sbatch --exclusive ./WA2-Vectorize-OpenMP/perf.sh WA-Vectorize-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        fi
        
    else
        echo "Directory $dir does not exist."
    fi
done
