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
    # "WA2-CSC-CSR/"
    "WA2-CSC-CSR-OpenMP/"
    # "WA2-CSC-OpenMP/"
    # "WA2-CSC-OpenMP-V2/"
    # "WA2-Vectorize-OpenMP/"
)


# Iterate over each directory starting with 'WA2'
for dir in "${dirs[@]}"; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        echo "Starting tests in directory: $dir"
        
        if [[ "$dir" == "WA2/" ]]; then
        sbatch --exclusive ./WA2/perf.sh WA-NoFlags ${SIZE}
        sbatch --exclusive ./WA2/perf_arm.sh WA-NoFlags-Arm ${SIZE}
            
        elif [[ "$dir" == "WA2-Vectorize/" ]]; then
        # sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags ${SIZE}
            for run in {1..5}; do
                echo "Run ${run} for ${threads} threads..."
                sbatch --exclusive ./WA2-Vectorize/perf.sh WA-AllFlags-${run} ${SIZE}
            done
        elif [[ "$dir" == "WA2-CSC/" ]]; then
        sbatch --exclusive ./WA2-CSC/perf.sh WA-CSC ${SIZE}

        elif [[ "$dir" == "WA2-CSC-CSR/" ]]; then
        sbatch --exclusive ./WA2-CSC-CSR/perf.sh WA-CSC-CSR ${SIZE}
            

    # 
        elif [[ "$dir" == "WA2-CSC-CSR-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32; do # 64 96 128; do
                echo "Running with ${threads} threads..."
                for run in {1..1}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-CSR-OpenMP/perf.sh WA-CSC-CSR-OpenMP-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done

            for threads in 2 4 8 16 32 48; do
                echo "Running with ${threads} threads..."
                for run in {1..1}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-CSR-OpenMP/perf_arm.sh WA-CSC-CSR-OpenMP-Arm-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done


        elif [[ "$dir" == "WA2-CSC-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32; do # 64 96 128; do
                echo "Running with ${threads} threads..."
                for run in {1..5}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-OneFor-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done

            for threads in 2 4 8 16 32 48; do
                echo "Running with ${threads} threads..."
                for run in {1..5}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-OpenMP/perf_arm.sh WA-CSC-OpenMP-Arm-OneFor-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        
        elif [[ "$dir" == "WA2-CSC-OpenMP-V2/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32; do # 64 96 128; do
                echo "Running with ${threads} threads..."
                for run in {1..1}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-OpenMP-V2/perf.sh WA-CSC-OpenMP-V2-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done

            for threads in 2 4 8 16 32 48; do
                echo "Running with ${threads} threads..."
                for run in {1..1}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-CSC-OpenMP-V2/perf_arm.sh WA-CSC-OpenMP-Arm-V2-${threads}threads-${run} ${SIZE} ${threads}
                done
                # sbatch --exclusive ./WA2-CSC-OpenMP/perf.sh WA-CSC-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        

        elif [[ "$dir" == "WA2-Vectorize-OpenMP/" ]]; then
            # sbatch --exclusive ./$dir/perf.sh WA-OpenMP-16threads ${SIZE} 16
            for threads in 2 4 8 16 32 64 96 128; do
                echo "Running with ${threads} threads..."

                for run in {1..5}; do
                    echo "Run ${run} for ${threads} threads..."
                    sbatch --exclusive ./WA2-Vectorize-OpenMP/perf.sh WA-Vectorize-OpenMP-${threads}threads-${run} ${SIZE} ${threads}
                done

                # sbatch --exclusive ./WA2-Vectorize-OpenMP/perf.sh WA-Vectorize-OpenMP-${threads}threads ${SIZE} ${threads}
            done
        fi
        
    else
        echo "Directory $dir does not exist."
    fi
done
