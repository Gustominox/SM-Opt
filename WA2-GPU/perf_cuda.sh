#!/bin/sh
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --output=./slurm-output/slurm-%j.out
#SBATCH --error=./slurm-err/slurm-%j.err

# Consider using SBATCH --exclusive option outside of the class
# It ensures that no other user pollutes your measurements

source /share/env/module_select.sh
module purge

module load GCC/12.3.0
module load CUDA/12.4.0

echo "Compiling..."
output_file="$1"
SIZE="$2"

if [ ! -d "logs-${SIZE}" ]; then
    echo "Directory logs-${SIZE} does not exist. Creating it..."
    mkdir -p logs-${SIZE}
else
    echo "Directory logs-${SIZE} already exists."
fi

cd WA2-GPU

make clean && make SIZE=${SIZE}

echo "Start timing"

perf stat -x, ./bin/sparse_cuda > ../logs-${SIZE}/${output_file}.csv 2>&1

echo "Finished"
