#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --output=./slurm-output/slurm-%j.out
#SBATCH --error=./slurm-err/slurm-%j.err

# Consider using SBATCH --exclusive option outside of the class
# It ensures that no other user pollutes your measurements

module --ignore_cache load GCC/12.3.0
module --ignore_cache load CUDA/12.4.0

echo "Compiling..."
output_file="$1"
SIZE="$2"

if [ ! -d "logs-${SIZE}" ]; then
    echo "Directory logs-${SIZE} does not exist. Creating it..."
    mkdir -p logs-${SIZE}
else
    echo "Directory logs-${SIZE} already exists."
fi

make SIZE=${SIZE} bin/${output_file}

echo "Start timing"

perf stat -x, ./bin/${output_file} > logs-${SIZE}/${output_file}.csv 2>&1

echo "Finished"
