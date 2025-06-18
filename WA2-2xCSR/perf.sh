#!/bin/sh
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --partition=normal-x86
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --output=./slurm-output/slurm-%j.out
#SBATCH --error=./slurm-err/slurm-%j.err

# Consider using SBATCH --exclusive option outside of the class
# It ensures that no other user pollutes your measurements

module load GCC/13.3.0
source /share/apps-x86/ohpc/pub/apps/intel/oneapi/setvars.sh --force

output_file="$1"
SIZE="$2"

if [ ! -d "./logs-${SIZE}" ]; then
    echo "Directory ./logs-${SIZE} does not exist. Creating it..."
    mkdir -p ./logs-${SIZE}
else
    echo "Directory ./logs-${SIZE} already exists."
fi

echo "Compiling..."

cd WA2-2xCSR
# make clean && 
# make clean && make SIZE=${SIZE}

echo "Start timing"

# time -v ./bin/sparse > ../logs/${output_file}.csv 2>&1
perf stat -x, ./bin/sparse > ../logs-${SIZE}/${output_file}.csv 2>&1

echo "Finished"
