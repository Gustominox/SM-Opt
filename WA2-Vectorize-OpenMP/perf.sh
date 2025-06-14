#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --partition=normal-x86
#SBATCH --account=f202500001hpcvlabepicurex
#SBATCH --output=./slurm-output/slurm-%j.out
#SBATCH --error=./slurm-err/slurm-%j.err

# Consider using SBATCH --exclusive option outside of the class
# It ensures that no other user pollutes your measurements

module --ignore_cache load GCC/13.3.0
source /share/apps-x86/ohpc/pub/apps/intel/oneapi/setvars.sh --force

echo "Compiling..."
output_file="$1"
SIZE="$2"
omp_num_threads=$3

cd WA2-Vectorize-OpenMP

if [ ! -d "../logs-${SIZE}" ]; then
    echo "Directory ../logs-${SIZE} does not exist. Creating it..."
    mkdir -p ../logs-${SIZE}
else
    echo "Directory ../logs-${SIZE} already exists."
fi

make SIZE=${SIZE} OUTFILE=${output_file}_${SIZE}

# Set the number of OpenMP threads
export OMP_NUM_THREADS=${omp_num_threads}  # You can adjust this based on the number of cores you want to use
export OMP_PROC_BIND=TRUE 
export OMP_PLACES=cores

echo "ompThreads=${OMP_NUM_THREADS}"

echo "Start timing"

perf stat -x, ./bin/${output_file}_${SIZE} > ../logs-${SIZE}/${output_file}.csv 2>&1

echo "Finished"
