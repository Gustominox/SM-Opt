#!/bin/sh
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202500001hpcvlabepicureg

# Consider using SBATCH --exclusive option outside of the class
# It ensures that no other user pollutes your measurements

module load GCC/13.3.0
module load CUDA/12.4.0

./bin/stencil