#!/bin/bash -l
#SBATCH -C haswell
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encode.out"
#SBATCH --partition=regular
#SBATCH --nodes=4
#SBATCH -t 00:30:00

# Load Modules
module load python/3.6-anaconda-4.4

# Variables
CORES=128                              # Nodes * 32

# Run Job
srun -n $CORES python3 src/datagen/graphs/generate.py data/KrasHras/
srun -n $CORES python3 src/datagen/volumes3d/generate.py --range 55.717 data/KrasHras/
srun -n $CORES python3 src/datagen/pairwise2d/generate.py --range 55.717 data/KrasHras/
