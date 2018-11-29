#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.out"
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH -t 00:30:00

# Load Modules
module load python/3.6-anaconda-4.4

# Variables
CORES = 64

srun -n $CORES python3 src/datagen/graphs/generate.py  -v data/KrasHras/
srun -n $CORES python3 src/datagen/volumes3d/generate.py  -v --range 55.717 data/KrasHras/
srun -n $CORES python3 src/datagen/pairwise3d/generate.py  -v --range 55.717 data/KrasHras/
