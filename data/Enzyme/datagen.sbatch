srun -n 64 python3 src/datagen/graphs/generate.py  -v data/OncoGenes/
srun -n 64 python3 src/datagen/volumes3d/generate.py  -v --range 85.230 data/OncoGenes/
srun -n 64 python3 src/datagen/pairwise3d/generate.py  -v --range 85.230 data/OncoGenes/
