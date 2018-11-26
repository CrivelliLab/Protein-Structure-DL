srun -n 128 python3 src/datagen/graphs/generate.py  -v data/Enzyme/
srun -n 128 python3 src/datagen/volumes3d/generate.py  -v --range 0.0 data/Enzyme/
srun -n 128 python3 src/datagen/pairwise3d/generate.py  -v --range 0.0 data/Enzyme/
