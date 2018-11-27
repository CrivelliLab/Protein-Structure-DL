srun -n 64 python3 src/datagen/graphs/generate.py  -v data/Kinase/
srun -n 64 python3 src/datagen/volumes3d/generate.py  -v --range 55.717 data/Kinase/
srun -n 64 python3 src/datagen/pairwise3d/generate.py  -v --range 55.717 data/Kinase/
