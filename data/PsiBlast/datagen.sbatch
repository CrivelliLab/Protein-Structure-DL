srun -n 64 python3 src/datagen/graphs/generate.py  -v data/PsiBlast/
srun -n 64 python3 src/datagen/volumes3d/generate.py  -v --range 73.33 data/PsiBlast/
srun -n 64 python3 src/datagen/pairwise3d/generate.py  -v --range 73.33 data/PsiBlast/
