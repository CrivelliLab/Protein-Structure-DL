## Parameters
GPUS=0

### Training commands ###

## GCNN Trainings
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 1 src/configs/psiblast/psiblast_graph.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/krashras/krashras_graph.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/kinase/kinase_graph.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/oncogenes/oncogenes_graph.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/enzyme/enzyme_graph.yaml

## 2D CNN Trainings
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/psiblast/psiblast_pairwise.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/krashras/krashras_pairwise.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/kinase/kinase_pairwise.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/oncogenes/oncogenes_pairwise.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/enzyme/enzyme_pairwise.yaml

## 3D CNN Trainings
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/psiblast/psiblast_volumes.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/krashras/krashras_volumes.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/kinase/kinase_volumes.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/oncogenes/oncogenes_volumes.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/enzyme/enzyme_volumes.yaml

## New GCNN with updated graph kernels
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 2 src/configs/psiblast/psiblast_graph_new.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 2 src/configs/krashras/krashras_graph_new.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 8 src/configs/kinase/kinase_graph_new.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 8 src/configs/oncogenes/oncogenes_graph_new.yaml
CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 8 src/configs/enzyme/enzyme_graph_new.yaml

## CASP Structure Scoring Experiments
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/casp/casp_graph.yaml

## SARS Ligand Binding Affinity Regression
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/sars/sars_graph.yaml
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/main.py --cores 6 src/configs/sars/sars_pairwise.yaml
