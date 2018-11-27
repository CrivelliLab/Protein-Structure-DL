# Parameters
GPUS="0"

# Training commands
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/psiblast/psiblast_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/krashras/krashras_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/kinase/kinase_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/oncogenes/oncogenes_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/enzyme/enzyme_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/psiblast/psiblast_dilated_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/krashras/krashras_dilated_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/kinase/kinase_dilated_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/oncogenes/oncogenes_dilated_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/enzyme/enzyme_dilated_graph.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/psiblast/psiblast_pairwise.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/krashras/krashras_pairwise.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/kinase/kinase_pairwise.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/oncogenes/oncogenes_pairwise.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/enzyme/enzyme_pairwise.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/psiblast/psiblast_volumes.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/krashras/krashras_volumes.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/kinase/kinase_volumes.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/oncogenes/oncogenes_volumes.yaml
CUDA_VISIBLE_DEVICES=GPUS python3 src/main.py src/configs/enzyme/enzyme_volumes.yaml
