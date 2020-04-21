## Parameters
GPUS=0
DATA_LOADER_CORES=1

### Interpretation commands ###

## NEW GCNN
#CUDA_VISIBLE_DEVICES=$GPUS python3 src/knowledge_graph.py --cores $DATA_LOADER_CORES src/configs/krashras/krashras_graph_new.yaml
CUDA_VISIBLE_DEVICES=$GPUS python3 src/analysis.py --cores $DATA_LOADER_CORES src/configs/krashras/krashras_graph_new.yaml
