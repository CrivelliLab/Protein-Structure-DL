# Deep Learning Exploration For Protein Structural Classification and Regression Tasks
Updated: 3/5/19

## Generating Data
This project explores three different data representations for proteomic data
and corresponding neural network architectures:
3D volumetric images, 2D pairwise statistical images and protein graphs.
In order to generate a new dataset, the following setup steps are required:

1. Create a folder for the new dataset under `/data/` in the project directory.
2. Create a file named `data.csv` within the newly created folder. This file should
list the dataset examples and respective classification or regression values. Formating
should be as follows:

```
data.csv

pdb_id, chain_id, (class number or regression value)

Example:
3gft, A, 1

```

> Note: If all chains within a pdb should be used chain_id, should be set to 0.

3. Gather PDB files for dataset using the following command:

`python3 src/datagen/fetch_pdbs.py path_to_data_folder`

4. Once PDB files for a data set have been downloaded, use one of the following
commands to generate one of the three different types of data representations:

- Volumetric images - `python3 src/datagen/volumes3d/generate.py path_to_data_folder`
This generates 3D voxel images of protein structures with each residue type defined
as a separate channel. The data is stored in binvox format to reduce file size.

> Note: image size and resolution can be adjusted using the --size and
> --resolution flags respectively. Default: 64(^3) and 1.0 angstroms

- Pairwise images - `python3 src/datagen/pairwise2d/generate.py path_to_data_folder`
This generates 2d histogram images of the pairwise distances between residue types.

> Note: histogram range and bins can be adjusted using the --range and --bins
flags respectively. Default: 50 angstroms and 10 bins

- Protein graphs - `python3 src/datagen/graphs/generate.py path_to_data_folder`
This generates graph representations of protein structures with nodes encoding
residues and edges encoding pairwise distances.

> All data generation scripts can be run in parallel using MPI.
> EX: $mpirun -n $NODES python3 src/datagen/.........

## Defining Tensorflow Training Experiment
Under `/src/configs/` are .yaml files defining neural network training configurations.
Depending on the type of model, model config parameters may vary. For further information,
please read model documentation. Each file has the following general fields:

```
data_config:
    name: protien_graphs
    data_path: data/PsiBlast
    task_type: classification
    nb_classes: 2
    nb_nodes: 381
    split: [0.7,0.1,0.2]
    augment: 3
    fuzzy_radius: 0.25
    seed: 1234

experiment_config:
    name: classifier
    output_dir: out/psiblast_graph_new

model_config:
    model_type: gcnn
    input_shape: [381,29,3]
    kernel_limit: 126.0
    kernels_per_layer: [2,2]
    conv_layers: [64,64]
    conv_dropouts: [0.1,0.1]
    pooling_layers: [4,4]
    fc_layers: [128,]
    fc_dropouts: [0.5,]
    nb_classes: 2
    optimizer: 'Adam'
    learning_rate: 0.0001

train_config:
    batch_size: 20
    nb_epochs: 100
    early_stop_metric: valid_loss
    early_stop_epochs: 10
    save_best: True
```

## Running Training
Once configuration file for training has been defined the following command is used
to run training. File loader has been paralleized using multithreading library and
number of cores to use can be set using the --cores flag.

`python3 src/main.py --cores $CORES config/config.yaml`

## GCNN Model and Layers
A tensorflow definition of a graph convolutional neural network (GCNN) can be found within
the /src/models/ folder. Layers used in the model are defined in /src/models/ops/graph_conv.py
