# Deep Learning Exploration For Protein Structural Classification and Regression Tasks
Updated: 11/21/18

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
    name:
    data_path: str;
    task_type: str;
    nb_classes: int; *for classification*
    nb_nodes: int; *for protein graphs*
    split: list(float); *[training,validation,test] sums to 1*
    seed: int;

experiment_config:
    name: *trainer to use*
    output_dir: str;

model_config:
    model_type: str;
    optimizer: str;
    learning_rate: float;
    input_shape: list(int); *shape of input data*
    nb_classes: int; *for classification*

train_config:
    batch_size: int;
    nb_epochs: int;
    save_best: bool;
```

## Running Training
Once configuration file for training has been defined the following command is used
to run training. File loader has been paralleized using multithreading library and
number of cores to use can be set using the --cores flag.

`python3 src/main.py --cores $CORES config/config.yaml`
