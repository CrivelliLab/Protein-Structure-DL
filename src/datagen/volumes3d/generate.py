'''
generate.py

README:

This script is used to parallelize data generation using the data_generator object.
To generate a new dataset make sure to fetch pdbs first using the fetch_pdbs.py script.
Compressed numpy files for 3D representations will be produced for each
PDB entry in dataset.

'''
import os
import scipy
import argparse
import numpy as np
from mpi4py import MPI
from channels import *
from itertools import product
from data_generator import data_generator

# Data generator parameters
channels = residues()

################################################################################

residue_indexes = None # Select only atoms of these indexes
seed = 1234

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('generate.py')
    add_arg = parser.add_argument
    add_arg('datafolder', nargs='?', default='/')
    add_arg('--size', nargs='?', default=64, type=int)
    add_arg('--range', nargs='?', default=50.0, type=float)
    add_arg('-v', '--verbose', action='store_true')

    return parser.parse_args()

def write_binvox(file_path, np_array):
    '''
    Method write numpy array as binvox file.

    Params:
        file_path - str; path to save file
        np_array - np.array; volumetric image

    '''
    with open(file_path, 'wb') as fp:
        dims = np_array.shape
        scale = 1.0
        translate = 0
        voxels_flat = np_array.flatten()
        fp.write('#binvox 1\n'.encode('ascii'))
        fp.write(('dim '+' '.join(map(str, dims))+'\n').encode('ascii'))
        fp.write(('translate 0\n').encode('ascii'))
        fp.write(('scale 1.0\n').encode('ascii'))
        fp.write('data\n'.encode('ascii'))

        # keep a sort of state machine for writing run length encoding
        state = voxels_flat[0]
        ctr = 0
        for c in voxels_flat:
            if c==state:
                ctr += 1
                # if ctr hits max, dump
                if ctr==255:
                    fp.write(int(state).to_bytes(1,'big'))
                    fp.write(int(ctr).to_bytes(1,'big'))
                    ctr = 0
            else:
                # if switch state, dump
                fp.write(int(state).to_bytes(1,'big'))
                fp.write(int(ctr).to_bytes(1,'big'))
                state = c
                ctr = 1

        # flush out remainders
        if ctr > 0:
            fp.write(int(state).to_bytes(1,'big'))
            fp.write(int(ctr).to_bytes(1,'big'))

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_folder = args.datafolder
    if data_folder[-1] != '/': data_folder += '/'
    all_chains = False # Generate graph for all chains found in PDB
    size = args.size
    resolution = args.range / size
    verbose = args.verbose

    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    # MPI task distribution
    if rank == 0:
        tasks = []
        with open(data_folder+'data.csv', 'r')as f:
            for i, _ in enumerate(f):
                row = _[:-1].split(',')
                tasks.append([row[0],row[1]])

        if not os.path.exists(data_folder+'volume3d'): os.mkdir(data_folder+'volume3d')

        # Shuffle for Random Distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Intialize Data Generator
    pdb_datagen = data_generator(size=size, resolution=resolution, channels=channels)

    # Generate data for each task
    for i in range(3):

        # Parse task
        pdb_id = tasks[i][0].lower()
        chain_id = tasks[i][1]
        filename = pdb_id + '_' + chain_id.lower() + '.binvox'

        # Use all chains
        if chain_id == '0': all_chains = True

        # Set path to pdb file
        if not os.path.exists(data_folder+'pdb/'+pdb_id+'.pdb'):
            if verbose: print("PDB not found: " + pdb_id+ '.pdb')
            continue
        pdb_path = data_folder+'pdb/'+pdb_id+'.pdb'

        # Generate and Save Data
        pdb_data = pdb_datagen.generate_data(pdb_path, chain_id, 0, res_i=residue_indexes, all_chains=all_chains)

        # if data was generated without error
        if len(pdb_data) > 0:
            if verbose: print('Generating: {} chain {}...'.format(pdb_id, chain_id))

            # Get data for each dimensionality
            array_3d = pdb_data[0]
            write_binvox(data_folder+'volume3d/'+filename, array_3d.astype('bool'))
            del array_3d
        else:
            if verbose: print("NO DATA: ", pdb_id, ',', chain_id)
