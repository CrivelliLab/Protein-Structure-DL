'''
generate.py

'''
import os
import numpy as np
import argparse
from mpi4py import MPI
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist
from itertools import combinations


################################################################################

# Static Parameters
seed = 458762 # For random distribution of tasks using MPI
atoms = ['C','H','O','N','S','UNK']
body = ['L','R','UNK']

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('generate.py')
    add_arg = parser.add_argument
    add_arg('datafolder', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--range', nargs='?', default=50.0, type=float)
    add_arg('--bins', nargs='?', default=10, type=int)

    return parser.parse_args()

def parse_file(path):
    '''
    '''
    # Parse atom type and atomic coordinates
    data = []
    with open(path, 'r')as f:
        for i, _ in enumerate(f):
            row = _[:-1].split()
            a = row[0]
            b = row[1]

            if a not in atoms: a = 'UNK'
            if b not in body: b = 'UNK'
            atom_data = [a+b,] + row[2:5]

            data.append(atom_data)

    data = np.array(data)
    if len(data) == 0: return []

    return data

def bin_pairwise_distances(protein_data, pairwise_distance_bins):
    '''
    Method bins pairwise distances of residue alpha carbons into 2D data grids.
    Params:
        protein_data - np.array;
        pairwise_distance_bins - list; list of bins used to bin pairwise distances
    Returns:
        binned_pairwise - np.array;
    '''
    # Pairwise distances
    dist = np.array(pdist(protein_data[:,-3:].astype('float')))
    labels = list(combinations(protein_data[:,0],2))
    labels = np.array([i[0] + i[1] for i in labels])

    # Bin pairwise distances
    combos = []
    for i in atoms:
        for j in body:
            combos.append(i+j)

    bin_x = []
    for r1 in combos:
        bin_y = []
        for r2 in combos:
            i = np.where(labels == r1+r2)
            H, bins = np.histogram(dist[i], bins=pairwise_distance_bins)
            H = gaussian_filter(H, 0.5)
            bin_y.append(H)
        bin_x.append(bin_y)
    binned_pairwise = np.array(bin_x)

    del dist
    del labels

    return binned_pairwise

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_folder = args.datafolder
    if data_folder[-1] != '/': data_folder += '/'
    verbose = args.verbose
    pairwise_distance_bins = np.arange(0, args.range, args.range/args.bins)

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
                if not os.path.exists(data_folder+'pairwise2d_protlig/'+row[0].lower()+ '.npz'):
                    tasks.append(row[0])

        if not os.path.exists(data_folder+'pairwise2d_protlig'): os.mkdir(data_folder+'pairwise2d_protlig')

        # Shuffle for Random Distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Load site
    site_data = parse_file(data_folder+'site.txt')

    # Fetch PDBs
    for t in tasks:

        # Task IDs
        file_path = data_folder+'ligands/'+t+'.txt'
        save_path = data_folder+'ligands/'+t+'.txt'

        # Parse PDB
        if not os.path.exists(file_path):
            if verbose: print("File not found: ", file_path)
            continue
        ligand_data = parse_file(file_path)
        if len(protein_data) == 0:
            if verbose: print("NO DATA: ", file_path)
            continue

        # Bin pairwise distances
        try:
            protein_data = np.concatenate([site_data, ligand_data])
            data = bin_pairwise_distances(protein_data, pairwise_distance_bins)
        except:
            if verbose: print("ERROR: ", file_path)
        del protein_data

        # Save data
        np.savez(save_path, data)
        if verbose: print('Generating: ', save_path)
        del data
