'''
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

residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
            'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
            'UNK', 'VAL']

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('generate_protein_pairwise2d.py')
    add_arg = parser.add_argument
    add_arg('datafolder', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--range', nargs='?', default=50.0, type=float)
    add_arg('--bins', nargs='?', default=10, type=int)

    return parser.parse_args()

def parse_pdb(path, chain, all_chains=False, first=False):
    '''
    '''
    # Parse residue, atom type and atomic coordinates
    protein_data = []
    res_i = 1
    chain_i = 1
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in lines:

            # Amino Level Information
            if row[:4] == 'ATOM':
                if not all_chains and row[21] != chain: pass
                else:
                    if row[12:17] in [' CA  ',' CA A']:
                        if row[17:20] not in residues: ress = 'UNK'
                        else: ress = row[17:20]
                        atom_data = [ress,
                                        row[30:38].strip(),
                                        row[38:46].strip(),
                                        row[47:54].strip()]
                        protein_data.append(atom_data)
                        res_i += 1
            if row[:3] == 'TER':
                if len(protein_data) > 0:
                    chain_i +=1
                    res_i = 0
                    if not all_chains or first: break

    protein_data = np.array(protein_data)
    if len(protein_data) == 0: return []

    return protein_data

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
    bin_x = []
    for r1 in residues:
        bin_y = []
        for r2 in residues:
            i = np.where(labels == r1+r2)
            H, bins = np.histogram(dist[i], bins=pairwise_distance_bins+1)
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
    all_chains = False # Generate graph for all chains found in PDB
    first = False # Collect only the first chain of in PDB
    verbose = args.verbose
    if first: all_chains=True
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
                if not os.path.exists(data_folder+'pairwise2d/'+row[0].lower()+ '_' + row[1].lower()  + '.npz'):
                    tasks.append([row[0],row[1]])

        if not os.path.exists(data_folder+'pairwise2d'): os.mkdir(data_folder+'pairwise2d')

        # Shuffle for Random Distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Fetch PDBs
    for t in tasks:

        # Task IDs
        pdb_id = t[0].lower()
        chain_id = t[1]
        filename = pdb_id + '_' + chain_id.lower()  + '.npz'

        # Use all chains
        if chain_id == '0': all_chains = True

        # Parse PDB
        if not os.path.exists(data_folder+'pdb/'+pdb_id+'.pdb'):
            if verbose: print("PDB not found: " + pdb_id+ '.pdb')
            continue
        protein_data = parse_pdb(data_folder+'pdb/'+pdb_id+'.pdb', chain_id, all_chains, first)
        if len(protein_data) == 0:
            if verbose: print("NO DATA: ", pdb_id, ',', chain_id)
            continue

        # Bin pairwise distances
        try:
            data = bin_pairwise_distances(protein_data, pairwise_distance_bins)
        except:
            if verbose: print("ERROR: ", pdb_id, ',', chain_id)
        del protein_data

        # Save data
        np.savez(data_folder+'pairwise2d/'+filename, data)
        if verbose: print('Generating: {} chain {}...'.format(pdb_id, chain_id))

        del data
