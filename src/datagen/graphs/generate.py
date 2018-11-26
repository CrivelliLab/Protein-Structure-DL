'''
generate.py

README:
This script parses pdb for resiude types and (x,y,z) coordinates for construction
of protein graphs.

'''
import os
import numpy as np
import pandas as pd
import argparse
from mpi4py import MPI

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
    parser = argparse.ArgumentParser('generate_protein_graphs.py')
    add_arg = parser.add_argument
    add_arg('datafolder', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')

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
                        atom_data = [int(row[23:26]), chain_i, res_i,
                                        residues.index(row[17:20]),
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

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_folder = args.datafolder
    if data_folder[-1] != '/': data_folder += '/'
    all_chains = False # Generate graph for all chains found in PDB
    first = False # Collect only the first chain of in PDB
    verbose = args.verbose
    if first: all_chains=True

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

        if not os.path.exists(data_folder+'graph'): os.mkdir(data_folder+'graph')

        # Shuffle for Random Distribution
        np.random.seed(seed)
        np.random.shuffle(tasks)

    else: tasks = None

    # Broadcast tasks to all nodes and select tasks according to rank
    tasks = comm.bcast(tasks, root=0)
    tasks = np.array_split(tasks, cores)[rank]

    # Fetch PDBs
    prime_lens = []
    diameters = []
    for t in tasks:

        # Task IDs
        pdb_id = t[0]
        chain_id = t[1]
        filename = pdb_id + '_' + chain_id + '.txt'

        # Use all chains
        if chain_id == '0': all_chains = True

        # Parse PDB
        if not os.path.exists(data_folder+'pdb/'+pdb_id+'.pdb'): continue
        protein_data = parse_pdb(data_folder+'pdb/'+pdb_id+'.pdb', chain_id, all_chains, first)
        if len(protein_data) == 0: continue
        prime_lens.append(len(protein_data))
        dia = protein_data[:,-3:].astype('float')
        dia = dia - dia.mean(axis=0)
        diameters.append(np.max(np.abs(dia))*2)

        # Save graph
        with open(data_folder+'graph/'+filename, 'w') as f:
            for i, _ in enumerate(protein_data):
                f.write(' '.join(_)+'\n')
        if verbose: print('Generating: {} chain {}...'.format(pdb_id, chain_id))

    # Print graph size stats
    prime_lens = np.expand_dims(prime_lens, axis=-1).astype('int')
    diameters = np.expand_dims(diameters, axis=-1)
    stats = np.concatenate([prime_lens, diameters], axis=-1)

    # Gather stats
    stats_ = comm.gather(stats,root=0)

    if rank == 0:
        stats = np.concatenate(stats_, axis=0)
        df = pd.DataFrame(stats)
        df.columns = ['nb_residues','diameters']
        summary = df.describe(percentiles=[0.1*i for i in range(10)])
        summary.to_csv(data_folder+'/graph.summary', float_format='%1.6f', sep=',')
