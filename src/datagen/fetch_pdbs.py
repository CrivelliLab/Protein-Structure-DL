'''
fetch_pdbs.py
Updated: 1/19/17

Script fetches pdbs from Protein Data Bank as defined in class.csv files in
data_folder path. class.csv files includes pdb identifier and chain identifier
pairs. PDBs are save into folders corresponding to class and PDB identifier.

Script parallelizes fetches over multiple cores using MPI.

'''
import os, wget
import numpy as np
import argparse
from mpi4py import MPI

###############################################################################

# For random distribution of tasks using MPI
seed = 1234

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('generate_protein_graphs.py')
    add_arg = parser.add_argument
    add_arg('datafolder', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')

    return parser.parse_args()

def fetch_PDB(path, pdb_id):
    '''
    Method fetches pdb file from Protein Data Bank repo and stores file in
    designated path.

    '''
    # Download PDB file from database.
    url = 'https://files.rcsb.org/download/' # URL used to fetch PDB files
    file_path = path + pdb_id.lower() + '.pdb'
    if not os.path.exists(file_path):
        file_path = wget.download(url + pdb_id + '.pdb', out=path, bar=None)

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_folder = args.datafolder
    if data_folder[-1] != '/': data_folder += '/'

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
                tasks.append(row[0])

        if not os.path.exists(data_folder+'pdb'): os.mkdir(data_folder+'pdb')

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
        pdb_id = t

        # Fetch PDB file and rename with task IDs
        fetch_PDB(data_folder+'pdb', pdb_id)
        print('Fetched: ' + pdb_id)
