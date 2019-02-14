'''
generate.py

README:
This script parses pdb for resiude types and (x,y,z) coordinates for construction
of protein graphs.

BUGS:
- There is a current bug with PDB files who's crystal structure residue indexes do not
match the sequence of the chain. These PDBs are not processed.

'''
import os
import numpy as np
import pandas as pd
from  scipy.spatial.distance import euclidean, cosine
from scipy.stats import percentileofscore as perc
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
    seq_data = []
    helix_data = []
    beta_data = []
    complex_data = {}
    protein_data = []
    res_ = None
    res_i = None
    res_c = None
    sidechain_data = []
    sidechain_flag = False
    sidechain_counter = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in lines:
            if row[:6] == 'SEQRES':
                row_ = row[:-1].split()
                if not all_chains and row_[2] != chain.upper(): continue
                for _ in row_[4:]:
                    try: ress = residues.index(_)
                    except: ress = residues.index('UNK')
                    seq_data.append([row_[2].upper(), ress])

            if row[:5] == 'HELIX':
                if not all_chains and row[19] != chain.upper(): continue
                helix_data.append([row[19], int(row[22:25]), int(row[34:37])])

            if row[:5] == 'SHEET':
                if not all_chains and row[21] != chain.upper(): continue
                beta_data.append([row[21], int(row[23:26]), int(row[34:37])])

            if row[:4] == 'ATOM':

                # Check if for chain
                if not all_chains and row[21] != chain.upper(): continue

                if res_i is None: res_i = row[22:26]

                if row[22:26] == res_i:

                    if row[12:17] in [' CA  ',' CA A']:
                        res_ = row[17:20]
                        res_c = [row[30:38].strip(), row[38:46].strip(), row[47:54].strip()]
                        sidechain_flag = True
                        sidechain_counter += 1
                    else:
                        if sidechain_flag:
                            if sidechain_counter > 2:
                                sidechain_data.append([row[30:38].strip(), row[38:46].strip(), row[47:54].strip()])
                            else:
                                sidechain_counter += 1

                else:
                    try: ress = residues.index(res_)
                    except: ress = residues.index('UNK')
                    if len(sidechain_data)> 0:
                        sidechain_data = np.array(sidechain_data).astype('float')
                        sidechain_c = np.mean(sidechain_data, axis=0).tolist()
                        sidechain_data = []
                    else:
                        sidechain_c = res_c
                    sidechain_flag = False
                    sidechain_counter = 0
                    if res_c is not None:
                        res_data = [res_i, ress] + res_c + sidechain_c
                        protein_data.append(res_data)
                    res_i = row[22:26]

            if row[:3] == 'TER':
                if sidechain_flag == True:
                    try: ress = residues.index(res_)
                    except: ress = residues.index('UNK')
                    if len(sidechain_data)> 0:
                        sidechain_data = np.array(sidechain_data).astype('float')
                        sidechain_c = np.mean(sidechain_data, axis=0).tolist()
                        sidechain_data = []
                    else:
                        sidechain_c = res_c
                    sidechain_flag = False
                    sidechain_counter = 0
                    if res_c is not None:
                        res_data = [res_i, ress] + res_c + sidechain_c
                        protein_data.append(res_data)

                if len(protein_data) > 0:
                    complex_data[row[21].upper()] = protein_data
                    protein_data = []
                    if not all_chains or first: break

    if len(complex_data)==0: return []
    # No Sequence Data
    if len(seq_data) < 1:
        chains_ = []
        ress_ = []
        for _ in complex_data:
            chains = np.array([' ' for i in range(len(_))])
            chains_.append(chains)
            ress_.append(_[:,1])
        chains_ = np.expand_dims(np.concatenate(chains_, axis=0),axis=-1)
        ress_ = np.expand_dims(np.concatenate(ress_, axis=0),axis=-1)
        seq_data = np.concatenate([chains_, ress_], axis=-1)

    # Set Sequence Data
    data = {}
    last_chain = -1
    temp = []
    ii = 1
    for i,_ in enumerate(seq_data):
        t = np.zeros((10))
        t[2] = _[1]
        t[1] = ii
        ii += 1
        if last_chain != _[0] or i+1 == len(seq_data):
            if i+1 == len(seq_data): temp.append(t)
            if len(temp) > 0:
                data[last_chain] = np.array(temp)
                temp = []
                ii = 0
            else: temp.append(t)
            last_chain = _[0]
        else:
            temp.append(t)

    '''
    last_chain = None
    temp_i = -1
    for i,_ in enumerate(helix_data):
        if last_chain != _[0]:
            last_chain = _[0]
            temp_i +=1
        data[temp_i][_[1]-1:_[2]-1,5] = 1

    last_chain = None
    temp_i = -1
    for i,_ in enumerate(beta_data):
        if last_chain != _[0].upper():
            last_chain = _[0].upper()
            temp_i +=1
        data[temp_i][_[1]-1:_[2]-1,6] = 1
    '''
    for i in data.keys():
        data[i] = data[i].astype('int').astype('str')

    for ii in complex_data.keys():
        chain_data = np.array(complex_data[ii])
        chain_c = chain_data[:,2:5].astype('float')
        chain_sc_c = chain_data[:,5:].astype('float')
        chain_centroid = np.mean(chain_c,axis=0)
        residue_depth = np.array([euclidean(chain_centroid, c) for c in chain_c])
        residue_depth_percentile = [1- perc(residue_depth, d)/100.0 for d in residue_depth]
        chain_c = chain_c - chain_centroid
        chain_sc_c = chain_sc_c - chain_centroid
        chain_sc_c = chain_sc_c - chain_c
        chain_c = -(chain_c)
        residue_orientation = [1-cosine(chain_c[i], chain_sc_c[i]) for i in range(len(chain_c))]

        if ii not in data: continue

        # Try First three res align
        offset = -1
        for j in range(len(chain_data)-3):
            for i, _ in enumerate(data[ii][:-3]):
                if data[ii][i:i+3,2].tolist() == chain_data[j:j+3, 1].tolist():
                    offset = int(data[ii][i,1]) - int(chain_data[j,0])
                    break
            if offset != -1: break

        if offset == -1:
            return []

        for i in range(len(chain_data)):
            ir = int(chain_data[i][0]) - 1 + offset
            if ir >= len(data[ii]): break
            if ir < 0: continue
            data[ii][ir,0] = 1
            data[ii][ir,3] = str(residue_depth_percentile[i])[:6]
            data[ii][ir,-3:] = chain_data[i,2:5]
            if np.isnan(residue_orientation[i]): data[ii][ir,4] = '0.0000'
            else: data[ii][ir,4] =  str(residue_orientation[i])[:6]

        tmp = 0
        for i, _ in enumerate(data[ii]):
            if data[ii][i,0] == '1': tmp = i
            else:
                data[ii][i,-3:] = data[ii][tmp,-3:]

    data = np.concatenate([data[ii] for ii in data.keys()], axis=0)
    if len(data) == 0: return []

    return data

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
    fails = 0
    for t in tasks:

        # Task IDs
        pdb_id = t[0].lower()
        chain_id = t[1]
        filename = pdb_id + '_' + chain_id.lower() + '.txt'

        # Use all chains
        if verbose: print('Generating: {} chain {}...'.format(pdb_id, chain_id))
        if chain_id == '0': all_chains = True

        # Parse PDB
        if not os.path.exists(data_folder+'pdb/'+pdb_id+'.pdb'):
            fails += 1
            if verbose: print("PDB not found: " + pdb_id+ '.pdb')
            continue
        protein_data = parse_pdb(data_folder+'pdb/'+pdb_id+'.pdb', chain_id, all_chains, first)
        if len(protein_data) == 0:
            fails += 1
            if verbose: print("NO DATA: ", pdb_id, ',', chain_id)
            continue
        prime_lens.append(len(protein_data))
        dia = protein_data[:,-3:].astype('float')
        dia = dia - np.mean(dia, axis=0)
        dia = np.max(np.abs(dia))*2
        diameters.append(dia)

        # Save graph
        with open(data_folder+'graph/'+filename, 'w') as f:
            for i, _ in enumerate(protein_data):
                f.write(' '.join(_)+'\n')

    # Print graph size stats
    prime_lens = np.expand_dims(prime_lens, axis=-1).astype('int')
    diameters = np.expand_dims(diameters, axis=-1)
    stats = np.concatenate([prime_lens, diameters], axis=-1)

    # Gather stats
    stats_ = comm.gather(stats,root=0)
    fails_ = comm.gather(fails,root=0)

    if rank == 0:
        if verbose: print("NUMBER OF FAILED GENERATIONS: ", sum(fails_))
        stats = np.concatenate(stats_, axis=0)
        df = pd.DataFrame(stats)
        df.columns = ['nb_residues','diameters']
        summary = df.describe(percentiles=[0.1*i for i in range(10)])
        summary.to_csv(data_folder+'/graph.summary', float_format='%1.6f', sep=',')
