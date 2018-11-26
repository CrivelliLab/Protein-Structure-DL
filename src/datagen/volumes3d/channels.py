'''
channels.py
Updated: 1/9/18

This script contains methods which are passsed to data_generator object as
channels. Methods select indexes of data with the defined characteristic.

'''
import numpy as np

################################################################################

def all_atoms(data):
    '''
    Method returns indexes of all atoms.

    '''
    i = np.arange(0,len(data)).astype('int')
    return i

def residues():
    '''
    '''
    i = []
    i.append(lambda data : np.where(data[:,0] == 'ALA')[0])
    i.append(lambda data : np.where(data[:,0] == 'ARG')[0])
    i.append(lambda data : np.where(data[:,0] == 'ASN')[0])
    i.append(lambda data : np.where(data[:,0] == 'ASP')[0])
    i.append(lambda data : np.where(data[:,0] == 'ASX')[0])
    i.append(lambda data : np.where(data[:,0] == 'CYS')[0])
    i.append(lambda data : np.where(data[:,0] == 'GLN')[0])
    i.append(lambda data : np.where(data[:,0] == 'GLU')[0])
    i.append(lambda data : np.where(data[:,0] == 'GLX')[0])
    i.append(lambda data : np.where(data[:,0] == 'GLY')[0])
    i.append(lambda data : np.where(data[:,0] == 'HIS')[0])
    i.append(lambda data : np.where(data[:,0] == 'ILE')[0])
    i.append(lambda data : np.where(data[:,0] == 'LEU')[0])
    i.append(lambda data : np.where(data[:,0] == 'LYS')[0])
    i.append(lambda data : np.where(data[:,0] == 'MET')[0])
    i.append(lambda data : np.where(data[:,0] == 'PHE')[0])
    i.append(lambda data : np.where(data[:,0] == 'PRO')[0])
    i.append(lambda data : np.where(data[:,0] == 'SER')[0])
    i.append(lambda data : np.where(data[:,0] == 'THR')[0])
    i.append(lambda data : np.where(data[:,0] == 'TRP')[0])
    i.append(lambda data : np.where(data[:,0] == 'TYR')[0])
    i.append(lambda data : np.where(data[:,0] == 'UNK')[0])
    i.append(lambda data : np.where(data[:,0] == 'VAL')[0])

    return i

def alpha_carbons(data):
    '''
    Method returns indexes of alpha carbon atoms.

    '''
    i = np.where(data[:,1] == 'CA')
    return i

def beta_carbons(data):
    '''
    Method returns indexes of beta carbon atoms.

    '''
    i = np.where(data[:,1] == 'CB')
    return i


def aliphatic_res(data):
    '''
    Method returns indexes of atoms belonging to aliphatic amino acids.

    '''
    i = np.concatenate([np.where(data[:,0] == 'ALA')[0],
                        np.where(data[:,0] == 'ILE')[0],
                        np.where(data[:,0] == 'LEU')[0],
                        np.where(data[:,0] == 'MET')[0],
                        np.where(data[:,0] == 'VAL')[0]], axis=0)
    return i

def aromatic_res(data):
    '''
    Method returns indexes of atoms belonging to aromatic amino acids.

    '''
    i = np.concatenate([np.where(data[:,0] == 'PHE')[0],
                        np.where(data[:,0] == 'TRP')[0],
                        np.where(data[:,0] == 'TYR')[0]], axis=0)
    return i

def neutral_res(data):
    '''
    Method returns indexes of atoms belonging to neutral charged amino acids.

    '''
    i = np.concatenate([np.where(data[:,0] == 'ASN')[0],
                        np.where(data[:,0] == 'CYS')[0],
                        np.where(data[:,0] == 'GLN')[0],
                        np.where(data[:,0] == 'SER')[0],
                        np.where(data[:,0] == 'THR')[0]], axis=0)
    return i

def acidic_res(data):
    '''
    Method returns indexes of atoms belonging to acidic amino acids.

    '''
    i = np.concatenate([np.where(data[:,0] == 'ASP')[0],
                        np.where(data[:,0] == 'GLU')[0]], axis=0)
    return i

def basic_res(data):
    '''
    Method returns indexes of atoms belonging to basic amino acids.

    '''
    i = np.concatenate([np.where(data[:,0] == 'ARG')[0],
                        np.where(data[:,0] == 'HIS')[0],
                        np.where(data[:,0] == 'LYS')[0]], axis=0)
    return i

def unique_res(data):
    '''
    '''
    i = np.concatenate([np.where(data[:,0] == 'GLY')[0],
                        np.where(data[:,0] == 'PRO')[0]], axis=0)
    return i
