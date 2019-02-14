'''
calc_GDTTS.py
Updated: 3/29/18

'''
import os, sys
import argparse
import numpy as np

# Parameters
lga_command = '-3 -ie -o0 -ch1:A -ch2:A -sda -d:4' # May need to change if decoy and target don't line up
################################################################################

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

def calculate_gdtts(pdb_path1, pdb_path2, chain_1, chain_2):
    '''
    '''

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists('TMP'): os.mkdir('TMP')
    if not os.path.exists('MOL2'): os.mkdir('MOL2')

    # Target path and Decoy path
    target_path = sys.argv[1]
    decoy_path = sys.argv[2]
    target_id = target_path.split('/')[-1].split('.')[0]
    decoy_id = decoy_path.split('/')[-1].split('.')[0]

    # Read Target PDB
    with open(target_path, 'r') as f: target_data = f.readlines()
    with open(decoy_path, 'r') as f: decoy_data = f.readlines()

    # Write combined PDB data
    with open('MOL2/'+decoy_id+'.'+target_id, 'w') as f:
        f.write('MOLECULE ' + decoy_id + '\n')
        f.writelines(decoy_data)
        if "END" not in decoy_data[-1]: f.write('END\n')
        f.write('MOLECULE ' + target_id + '\n')
        f.writelines(target_data)
        if "END" not in target_data[-1]: f.write('END\n')

    # Calculate GDT_TS using LGA
    lga_syscall = "LGA/lga.linux " + lga_command + ' ' + decoy_id+'.'+target_id
    os.system(lga_syscall)

    #
