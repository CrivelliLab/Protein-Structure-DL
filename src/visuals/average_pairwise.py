import os
import numpy as np
import pptk as pk
import argparse

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('graph_conv.py')
    add_arg = parser.add_argument
    add_arg('data_path', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_path = args.data_path
    verbose = args.verbose

    # Load data
    x = np.random.rand((100,3))

    # Render data
    v = pptk.viewer(x)
    v.set(point_size=0.01)
