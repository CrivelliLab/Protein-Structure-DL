'''
'''
import os
import argparse
import logging
import yaml
import numpy as np
import tensorflow as tf
from datasets import get_datasets, DataLoader
from trainers import get_trainer
from interpret.activation import *
from interpret.attribution import *
from trainers.base_trainer import bg
import matplotlib.pyplot as plt
from models.ops.graph_conv import *

################################################################################

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('main.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--cores', nargs='?', default=1, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f: config = yaml.load(f)
    data_config = config['data_config']
    model_config = config.get('model_config', {})
    train_config = config['train_config']
    experiment_config = config['experiment_config']
    output_dir = experiment_config.pop('output_dir', None)

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if not os.path.exists(output_dir+'/interpret'): os.makedirs(output_dir+'/interpret')
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(),
                        logging.FileHandler(output_dir+'/interpret/analysis.log')])
    logging.info('Initializing')

    # Load the datasets
    data_config['split'] = [1.0,0,0]
    data_config['augment'] = 1
    batch_size = 1
    dataset, _, __ = get_datasets(**data_config); del _; del __;
    data_loader = DataLoader(dataset, batch_size=batch_size, cores=args.cores)
    logging.info('Loaded %g samples', len(dataset))

    # Load the trainer
    trainer = get_trainer(output_dir=output_dir, **experiment_config)

    # Build the model
    trainer.build_model(**model_config)
    trainer.print_model_summary()
    trainer.saver = tf.train.Saver()

    # Run analysis of model

    with tf.Session() as sess:

        # Setup the variable initialisation
        sess.run(tf.global_variables_initializer())
        sess.run(trainer.running_vars_initializer)
        trainer.saver.restore(sess, output_dir+"/model/model.ckpt")

        # Look for operators
        graph_kernels = {}
        angular_contr = {}
        graphconv_filters = {}
        presoftmax_out = trainer.outputs[-1]
        for op in sess.graph.get_operations():
            if 'graphconv' in op.name and op.name.endswith('vprime'):
                graphconv_filters[op.name] = op.values()

            if 'graphkernels' in op.name and len(op.name.split('/'))==1:
                graph_kernels[op.name] = op.values()

            if 'angularcontr' in op.name and len(op.name.split('/'))==1:
                angular_contr[op.name] = op.values()

        # Fetch Example
        trainer.logger.info('Analyzing...')
        for i, _ in enumerate(data_loader):
            sample = [False,] + _
            sample_id = data_loader.dataset.data[i][0].split('/')[-1][:-4]
            sample_class = data_loader.dataset.data[i][1]

            # PDist and CosinePDist
            a = activation(trainer.inputs, L2PDist(trainer.inputs[-2]), sample, sess)[0]
            plt.matshow(a, cmap='inferno')
            plt.title('A' + ' : ' + sample_id + ' : ' + sample_class)
            plt.xlabel("Nodes")
            plt.ylabel("Nodes")
            plt.colorbar()
            plt.show()

            ca = activation(trainer.inputs, CosinePDist(trainer.inputs[-2]), sample, sess)[0]
            plt.matshow(ca, cmap='inferno')
            plt.title('cA' + ' : ' + sample_id + ' : ' + sample_class)
            plt.xlabel("Nodes")
            plt.ylabel("Nodes")
            plt.colorbar()
            plt.show()

            # Graph Kernels For Input
            for gk in graph_kernels.keys():
                kernel = activation(trainer.inputs, graph_kernels[gk], sample, sess)[0]
                kernel = np.exp(-(kernel*kernel))
                plt.matshow(kernel[0], cmap='inferno', vmin=0, vmax=1)
                plt.title(gk + ' : ' + sample_id + ' : ' + sample_class)
                plt.xlabel("Nodes")
                plt.ylabel("Nodes")
                plt.colorbar()
                plt.show()

            # Angular Contribution For Input
            for ac in angular_contr.keys():
                contr = activation(trainer.inputs, angular_contr[ac], sample, sess)[0]
                plt.matshow(contr[0], cmap='inferno', vmin=0, vmax=1)
                plt.title(ac + ' : ' + sample_id + ' : ' + sample_class)
                plt.xlabel("Nodes")
                plt.ylabel("Nodes")
                plt.colorbar()
                plt.show()

            # Input Attribution To Graph Filters

            # Input Attribution To Classification

            # Graph Kernels Attribution to Classifciation

            # Graph Conv Filter Attribution to Classifciation

            # Graph Conv Filter Max Activation Sample

            # Classification Max Activation Sample

    # Print some conclusions
    tf.keras.backend.clear_session()
    logging.info('All done!')
