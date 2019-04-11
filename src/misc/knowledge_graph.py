'''
'''
import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import get_datasets, DataLoader
from trainers import get_trainer
from interpret.activation import *
from interpret.attribution import *
from trainers.base_trainer import bg
import matplotlib.pyplot as plt
from models.ops.graph_conv import *
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

################################################################################
residues = ['A', 'R', 'N', 'D', 'N', 'C', 'Q',
            'E', 'G', 'G', 'H', 'I', 'L', 'K',
            'M', 'F', 'P', 'S', 'T', 'W', 'Y',
            '?', 'V']

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

def align_protein(data, alignment):
    '''
    '''
    # Algin KrasHras
    ind = 0
    for j in range(len(data[1][0])-2):
        args = np.argmax(data[1][0][j:j+3,:23],axis=-1).tolist()
        if args == alignment:
            ind = j
            break
    if ind > 0:
        data[1][0] = np.concatenate([data[1][0][ind:], np.zeros((ind, data[1][0].shape[1]))],axis=0)
        data[2][0] = np.concatenate([data[2][0][ind:], np.zeros((ind, data[2][0].shape[1]))],axis=0)
        temp = np.zeros(data[3][0].shape)
        temp[:temp.shape[0]-ind, :temp.shape[1]-ind] = data[3][0][ind:, ind:]
        data[3][0] = temp

    return data


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

        # Input Attributions

        # Outputs
        loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])

        # Input Outputs Attributions
        input_shape = model_config['input_shape']
        network_attributions = {}
        #network_attributions['v'] = [spatial_attribution(trainer.inputs, loss, trainer.inputs[1]), [], ['v_'+str(j) for j in range(input_shape[0])]]
        #network_attributions['c'] = [spatial_attribution(trainer.inputs, loss, trainer.inputs[2]), [], ['c_'+str(j) for j in range(input_shape[0])]]
        vc = spatial_attribution(trainer.inputs, loss, trainer.inputs[2]) + spatial_attribution(trainer.inputs, loss, trainer.inputs[1])
        network_attributions['vc'] = [vc, [], ['vc_'+str(j) for j in range(input_shape[0])]]

        #['cos_a'] = [layer_attribution(trainer.inputs, loss, ]), [], ['cd_root']]

        # Layer attribution
        pgm_edges = []
        last_kernel_dim = 0
        last_graphconv_dim = 0

        for op in sess.graph.get_operations():
            if 'graphconv' in op.name and op.name.endswith('vprime/batchnorm/add_1'):
                l = int(op.name.split('_')[1])
                d = op.values()[0].shape[-1]
                ns = ['gc_'+str(l)+'_'+str(i) for i in range(d)]
                network_attributions[op.name] = [channel_attribution(trainer.inputs, loss, op.values()[0]), [], ns]
                if l > 0:
                    for i in range(d):
                        #pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'cd_'+str(l-1)))
                        for j in range(last_kernel_dim):
                            pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'gk_'+str(l-1)+'_'+str(j)))

                        for j in range(last_graphconv_dim):
                            pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'gc_'+str(l-1)+'_'+str(j)))
                else:
                    for i in range(d):
                        #pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'cd_root'))
                        for j in range(input_shape[0]):
                            pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'vc_'+str(j)))
                            #pgm_edges.append(('gc_'+str(l)+'_'+str(i), 'c_'+str(j)))

                last_graphconv_dim = d
                last_kernel_dim = 0
            '''
            if 'graphkernels_' in op.name and 'k_' in op.name and len(op.name.split('/'))==1:
                l = int(op.name.split('_')[1])
                d = int(op.name.split('_')[-1])
                ns = ['gk_'+str(l)+'_'+str(d),]
                network_attributions[op.name] = [layer_attribution(trainer.inputs, loss, op.values()[0]), [], ns]
                if l > 0:
                    for j in range(last_graphconv_dim):
                        pgm_edges.append(('gk_'+str(l)+'_'+str(d), 'gc_'+str(l-1)+'_'+str(j)))
                else:
                    #pgm_edges.append(('gk_'+str(l)+'_'+str(d), 'cd_root'))
                    for j in range(input_shape[0]):
                        pgm_edges.append(('gk_'+str(l)+'_'+str(d), 'v_'+str(j)))
                        pgm_edges.append(('gk_'+str(l)+'_'+str(d), 'c_'+str(j)))

                last_kernel_dim += 1
            '''
            #if 'averseq' in op.name and op.name.endswith('cospdist_a'):
                #l = int(op.name.split('_')[1])
                #network_attributions[op.name] = [layer_attribution(trainer.inputs, loss, op.values()[0]), [], ['cd_'+str(l-1),]]

        network_attributions['output'] = [tf.argmax(tf.nn.softmax(trainer.outputs[-1],axis=-1),axis=-1), [], ['class']]

        for j in range(last_graphconv_dim):
            pgm_edges.append(('class', 'gc_'+str(l)+'_'+str(j)))

        '''
        for j in range(input_shape[0]):
            pgm_edges.append(('class', 'vc_'+str(j)))
            #pgm_edges.append(('vc_'+str(j), 'class'))
            #pgm_edges.append(('class', 'c_'+str(j)))

        '''

        print("NUMBER OF EDGES IN PGM: ", len(pgm_edges))

        # Fetch Example
        trainer.logger.info('Analyzing...')
        for i, _ in enumerate(data_loader):
            # Get Sample
            data = [False,] + _
            data_id = data_loader.dataset.data[i][0].split('/')[-1][:-4]
            data_class = data_loader.dataset.data[i][1]
            data = align_protein(data, [18,7,20])


            # Input Attribution To Classification
            for key in network_attributions.keys():
                nl = network_attributions[key]
                atr = sess.run(nl[0], feed_dict={i: d for i, d in zip(trainer.inputs, data)})[0]
                if key == 'output':
                    nl[1].append(np.array([atr,]))
                    continue
                atr = atr / np.max(np.abs(atr))
                atr[atr >= 0.5] = 1
                atr[atr < 0.5] = 0
                nl[1].append(atr.astype('int'))

        columns = []
        data = []
        for key in network_attributions.keys():
            a = network_attributions[key]
            columns += a[-1]
            data.append(np.array(a[1]))
        data = np.concatenate(data, axis=-1)
        print(columns)
        print(len(columns), data.shape)
        l = len(data[0])
        data = pd.DataFrame(np.array(data), columns=columns)
        #z = np.zeros((1,l))
        #o = np.ones((1,l))
        #dz = pd.DataFrame(z, columns=columns)
        #data = data.append(dz)
        #dz = pd.DataFrame(o, columns=columns)
        #data = data.append(dz)
        #z[:,-1] = 1
        #o[:,-1] = 0
        #dz = pd.DataFrame(z, columns=columns)
        #ata = data.append(dz)
        #dz = pd.DataFrame(o, columns=columns)
        #data = data.append(dz)


        # Calibrate all CPDs of `model` using MLE:
        model = BayesianModel(pgm_edges)
        model.fit(data, estimator=MaximumLikelihoodEstimator)

        for cpd in model.get_cpds():
            print("CPD of {variable}:".format(variable=cpd.variable))
            print(cpd)
            print(cpd.values)
            print(cpd.variables)

        exit()
        # Show Probalilites
        plt.matshow(inference, cmap='seismic', vmin=-1, vmax=1)
        plt.title('PGM Inference' , y=1.15)
        plt.xlabel("Nodes")
        plt.colorbar()
        plt.show()
