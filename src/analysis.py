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
from models.ops.interpret import *
from trainers.base_trainer import bg
import matplotlib.pyplot as plt
from models.ops.graph_conv import *
import matplotlib.ticker as ticker

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

def prepare_ksegments(series,weights):
    '''
    '''
    N = len(series)
    #
    wgts = np.diag(weights)
    wsum = np.diag(weights*series)
    sqrs = np.diag(weights*series*series)

    dists = np.zeros((N,N))
    means = np.diag(series)

    for i in range(N):
        for j in range(N-i):
            r = i+j
            wgts[j,r] = wgts[j,r-1] + wgts[r,r]
            wsum[j,r] = wsum[j,r-1] + wsum[r,r]
            sqrs[j,r] = sqrs[j,r-1] + sqrs[r,r]
            means[j,r] = wsum[j,r] / wgts[j,r]
            dists[j,r] = sqrs[j,r] - means[j,r]*wsum[j,r]

    return dists, means

def regress_ksegments(series, weights, k):
    '''
    '''
    N = len(series)

    dists, means = prepare_ksegments(series, weights)

    k_seg_dist = np.zeros((k,N+1))
    k_seg_path = np.zeros((k,N))
    k_seg_dist[0,1:] = dists[0,:]

    k_seg_path[0,:] = 0
    for i in range(k):
        k_seg_path[i,:] = i

    for i in range(1,k):
        for j in range(i,N):
            choices = k_seg_dist[i-1, :j] + dists[:j, j]
            best_index = np.argmin(choices)
            best_val = np.min(choices)

            k_seg_path[i,j] = best_index
            k_seg_dist[i,j+1] = best_val

    reg = np.zeros(series.shape)
    rhs = len(reg)-1
    for i in reversed(range(k)):
        lhs = k_seg_path[i,rhs]
        reg[int(lhs):rhs] = means[int(lhs),rhs]
        rhs = int(lhs)

    return reg

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
                if 'c_' in op.name:
                    graph_kernels[op.name] = op.values()

            if 'angularcontr' in op.name and len(op.name.split('/'))==1:
                angular_contr[op.name] = op.values()

        # Outputs
        loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
        v_attr = attribution(trainer.inputs, loss, trainer.inputs[1])
        c_attr = attribution(trainer.inputs, loss, trainer.inputs[2])

        # Fetch Example
        k = 10
        losses = {}
        attributions = {}
        trainer.logger.info('Analyzing...')
        for i, _ in enumerate(data_loader):
            sample = [False,] + _
            sample_id = data_loader.dataset.data[i][0].split('/')[-1][:-4]
            sample_class = data_loader.dataset.data[i][1]
            mask = np.sum(sample[3][0],axis=-1)
            mask[mask<2] = 0
            mask[mask>0] = 1

            i_ = 0
            for t in range(len(mask)):
                if mask[t] == 1:
                    i_ = t
                    break

            # Algin KrasHras
            start_res = [18,7,20] #14
            ind = 0
            for j in range(len(sample[1][0])-2):
                args = np.argmax(sample[1][0][j:j+3,:23],axis=-1).tolist()
                if args == start_res:
                    ind = j
                    break
            if ind > 0:
                mask = np.concatenate([mask[ind:], np.zeros((ind))],axis=0)
                sample[1][0] = np.concatenate([sample[1][0][ind:], np.zeros((ind, sample[1][0].shape[1]))],axis=0)
                sample[2][0] = np.concatenate([sample[2][0][ind:], np.zeros((ind, sample[2][0].shape[1]))],axis=0)
                temp = np.zeros(sample[3][0].shape)
                temp[:temp.shape[0]-ind, :temp.shape[1]-ind] = sample[3][0][ind:, ind:]
                sample[3][0] = temp

            # Get sequence
            seq = []
            for j in range(len(sample[1][0])):
                if np.sum(sample[1][0][j,:23]) == 0:
                    l = '-'
                else:
                    ind_ = np.argmax(sample[1][0][j,:23])
                    l = residues[ind_]
                seq.append(l)
            ind = ind - i_


            # Input Attribution To Classification
            mask = np.repeat(np.expand_dims(mask,axis=-1), 3, axis=-1)
            loss_val = activation(trainer.inputs, loss, sample, sess)
            attrib_v = _ = sess.run(v_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            attrib_v = attrib_v * sample[1][0]
            attrib_c = sess.run(c_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            attrib_c = attrib_c * sample[2][0] * mask
            vc_all = np.sum(np.concatenate([attrib_v*1000, attrib_c*1000],axis=-1), axis=-1)
            attrib = np.array([ -(mask[:,0]-1), vc_all])

            '''
            # Graph Kernels For Input
            kernels = []
            for gk in graph_kernels.keys():
                kernel = activation(trainer.inputs, graph_kernels[gk], sample, sess)[0]
                kernel
            '''

            # Stored Example data
            if sample_class not in losses:
                losses[sample_class] = []
                losses[sample_class].append([sample_id, loss_val])
            else: losses[sample_class].append([sample_id, loss_val])
            attributions[sample_id] = [attrib,seq,ind]

    # Plot Attributions for top n examples
    for key in losses.keys():
        losses_ = np.array(losses[key])
        top_ = losses_[:,1].astype('float').argsort()[:150]
        top_  = losses_[top_]

        y = []
        y_labels = []
        y_ticks = []
        temp = 0
        for i, _ in enumerate(top_):
            x = attributions[_[0]][0]
            x = x[:,:160]
            y.append(x)
            y_ticks.append((i-temp)*2)
            y_labels.append(_[0]+":{:1.4f}".format(float(_[1])))

        fig, ax = plt.subplots()
        ax.matshow(np.concatenate(y,axis=0), cmap='seismic', vmin=-1, vmax=1)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_ticks([int(10*j) for j in range(x.shape[-1]//10)])
        ax.set_xticklabels([str(int(10*j)+2) for j in range(x.shape[-1]//10)])
        ax.xaxis.set_minor_locator(ticker.FixedLocator([z for z in range(len(x[0]))]))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(attributions[_[0]][1]))
        ax.tick_params(axis="x", which="minor", direction="out",
                   top=1, bottom=0, labelbottom=0, labeltop=1, labelsize=6)
        ax.yaxis.set_ticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('PDB:[Loss]')
        ax.set_xlabel('Residue')
        plt.show()

    # Save Interpretation data
    data = []
    label = []
    offsets = []
    for key in attributions.keys():
        x = attributions[key][0].T
        data.append(x)
        label.append(key)
        offsets.append(attributions[key][2])
    data = np.array(data)
    label = np.array(label)
    offsets = np.array(offsets)
    np.savez(output_dir+'/interpret/attributions.npz', data=data, labels=label, offsets=offsets)

    # Print some conclusions
    tf.keras.backend.clear_session()
    logging.info('All done!')
