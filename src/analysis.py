'''
analysis.py

Script used to analyse attributions of trained networks.

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

        # Outputs
        loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
        v_attr = attribution(trainer.inputs, loss, trainer.inputs[1])
        c_attr = attribution(trainer.inputs, loss, trainer.inputs[2])

        # Look for operators
        graph_kernels = {}
        graph_edges = {}
        angular_contr = {}
        graphconv_filters = {}
        presoftmax_out = trainer.outputs[-1]
        for op in sess.graph.get_operations():
            if 'graphconv' in op.name and op.name.endswith('vprime'):
                graphconv_filters[op.name] = op.values()

            if 'graphkernels' in op.name and len(op.name.split('/'))==1:
                if 'c_' in op.name:
                    graph_kernels[op.name] = op.values()
                else: graph_edges[op.name] = layer_attribution(trainer.inputs, loss, op.values()[0])

            if 'angularcontr' in op.name and len(op.name.split('/'))==1:
                angular_contr[op.name] = op.values()


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

            #if sample_id not in kras_list and sample_id not in hras_list: continue

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
            mask2 = np.repeat(np.expand_dims(mask,axis=-1), 29, axis=-1)
            mask = np.repeat(np.expand_dims(mask,axis=-1), 3, axis=-1)
            loss_val = activation(trainer.inputs, loss, sample, sess)

            # V Attribution
            '''
            attrib_v_ = []
            N = 50
            for k in range(1, N):
                sample_ = sample.copy()
                sample_[1][0] = sample_[1][0] * (k/N)
                attrib_v = sess.run(v_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample_)})[0]
                attrib_v_.append(attrib_v)
            attrib_v_ = np.array(attrib_v_)
            attrib_v = np.average(attrib_v_, axis=0) * sample[1][0] * mask2
            '''
            attrib_v = sess.run(v_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            attrib_v = attrib_v * sample[1][0] * mask2

            # C Attribution
            '''
            attrib_c_ = []
            for k in range(1, N):
                sample_ = sample.copy()
                sample_[2][0] = sample_[2][0] * (k/N)
                attrib_c = sess.run(c_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample_)})[0]
                attrib_c_.append(attrib_c)
            attrib_c_ = np.array(attrib_c_)
            attrib_c = np.average(attrib_c_, axis=0) * sample[2][0] * mask
            '''
            attrib_c = sess.run(c_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            attrib_c = attrib_c * sample[2][0] * mask

            #
            #vc_all = np.sum(attrib_v*1000, axis=-1)
            #vc_all = attrib_v*1000
            vc_all = np.sum(np.concatenate([attrib_v*1000, attrib_c*1000],axis=-1), axis=-1)
            vc_all = vc_all / np.max(np.abs(vc_all))
            attrib = np.array([ -(mask[:,0]-1), vc_all])


            # Graph Kernels For Input
            # GET MOST ACTIVATED KERNEL
            '''
            kernels = []
            highest_atr = 0.0
            high_op_name = None
            for gk in graph_edges.keys():
                atr = sess.run(graph_edges[gk], feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
                if atr > highest_atr:
                    highest_atr = atr
                    high_op_name = gk
            high_op_name = list(high_op_name)
            high_op_name[-3] = 'c'
            high_op_name = ''.join(high_op_name)
            '''

            kernels = []
            for gk in graph_kernels.keys():
                kernel = activation(trainer.inputs, graph_kernels[gk], sample, sess)[0]
                kernel = np.average(kernel[0],axis=-1)
                if len(kernel) < len(sample[1][0]):
                    s = len(sample[1][0])//len(kernel)
                    kernel = np.concatenate([np.expand_dims(kernel,axis=-1) for i in range(s)],axis=-1).flatten()
                else:
                    s = len(sample[1][0])%2
                    kernel = kernel[:-s]
                kernels.append(kernel)
            kernels = np.array(kernels)

            # Stored Example data
            if sample_class not in losses:
                losses[sample_class] = []
                losses[sample_class].append([sample_id, loss_val])
            else: losses[sample_class].append([sample_id, loss_val])
            attributions[sample_id] = [attrib,seq,ind,kernels]

    #
    kras_list = ['5tb5_c', '4dsn_a', '3gft_f', '5v6v_b', '4luc_b',
                 '4m22_a', '4lv6_b', '4epy_a', '4lrw_b', '4epx_a',
                 '4m1w_a', '5uqw_a', '4pzy_b', '4m21_a', '5us4_a',
                 '5f2e_a', '4q03_a', '4lyh_c', '4m1o_b', '4pzz_a']
    hras_list = ['1p2v_a', '4urz_r', '2x1v_a', '3lo5_c', '2quz_a',
                '3kud_a', '1aa9_a', '1plk_a', '4k81_d', '5wdq_a',
                '1iaq_a', '1xd2_a', '3i3s_r', '4efl_a', '4l9w_a',
                '3lo5_a', '5b2z_a', '1nvv_q', '4efm_a', '3l8z_a']

    # Plot Attributions for top n examples
    all_= []
    for key in losses.keys():
        losses_ = np.array(losses[key])
        top_ = losses_[:,1].astype('float').argsort()[:150]
        top_  = losses_[top_]

        y = []
        y_labels = []
        y_ticks = []
        temp = 0
        for i, _ in enumerate(top_):
            if _[0] not in kras_list and _[0] not in hras_list: continue
            x = attributions[_[0]][0]
            x = x[:,:160]
            y.append(x)
            y_ticks.append((temp)*2)
            y_labels.append(_[0]+":{:1.4f}".format(float(_[1])))
            temp +=1

        temp_ = np.sum(np.concatenate(y,axis=0),axis=0)
        temp_ = regress_ksegments(temp_,np.ones(temp_.size),25)
        temp_ = temp_ / np.max(np.abs(temp_))
        y = [np.expand_dims(temp_,axis=0)] + y
        y_ticks.append((len(y)-1)*2)
        y_labels = ["K-25"] + y_labels
        all_.append(temp_)

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
    kernels = []
    for key in attributions.keys():
        x = attributions[key][0].T
        data.append(x)
        label.append(key)
        offsets.append(attributions[key][2])
        kernels.append(attributions[key][3])
    data = np.array(data)
    label = np.array(label)
    offsets = np.array(offsets)
    kernels = np.array(kernels)
    all_ = np.array(all_)
    np.savez(output_dir+'/interpret/attributions.npz', data=data, labels=label, offsets=offsets, kernels=kernels, all_=all_)

    # Print some conclusions
    tf.keras.backend.clear_session()
    logging.info('All done!')
