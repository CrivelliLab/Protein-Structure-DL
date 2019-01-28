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
                graph_kernels[op.name] = op.values()

            if 'angularcontr' in op.name and len(op.name.split('/'))==1:
                angular_contr[op.name] = op.values()

        # Fetch Example
        trainer.logger.info('Analyzing...')

        # Graph Conv Filter Max Activation Sample

        # Classification Max Activation Sample
        #loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
        #attrib = channel_attribution(trainer.inputs, loss, graphconv_filters[gf][0], sample, sess)[0]
        k = 10
        v_atr = {}
        c_atr = {}
        vc_atr = {}
        losses = {}
        l2pdist =  L2PDist(trainer.inputs[-2])
        cpdist = CosinePDist(trainer.inputs[-2])
        loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
        v_attr = attribution(trainer.inputs, loss, trainer.inputs[1]) #sample, sess
        c_attr = attribution(trainer.inputs, loss, trainer.inputs[2])
        for i, _ in enumerate(data_loader):
            sample = [False,] + _
            sample_id = data_loader.dataset.data[i][0].split('/')[-1][:-4]
            sample_class = data_loader.dataset.data[i][1]

            loss_val = activation(trainer.inputs, loss, sample, sess)

            # PDist and CosinePDist

            a = activation(trainer.inputs, l2pdist, sample, sess)[0]
            a__ = a / np.max(a)
            a = np.exp(-(a__*a__))
            '''
            plt.matshow(a__, cmap='seismic')
            plt.title('A Normalized by Max:' + ' : ' + sample_id + ' : ' + sample_class)
            plt.xlabel("Nodes")
            plt.ylabel("Nodes")
            plt.colorbar()
            plt.show()
            '''

            ca = activation(trainer.inputs, cpdist , sample, sess)[0]
            '''
            plt.matshow(ca, cmap='seismic')
            plt.title('Cosine A' + ' : ' + sample_id + ' : ' + sample_class)
            plt.xlabel("Nodes")
            plt.ylabel("Nodes")
            plt.colorbar()
            plt.show()
            '''

            emp = np.expand_dims(np.sum(sample[1][0],axis=-1),axis=-1)
            emp[emp>0] = 1
            a = emp*a
            a = emp.T*a
            ca = ca*emp
            ca = ca*emp.T
            a_ = np.average(a, axis=-1) * np.average(ca, axis=-1)
            '''
            # Graph Kernels For Input
            for gk in graph_kernels.keys():
                kernel = activation(trainer.inputs, graph_kernels[gk], sample, sess)[0]
                kernel = np.exp(-(kernel*kernel))
                plt.matshow(kernel[0], cmap='seismic', vmin=0, vmax=1)
                plt.title(gk + ' : ' + sample_id + ' : ' + sample_class)
                plt.xlabel("Nodes")
                plt.ylabel("Nodes")
                plt.colorbar()
                plt.show()

            # Angular Contribution For Input
            for ac in angular_contr.keys():
                contr = activation(trainer.inputs, angular_contr[ac], sample, sess)[0]
                plt.matshow(contr[0], cmap='seismic', vmin=0, vmax=1)
                plt.title(ac + ' : ' + sample_id + ' : ' + sample_class)
                plt.xlabel("Nodes")
                plt.ylabel("Nodes")
                plt.colorbar()
                plt.show()
            '''

            # Input Attribution To Classification
            attrib_v = _ = sess.run(v_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            #attrib_v = attrib_v * sample[1][0]
            attrib_v = np.sum(attrib_v,axis=-1)
            attrib_v = regress_ksegments(attrib_v, np.ones(attrib_v.shape), k)
            attrib_v = attrib_v / np.max(np.abs(attrib_v))
            if sample[-1][0][0] not in v_atr:
                v_atr[sample[-1][0][0]]=[]
                v_atr[sample[-1][0][0]].append(attrib_v)
            else: v_atr[sample[-1][0][0]].append(attrib_v)
            #attrib_v = np.transpose(np.expand_dims(attrib_v,axis=-1))
            #attrib_v = np.concatenate([attrib_v for j in range(20)], axis=0)
            '''
            plt.matshow(attrib_v, cmap='seismic', vmin=-1, vmax=1)
            plt.title('Input V Attribution : ' + sample_id + ' : ' + sample_class, y=1.15)
            plt.xlabel("Nodes")
            plt.colorbar()
            plt.show()
            '''

            attrib_c = sess.run(c_attr, feed_dict={i: d for i, d in zip(trainer.inputs, sample)})[0]
            #attrib_c = attrib_c * np.concatenate([np.expand_dims(a_,axis=-1) for j in range(attrib_c.shape[-1])], axis=-1)
            attrib_c = np.sum(attrib_c,axis=-1)
            #attrib_c = np.sum(attrib_c[0],axis=-1)
            attrib_c = regress_ksegments(attrib_c, np.ones(attrib_c.shape), k)
            attrib_c = attrib_c / np.max(np.abs(attrib_c))
            if sample[-1][0][0] not in c_atr:
                c_atr[sample[-1][0][0]]=[]
                c_atr[sample[-1][0][0]].append(attrib_c)
            else: c_atr[sample[-1][0][0]].append(attrib_c)
            #attrib_c = np.transpose(np.expand_dims(attrib_c,axis=-1))
            #attrib_c = np.concatenate([attrib_c for j in range(20)], axis=0)

            attrib_vc = (attrib_c + attrib_v)/2.0
            if sample[-1][0][0] not in vc_atr:
                vc_atr[sample[-1][0][0]]=[]
                vc_atr[sample[-1][0][0]].append(attrib_vc)
            else: vc_atr[sample[-1][0][0]].append(attrib_vc)

            if sample[-1][0][0] not in losses:
                losses[sample[-1][0][0]]=[]
                losses[sample[-1][0][0]].append([sample_id, loss_val])
            else: losses[sample[-1][0][0]].append([sample_id, loss_val])

            '''
            plt.matshow(attrib_c, cmap='seismic', vmin=-1, vmax=1)
            plt.title('Input C Attribution : ' + sample_id + ' : ' + sample_class, y=1.15)
            plt.xlabel("Nodes")
            plt.colorbar()
            plt.show()
            '''

            continue
            '''

            # Graph Kernels Attribution to Classifciation
            print('Graph Kernel Attribution to Classifciations')
            for gk in graph_kernels.keys():
                loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
                attrib = layer_attribution(trainer.inputs, loss, graph_kernels[gk][0], sample, sess)[0]
                print(gk, ':', attrib)

            # Graph Conv Filter Attribution to Classifciation
            print('Graph Filter Attribution to Classifciations')
            filters_ = {}
            for gf in graphconv_filters.keys():
                loss = tf.losses.softmax_cross_entropy(trainer.inputs[-1], trainer.outputs[-1])
                attrib = channel_attribution(trainer.inputs, loss, graphconv_filters[gf][0], sample, sess)[0]
                attrib = (attrib - np.min(attrib)) / (np.max(attrib) - np.min(attrib))
                attrib_c = (2*attrib_c) - 1
                for j,a in enumerate(attrib): print(gf, ':', j, ':', a)
                filters_[gf] = [np.argmax(attrib), np.argmin(attrib)]

            # Input Attribution To Graph Filters
            for gf in graphconv_filters.keys():
                filters = tf.split(graphconv_filters[gf],[1 for k in range(graphconv_filters[gf][0].shape[-1])],axis=-1)
                for k,f in enumerate(filters):
                    if k in filters_[gf]:
                        loss = tf.ones_like(f) - f
                        attrib_v = attribution(trainer.inputs, loss, trainer.inputs[1], sample, sess)[0]
                        attrib_v = attrib_v * sample[1][0]
                        attrib_v_ = np.transpose(np.expand_dims(np.average(attrib_v,axis=-1),axis=-1))
                        attrib_v = (attrib_v_ - np.min(attrib_v_)) / (np.max(attrib_v_) - np.min(attrib_v_))
                        attrib_v = (2*attrib_v) - 1
                        attrib_v = np.concatenate([attrib_v for j in range(20)], axis=0)

                        attrib_c = attribution(trainer.inputs, loss, trainer.inputs[2], sample, sess)[0]
                        attrib_c = np.transpose(np.expand_dims(np.sum(attrib_c,axis=-1),axis=-1))
                        attrib_c_ = attrib_c*a_#np.transpose(np.expand_dims(np.sum(sample[1][0],axis=-1),axis=-1))
                        attrib_c = (attrib_c_ - np.min(attrib_c_)) / (np.max(attrib_c_) - np.min(attrib_c_))
                        attrib_c = (2*attrib_c) - 1
                        attrib_c = np.concatenate([attrib_c for j in range(20)], axis=0)

                        combined = attrib_c_+ attrib_v_
                        combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))
                        combined = (2*combined) - 1
                        combined = np.concatenate([combined for j in range(20)], axis=0)

                        plt.matshow(attrib_v, cmap='seismic', vmin=-1, vmax=1)
                        plt.title('Input V Attribution : '+gf + ' : ' +str(k) + ' : '+ sample_id + ' : ' + sample_class, y=1.75)
                        plt.xlabel("Nodes")
                        plt.colorbar()
                        plt.show()
                        plt.matshow(attrib_c, cmap='seismic', vmin=-1, vmax=1)
                        plt.title('Input C Attribution : '+gf + ' : ' +str(k) + ' : '+ sample_id + ' : ' + sample_class, y=1.75)
                        plt.xlabel("Nodes")
                        plt.colorbar()
                        plt.show()

                        plt.matshow(combined, cmap='seismic')
                        plt.title('Input V*C Attribution : '+gf + ' : ' +str(k) + ' : '+ sample_id + ' : ' + sample_class, y=1.75)
                        plt.xlabel("Nodes")
                        plt.colorbar()
                        plt.show()
        '''
        #Plot Range
        for key in c_atr.keys():
            x = list(range(len(c_atr[key][0])))
            y = np.array(c_atr[key])
            plt.plot(x,np.zeros((len(x))), color='black')
            plt.plot(x,np.average(y,axis=0), color='black', linestyle='dashed')
            plt.plot(x,np.average(y,axis=0)-np.std(y,axis=0), color='black')
            plt.plot(x,np.average(y,axis=0)+np.std(y,axis=0), color='black')
            plt.fill_between(x,np.zeros((len(x))),np.ones((len(x))), color='red', alpha=0.4)
            plt.fill_between(x,-1*np.ones((len(x))),np.zeros((len(x))), color='blue', alpha=0.4)
            plt.ylim(-1,1)
            plt.title('C'+str(key))
            plt.show()

        for key in v_atr.keys():
            x = list(range(len(v_atr[key][0])))
            y = np.array(v_atr[key])
            plt.plot(x,np.zeros((len(x))), color='black')
            plt.plot(x,np.average(y,axis=0), color='black', linestyle='dashed')
            plt.plot(x,np.average(y,axis=0)-np.std(y,axis=0), color='black')
            plt.plot(x,np.average(y,axis=0)+np.std(y,axis=0), color='black')
            plt.fill_between(x,np.zeros((len(x))),np.ones((len(x))), color='red', alpha=0.4)
            plt.fill_between(x,-1*np.ones((len(x))),np.zeros((len(x))), color='blue', alpha=0.4)
            plt.ylim(-1,1)
            plt.title('V'+str(key))
            plt.show()

        for key in vc_atr.keys():
            x = list(range(len(vc_atr[key][0])))
            y = np.array(vc_atr[key])
            plt.plot(x,np.zeros((len(x))), color='black')
            plt.plot(x,np.average(y,axis=0), color='black', linestyle='dashed')
            plt.plot(x,np.average(y,axis=0)-np.std(y,axis=0), color='black')
            plt.plot(x,np.average(y,axis=0)+np.std(y,axis=0), color='black')
            plt.fill_between(x,np.zeros((len(x))),np.ones((len(x))), color='red', alpha=0.4)
            plt.fill_between(x,-1*np.ones((len(x))),np.zeros((len(x))), color='blue', alpha=0.4)
            plt.ylim(-1,1)
            plt.title('VC:'+str(key))
            plt.show()

        for key in losses.keys():
            x = list(range(len(vc_atr[key][0])))
            losses_ = np.array(losses[key])
            top_10 = losses_[:,1].astype('float').argsort()[:10]
            for _ in top_10:
                y = np.array(v_atr[key][_])
                y = np.transpose(np.expand_dims(y,axis=-1))
                y = np.concatenate([y for j in range(15)], axis=0)
                plt.matshow(y, cmap='seismic', vmin=-1, vmax=1)
                plt.title('Input VC Attribution : ' + losses_[_,0] + ' : ' + str(key), y=1.15)
                plt.xlabel("Nodes")
                plt.colorbar()
                plt.show()


    # Print some conclusions
    tf.keras.backend.clear_session()
    logging.info('All done!')
