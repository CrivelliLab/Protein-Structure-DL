'''
gcnn.py

README:

This script defines a graph convolutional network for tensorflow.

'''
import numpy as np
import tensorflow as tf
from models import Model
from models.ops.graph_conv import *

class GCNN(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(GCNN, self).__init__()
        self.model_name = 'Graph-CNN'
        self.define_model(**kwargs)

    def define_model(self, input_shape, kernels_per_layer,
                        conv_layers, conv_dropouts, pooling_layers, fc_layers, fc_dropouts):
        '''
        Params:
            input_shape - list(int); [nb_nodes, nb_coords, nb_features]
            diameter - float; diameter limit of pairwise distances, set according to dataset
            conv_layers - list(tuples); list of tuples defining number of filters and dropouts
            fc_layers - list(tuples); list of tuples defining number of neurons and dropouts

        '''
        # Inputs
        V, C, A = VCAInput(input_shape[0], input_shape[2], input_shape[1])
        is_training = tf.placeholder_with_default(True, shape=())
        self.inputs = [is_training, V, C]

        # Graph Convolutions
        for _ in list(zip(kernels_per_layer,conv_layers,conv_dropouts,pooling_layers)):

            # Graph Kerenels
            A_ = GraphKernels(V,A,int(_[0]))

            # Preform Graph Covolution
            V = GraphConv(V, A_, int(_[1]))
            V = tf.nn.tanh(V)
            V = tf.layers.batch_normalization(V, training=is_training)
            V = tf.layers.dropout(V, float(_[2]), training=is_training)

            # Sequence Graph Pooling
            if int(_[3]) > 1: V,C,A = GraphPool(V,C,int(_[3]))


        # Fully Connected Layers
        F = tf.contrib.layers.flatten(V)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]), training=is_training)

        # Outputs
        self.outputs = [F,]
