'''
gcnn.py

README:


This script defines a graph convolutional network for tensorflow.

'''
import numpy as np
import tensorflow as tf
from models import Model
from models.ops.graphconv import *

class GCNN(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(GCNN, self).__init__()
        self.model_name = 'Graph-CNN'
        self.define_model(**kwargs)

    def define_model(self, input_shape, dilations,
                        conv_layers, conv_dropouts, pooling_factor, fc_layers, fc_dropouts):
        '''
        Params:
            input_shape - list(int); [nb_nodes, nb_coords, nb_features]
            diameter - float; diameter limit of pairwise distances, set according to dataset
            conv_layers - list(tuples); list of tuples defining number of filters and dropouts
            fc_layers - list(tuples); list of tuples defining number of neurons and dropouts

        '''
        # Inputs
        V, C, A = VCAInputVanilla(input_shape[0], input_shape[2], input_shape[1])
        is_training = tf.placeholder_with_default(True, shape=())
        self.inputs = [is_training, V, C]

        # Graph Convolutions
        A_ = LearnedENorm(V,A,dilations)

        '''
        for _ in list(zip(conv_layers,conv_dropouts)):
            V = GraphConv(V, A_, int(_[0]), dropout=float(_[1]))
        '''

        for _ in list(zip(conv_layers,conv_dropouts)):
            __ = [GraphConv(V, [d,], _[0], batch_norm=False, activation=None) for d in A_]
            #V = tf.concat(__, axis=-1)
            V = __[0]
            for t in __[1:]: V += t
            V = V / len(__)

            V = tf.nn.tanh(V)
            V = tf.layers.batch_normalization(V, training=is_training)
            V = tf.layers.dropout(V, float(_[1]), training=is_training)


        # Max Pooling
        if pooling_factor > 1:
            V = tf.layers.max_pooling1d(V,pooling_factor,pooling_factor)

        '''

        for _ in list(zip(conv_layers,conv_dropouts)):
            __ = [GraphConv(V, [d,], int(_[0])//len(A_)) for d in A_]
            V = tf.concat(__, axis=-1)
            V = tf.layers.dropout(V, float(_[1]))
        '''


        # Fully Connected Layers
        F = tf.contrib.layers.flatten(V)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]), training=is_training)

        # Outputs
        self.outputs = [F,]
