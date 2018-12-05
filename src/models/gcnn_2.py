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

    def define_model(self, input_shape, diameter,
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
        self.inputs = [V, C]

        # Graph Convolutions
        A_ = LearnedENorm(V,A,10,diameter)
        for _ in list(zip(conv_layers,conv_dropouts)):
            __ = [GraphConv(V, [d,], _[0], batch_norm=True, activation=tf.nn.softsign) for d in A_]
            #V = tf.concat(__, axis=-1)
            V = __[0]
            for t in __[1:]: V += t
            V = V / len(__)
            V = tf.layers.dropout(V, float(_[1]))

        # Max Pooling
        if pooling_factor > 1:
            V = tf.layers.max_pooling1d(V,pooling_factor,pooling_factor)

        '''
        #print(tf.shape(V)[0], tf.shape(A)[0], tf.shape(A)[0]);exit()

        A__ = LearnedENorm(V_,A_,3)
        for _ in list(zip(conv_layers[1:],conv_dropouts[1:])):
            __ = [GraphConv(V_, [d,], int(_[0])//len(A__)) for d in A__]
            V_ = tf.concat(__, axis=-1)
            V_ = tf.layers.dropout(V_, float(_[1]))
        '''

        # Fully Connected Layers
        F = tf.contrib.layers.flatten(V)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]))

        # Outputs
        self.outputs = [F,]
