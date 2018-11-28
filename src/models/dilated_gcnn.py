'''
dilated_gcnn.py

README:

This script defines a dilated graph convolutional network for tensorflow.

'''
import numpy as np
import tensorflow as tf
from models import Model
from models.modules.layers import *

class DilatedGCNN(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(DilatedGCNN, self).__init__()
        self.model_name = 'Dilated_Graph-CNN'
        self.define_model(**kwargs)

    def define_model(self, input_shape, diameter,
                        conv_layers, conv_dropouts, pooling_factor, fc_layers, fc_dropouts, dilations):
        '''
        Params:
            input_shape - list(int); [nb_nodes, nb_coords, nb_features]
            diameter - float; diameter limit of pairwise distances, set according to dataset
            conv_layers - list(tuples); list of tuples defining number of filters and dropouts
            fc_layers - list(tuples); list of tuples defining number of neurons and dropouts

        '''
        # Inputs
        V, C, A = VCAInput(input_shape[0], input_shape[2], input_shape[1],
                                        e_radius=diameter, dialations=dilations)
        self.inputs = [V, C]

        # Graph Convolutions
        for _ in list(zip(conv_layers,conv_dropouts)):
            __ = [GraphConv(V, [d,], int(_[0])//len(A)) for d in A]
            V = tf.concat(__, axis=-1)
            V = tf.layers.dropout(V, float(_[1]))

        # Max Pooling
        if pooling_factor > 1:
            V = tf.layers.max_pooling1d(V,pooling_factor,pooling_factor)

        # Fully Connected Layers
        F = tf.contrib.layers.flatten(V)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]))

        # Outputs
        self.outputs = [F,]
