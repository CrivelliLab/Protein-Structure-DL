import numpy as np
import tensorflow as tf
from models import Model

class CNN3D(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(CNN3D, self).__init__()
        self.model_name = '3D-CNN'
        self.define_model(**kwargs)

    def define_model(self, input_shape,
                        conv_layers, kernel_shapes, conv_dropouts, fc_layers, fc_dropouts):
        '''
        '''
        # Inputs
        X = tf.placeholder(tf.float32, [None,] + input_shape)
        self.inputs = [X,]

        # Network Defintion
        for _ in list(zip(conv_layers,kernel_shapes,conv_dropouts)):
            X = tf.layers.conv3d(X, int(_[0]), int(_[1]))
            X = tf.layers.batch_normalization(X)
            X = tf.nn.relu(X)
            X = tf.layers.max_pooling3d(X, 2, 2)
            X = tf.layers.dropout(X, float(_[2]))

        # Fully Connected Layers
        F = tf.contrib.layers.flatten(X)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]))

        # Outputs
        self.outputs = [F,]
