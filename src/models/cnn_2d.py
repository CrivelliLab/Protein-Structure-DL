import numpy as np
import tensorflow as tf
from models import Model

class CNN2D(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(CNN2D, self).__init__()
        self.model_name = '2D-CNN'
        self.define_model(**kwargs)

    def define_model(self, input_shape, conv_layers, kernel_shapes, conv_dropouts,
                           pooling_layers, fc_layers, fc_dropouts):
        '''
        '''
        # Inputs
        X = tf.placeholder(tf.float32, [None,] + input_shape)
        is_training = tf.placeholder_with_default(True, shape=())
        self.inputs = [is_training,X,]

        # Network Defintion
        for _ in list(zip(conv_layers,kernel_shapes,conv_dropouts,pooling_layers)):
            X = tf.layers.conv2d(X, int(_[0]), int(_[1]))
            X = tf.layers.batch_normalization(X, training=is_training)
            X = tf.nn.relu(X)
            X = tf.layers.max_pooling2d(X, int(_[3]), int(_[3]))
            X = tf.layers.dropout(X, float(_[2]),training=is_training)

        # Fully Connected Layers
        F = tf.contrib.layers.flatten(X)
        for _ in list(zip(fc_layers,fc_dropouts)):
            F = tf.layers.dense(F, int(_[0]), activation=tf.nn.relu)
            F = tf.layers.dropout(F, float(_[1]), training=is_training)

        # Outputs
        self.outputs = [F,]
