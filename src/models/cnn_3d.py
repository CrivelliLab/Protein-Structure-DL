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

    def define_model(self, input_shape):
        '''
        '''
        # Inputs
        X = tf.placeholder(tf.float32, [None,] + input_shape)
        self.inputs = [X,]

        # Network Defintion
        C1 = tf.layers.conv3d(X, 32, 5, activation=tf.nn.relu)
        P1 = tf.layers.max_pooling3d(C1, 2, 2)
        C2 = tf.layers.conv3d(P1, 32, 5, activation=tf.nn.relu)
        P2 = tf.layers.max_pooling3d(C2, 2, 2)
        C3 = tf.layers.conv3d(P2, 32, 5, activation=tf.nn.relu)
        P3 = tf.layers.max_pooling3d(C3, 2, 2)

        # Fully Connected Layers
        F1 = tf.contrib.layers.flatten(P3)
        D1 = tf.layers.dense(F1, 128, activation=tf.nn.relu)
        D1 = tf.layers.dropout(D1, 0.5)

        # Outputs
        self.outputs = [D1,]
