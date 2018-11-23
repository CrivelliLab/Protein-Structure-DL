import numpy as np
import tensorflow as tf
from models import Model
from models.modules.layers import *

class Transformer(Model):

    def __init__(self, **kwargs):
        '''
        '''
        super(GCNN, self).__init__()
        self.model_name = 'Transformer'
        self.define_model(**kwargs)

    def define_model(self, input_shape):
        '''
        '''
        # Inputs
        X = tf.placeholder(tf.float32, [None,] + input_shape)
        self.inputs = [X]

        # Network Defintion
        MHA1 =

        # Fully Connected Layers
        F1 = tf.contrib.layers.flatten(MHA1)
        D1 = tf.layers.dense(F1, 128, activation=tf.nn.relu)
        D1 = tf.layers.dropout(D1, 0.5)

        # Outputs
        self.outputs = [D1,]
