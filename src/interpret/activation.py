'''
'''
import numpy as np
import tensorflow as tf

################################################################################

def activation(inputs, outputs, data, sess):
    '''
    '''
    _ = sess.run(outputs, feed_dict={i: d for i, d in zip(inputs, data)})
    return _

def generate_max_activation_sample(inputs, outputs, seed_sample=None):
    '''
    '''
    pass
