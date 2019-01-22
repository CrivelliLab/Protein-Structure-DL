'''
'''
import numpy as np
import tensorflow as tf
from trainers.base_trainer import bg

################################################################################

def attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    return var_grad

def channel_attribution(inputs, output_loss, layer, data, sess):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    linear_atrribution = var_grad * layer
    linear_atrribution = tf.reduce_mean(linear_atrribution, axis=1)
    _ = sess.run(linear_atrribution, feed_dict={i: d for i, d in zip(inputs, data)})
    return _

def layer_attribution(inputs, output_loss, layer, data, sess):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    linear_atrribution = var_grad * layer
    linear_atrribution = tf.reduce_mean(linear_atrribution, axis=-1)
    linear_atrribution = tf.reduce_mean(linear_atrribution, axis=-1)
    _ = sess.run(linear_atrribution, feed_dict={i: d for i, d in zip(inputs, data)})
    return _
