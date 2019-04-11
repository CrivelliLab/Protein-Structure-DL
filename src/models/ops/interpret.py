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

def attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    return var_grad

def attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    return var_grad
    
def spatial_attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    attribution = var_grad * layer
    spatial_attribution = tf.reduce_mean(attribution, axis=-1)
    return spatial_attribution

def channel_attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    attribution = var_grad * layer
    channel_attribution = tf.reduce_mean(attribution, axis=-2)
    return channel_attribution

def layer_attribution(inputs, output_loss, layer):
    '''
    '''
    var_grad = tf.gradients(output_loss, [layer])[0]
    attribution = var_grad * layer
    layer_attribution = tf.reduce_mean(tf.reduce_mean(attribution, axis=-1),axis=-1, keep_dims=True)

    return layer_attribution
