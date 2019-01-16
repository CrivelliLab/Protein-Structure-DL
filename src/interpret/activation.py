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

def generate_max_activation_sample(inputs, output_loss, sess, seed_sample=None):
    '''
    '''
    v_ = np.random.uniform(0, 1.0, shape=[1,]+inputs[1].shape[1:])
    c_ = np.random.uniform(-100, )

    opt = tf.train.AdamOptimizer(0.001).minimize(output_loss)
    #optimizer.compute_gradients(
