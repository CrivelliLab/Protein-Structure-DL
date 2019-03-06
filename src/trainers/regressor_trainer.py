'''
regressor_trainer.py

Script defines trainer for regression tasks. Trainer uses mean squared error as
loss function. The following metrics are defined for trainier:

    - r2

'''
import time
import numpy as np
import tensorflow as tf
from .base_trainer import BaseTrainer, bg
from models import get_model

class RegressorTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(RegressorTrainer, self).__init__(**kwargs)

    def build_model(self, model_type='gcnn', optimizer='Adam', learning_rate=0.001, nb_classes=0, **model_args):
        '''
        Method intanstiates tensorflow model and metrics for regression tasks.

        Params:
            model_type - str;
            optimizer - str;
            learning_rate - float;
            model_args - arguments for model

        '''
        # Retrieve Model
        self.model = get_model(name=model_type, **model_args)
        self.inputs, self.outputs = self.model.get_inputs_outputs()

        # Add ouput regressor layer
        Y = tf.placeholder(tf.float32, [None,1])
        self.inputs.append(Y)
        y_out = tf.layers.dense(self.outputs[0], 1)

        # Add Loss and Optimizers
        loss = tf.losses.mean_squared_error(Y, y_out)
        if optimizer == 'Adam':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else: raise Exception('Optimizer %s unknown' % optimizer)

        # Metrics
        total_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.reduce_mean(Y))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y, y_out)))
        r2 = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
        self.operators = [opt, loss, r2]

        # Metric Vars
        running_vars = []
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        # Get number of trainable parameters
        self.nb_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            x = 1
            for _ in shape: x *= int(_)
            self.nb_parameters += x

    def train_epoch(self, data_loader, sess):
        '''
        Method defines regressor training epoch.

        Params:
            data_loader - DataLoader; training examples generator
            sess - TF.sess;

        Return:
            summary - dict; training results

        '''
        # Summary
        summary = dict()
        loss = []
        r2 = []
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Training...')
        for i, data in enumerate(bg(data_loader)):
            data = [True,] + data
            out = sess.run(self.operators, feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[1])
            r2.append(out[2])

        # Get metrics
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = np.average(loss)
        summary['train_r2'] = np.average(r2)

        # Log Epoch results
        self.logger.info('train time: %.3f' % summary['train_time'])
        self.logger.info('train loss: %.3f' % summary['train_loss'])
        self.logger.info('train r2: %.3f' % summary['train_r2'])

        return summary

    def evaluate(self, data_loader, sess, mode='valid'):
        '''
        Method defines evaluation epoch for validation and test examples.

        Params:
            data_loader - DataLoader; validation or test examples generator
            sess - tf.Session
            mode - str; evalutation mode either 'valid' or 'test'

        Return:
            summary - dict; evaluation results

        '''
        # Summary
        summary = dict()
        loss = []
        r2 = []
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Evaluating...')
        for i, data in enumerate(data_loader):
            data = [False,] + data
            out = sess.run(self.operators[1:], feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[0])
            r2.append(out[1])

        # Get Metrics
        summary[mode+'_time'] = time.time() - start_time
        summary[mode+'_loss'] = np.average(loss)
        summary[mode+'_r2'] = np.average(r2)

        # Log epoch results
        self.logger.info(mode+' time: %.3f' % summary[mode+'_time'])
        self.logger.info(mode+' loss: %.3f' % summary[mode+'_loss'])
        self.logger.info(mode+' r2: %.3f' % summary[mode+'_r2'])

        return summary
