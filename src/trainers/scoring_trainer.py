'''
**UNDERCONSTRUCTION**
scoring_trainer.py

Script defines trainer for scoring tasks. Trainer uses mean squared error as
loss function. The following metrics are defined for trainier:

    - average loss
    - median loss
    - loss range

TODO:
- Sampling decoys has not yet been implemented.

'''
import time
import numpy as np
import tensorflow as tf
from .base_trainer import BaseTrainer, bg
from models import get_model

class ScoringTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(ScoringTrainer, self).__init__(**kwargs)

    def build_model(self, model_type='gcnn', optimizer='Adam', learning_rate=0.001, **model_args):
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

        # Add ouput Scoring layer
        Y = tf.placeholder(tf.float32, [None,1])
        self.inputs.append(Y)
        y_out = tf.layers.dense(self.outputs[0], 1)

        # Add Loss and Optimizers
        batch_size = tf.shape(Y)[0]
        Y_1 = tf.tile(Y, [1,batch_size])
        Y_2 = tf.tile(tf.transpose(Y, [1,0]), [batch_size,1])
        y_1 = tf.tile(y_out, [1,batch_size])
        y_2 = tf.tile(tf.transpose(y_out, [1,0]), [batch_size,1])
        gdt_ts_diff = Y_1 - Y_2
        score_diff = y_1 - y_2
        Y__ = tf.where(tf.greater(Y_1, Y_2), -tf.ones_like(gdt_ts_diff), tf.ones_like(gdt_ts_diff))
        L_ = 1 - (Y__*score_diff)
        L_ = tf.where(tf.greater(L_, 0), L_, tf.zeros_like(gdt_ts_diff))
        L_ = tf.where(tf.greater(tf.abs(gdt_ts_diff), 0.1), L_, tf.zeros_like(L_))
        loss_train = tf.reduce_sum(tf.reduce_sum(L_,axis=-1,),axis=0) / tf.cast(batch_size*batch_size, tf.float32)
        loss_eval = tf.abs(tf.reduce_max(Y)-tf.reduce_max(tf.gather(Y, tf.argmin(Y,axis=0))))

        if optimizer == 'Adam':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate).minimize(loss_train)
        else: raise Exception('Optimizer %s unknown' % optimizer)

        # Metrics
        self.operators = [opt, loss_train, loss_eval]

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
        Method defines Scoring training epoch.

        Params:
            data_loader - DataLoader; training examples generator
            sess - TF.sess;

        Return:
            summary - dict; training results

        '''
        # Summary
        summary = dict()
        loss = []
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Training...')
        for i, data in enumerate(bg(data_loader)):
            data = [True,] + data
            out = sess.run(self.operators[:2], feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[1])

        # Get metrics
        summary['train_time'] = time.time() - start_time
        summary['train_loss_average'] = np.average(loss)
        summary['train_loss_median'] = np.median(loss)
        summary['train_loss_range'] = np.max(loss) - np.min(loss)

        # Log Epoch results
        self.logger.info('train time: %.3f' % summary['train_time'])
        self.logger.info('train loss average: %.3f' % summary['train_loss_average'])
        self.logger.info('train loss median: %.3f' % summary['train_loss_median'])
        self.logger.info('train loss range: %.3f' % summary['train_loss_range'])

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
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Evaluating...')
        for i, data in enumerate(bg(data_loader)):
            data = [False,] + data
            out = sess.run(self.operators[1:], feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[1])

        # Get Metrics
        summary[mode+'_time'] = time.time() - start_time
        summary[mode+'_loss_average'] = np.average(loss)
        summary[mode+'_loss_median'] = np.median(loss)
        summary[mode+'_loss_range'] = np.max(loss) - np.min(loss)

        # Log epoch results
        self.logger.info(mode+' time: %.3f' % summary[mode+'_time'])
        self.logger.info(mode+' loss average: %.3f' % summary[mode+'_loss_average'])
        self.logger.info(mode+' loss median: %.3f' % summary[mode+'_loss_median'])
        self.logger.info(mode+' loss range: %.3f' % summary[mode+'_loss_range'])

        return summary
