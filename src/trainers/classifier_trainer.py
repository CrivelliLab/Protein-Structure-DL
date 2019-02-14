'''
classifier_trainer.py

Script defines trainer for classification tasks. Trainer uses categorical
cross entropy as loss function. The following metrics are defined for trainier:

    - accuracy
    - precision
    - recall
    - auc

'''
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import *
from .base_trainer import BaseTrainer, bg
from models import get_model

class ClassifierTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(ClassifierTrainer, self).__init__(**kwargs)

    def build_model(self, model_type='gcnn', optimizer='Adam', learning_rate=0.001, nb_classes=2, **model_args):
        '''
        Method intanstiates tensorflow model and metrics for classification tasks.

        Params:
            model_type - str;
            optimizer - str;
            learning_rate - float;
            nb_classes - int;
            model_args - arguments for model

        '''
        # Retrieve Model
        self.nb_classes = nb_classes
        self.model = get_model(name=model_type, **model_args)
        self.inputs, self.outputs = self.model.get_inputs_outputs()

        # Add ouput classification layer
        Y = tf.placeholder(tf.float32, [None, nb_classes])
        self.inputs.append(Y)
        y_out = tf.layers.dense(self.outputs[0], nb_classes, name='presoftmax_out')
        self.outputs.append(y_out)

        # Add Loss and Optimizers
        loss = tf.losses.softmax_cross_entropy(self.inputs[-1], y_out)
        if optimizer == 'Adam':
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else: raise Exception('Optimizer %s unknown' % optimizer)

        # Metrics
        out = tf.nn.softmax(y_out, axis=-1)
        acc,  acc_update = tf.metrics.accuracy(tf.argmax(self.inputs[-1], 1), tf.argmax(out, 1), name="accuracy")
        prec,  prec_update = tf.metrics.precision(tf.argmax(self.inputs[-1], 1), tf.argmax(out, 1), name="precision")
        recall,  recall_update = tf.metrics.recall(tf.argmax(self.inputs[-1], 1), tf.argmax(out, 1), name="recall")
        f1 = (2 * (prec * recall)) / (prec + recall)
        auc,  auc_update = tf.metrics.auc(tf.argmax(self.inputs[-1], 1), tf.argmax(out, 1), name="auc")
        self.metrics = [acc, prec, recall, f1, auc]
        self.operators = [opt, loss, out, acc_update, prec_update, recall_update, auc_update]
        if nb_classes > 2:
            self.metrics = self.metrics[:-1]
            self.operators = self.operators[:-1]

        # Metric Vars
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy") + \
                        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision") + \
                        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall") + \
                        tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="auc")
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
        Method defines classifier training epoch.

        Params:
            data_loader - DataLoader; training examples generator
            sess - TF.sess;

        Return:
            summary - dict; training results

        '''
        # Summary
        summary = dict()
        loss = []
        if self.nb_classes > 2:
            y_true = []
            y_pred = []
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Training...')
        for i, data in enumerate(bg(data_loader)):
            data = [True,] + data
            out = sess.run(self.operators, feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[1])
            if self.nb_classes > 2:
                y_true.append(np.argmax(data[-1],axis=-1))
                y_pred.append(np.argmax(out[2],axis=-1))

        # Get metrics
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = np.average(loss)
        if self.nb_classes < 3:
            metrics_ = sess.run(self.metrics)
            summary['train_acc'] = metrics_[0]
            summary['train_prec'] = metrics_[1]
            summary['train_recall'] = metrics_[2]
            summary['train_f1'] = metrics_[3]
            summary['train_auc'] = metrics_[4]
        else:
            # Sklearn Metrics
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            print(y_true)
            print(y_pred)
            summary['train_acc'] = accuracy_score(y_true, y_pred)
            summary['train_prec'] = precision_score(y_true, y_pred, average='macro')
            summary['train_recall'] = recall_score(y_true, y_pred, average='macro')
            summary['train_f1'] = f1_score(y_true, y_pred, average='macro')
            summary['train_auc'] = roc_auc_score(y_true, y_pred, average="macro")

        # Log Epoch results
        self.logger.info('train time: %.3f' % summary['train_time'])
        self.logger.info('train loss: %.3f' % summary['train_loss'])
        self.logger.info('train accuracy: %.3f' % summary['train_acc'])
        self.logger.info('train precision: %.3f' % summary['train_prec'])
        self.logger.info('train recall: %.3f' % summary['train_recall'])
        self.logger.info('train F1: %.3f' % summary['train_f1'])
        if self.nb_classes < 3: self.logger.info('train AUC: %.3f' % summary['train_auc'])

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
        if self.nb_classes > 2:
            y_true = []
            y_pred = []
        start_time = time.time()

        # Loop over training batches
        self.logger.info('Evaluating...')
        for i, data in enumerate(bg(data_loader)):
            data = [False,] + data
            out = sess.run(self.operators[1:], feed_dict={i: d for i, d in zip(self.inputs, data)})
            loss.append(out[0])
            if self.nb_classes > 2:
                y_true.append(np.argmax(data[-1],axis=-1))
                y_pred.append(np.argmax(out[1],axis=-1))

        # Get Metrics
        metrics_ = sess.run(self.metrics)
        summary[mode+'_time'] = time.time() - start_time
        summary[mode+'_loss'] = np.average(loss)
        if self.nb_classes < 3:
            summary[mode+'_acc'] = metrics_[0]
            summary[mode+'_prec'] = metrics_[1]
            summary[mode+'_recall'] = metrics_[2]
            summary[mode+'_f1'] = metrics_[3]
            summary[mode+'_auc'] = metrics_[4]
        else:
            # Sklearn Metrics
            print(y_true)
            print(y_pred)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            summary[mode+'_acc'] = accuracy_score(y_true, y_pred)
            summary[mode+'_prec'] = precision_score(y_true, y_pred, average='macro')
            summary[mode+'_recall'] = recall_score(y_true, y_pred, average='macro')
            summary[mode+'_f1'] = f1_score(y_true, y_pred, average='macro')
            summary[mode+'_auc'] = roc_auc_score(y_true, y_pred, average="macro")

        # Log epoch results
        self.logger.info(mode+' time: %.3f' % summary[mode+'_time'])
        self.logger.info(mode+' loss: %.3f' % summary[mode+'_loss'])
        self.logger.info(mode+' accuracy: %.3f' % summary[mode+'_acc'])
        self.logger.info(mode+' precision: %.3f' % summary[mode+'_prec'])
        self.logger.info(mode+' recall: %.3f' % summary[mode+'_recall'])
        self.logger.info(mode+' F1: %.3f' % summary[mode+'_f1'])
        if self.nb_classes < 3:  self.logger.info(mode+' AUC: %.3f' % summary[mode+'_auc'])

        return summary
