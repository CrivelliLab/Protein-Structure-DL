'''
base_trainer.py

README:

Script defines base trainer class for Tensorflow neural network training.
BaseTrainer class is an abstract class that requires three methods to be defined
for different training experiments:

    - build_model()
    - train_epoch()
    - evaluate()

'''
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing

def bg(gen):
    '''
    Method is used to run batch retrieval as a parallel process.
    This allows the next bathc to be compiled during tensorflow network
    updates.

    '''
    def _bg_gen(gen,conn):
        for data in gen:
            if conn.recv(): conn.send(data)
        conn.send(StopIteration)

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_bg_gen, args=(gen,child_conn))
    p.start()

    parent_conn.send(True)
    while True:
        parent_conn.send(True)
        x = parent_conn.recv()
        if x is StopIteration: return
        else: yield x

class BaseTrainer(object):

    def __init__(self, output_dir=None):
        '''
        Params:
            output_dir - str; path to output directory.

        '''
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir) if output_dir is not None else None)
        self.summary = {}
        self.running_vars_initializer = None
        self.model = None
        self.nb_parameters = None
        self.saver = None
        self.final_score = None
        tf.random.set_random_seed(123456)

    def print_model_summary(self):
        '''
        '''
        self.logger.info('Model: %s' % self.model.get_model_name())
        self.logger.info('Parameters: %i' % self.nb_parameters)

    def save_summary(self, summary):
        '''
        Params:
            summary - dict; training summary dictionary.

        '''
        for (key, val) in summary.items():
            summary_vals = self.summary.get(key, [])
            self.summary[key] = summary_vals + [val]

    def write_summary(self, kfold_i=None):
        '''
        '''
        # Gather History
        history = []
        test = 'test, '
        header = ''
        for _ in self.summary:
            if 'test' in _: test += str(self.summary[_][0]) + ', '
            else:
                header += str(_) + ', '
                history.append(np.expand_dims(self.summary[_],axis=-1))
        history = np.concatenate(history, axis=-1)

        # Summary
        df = pd.DataFrame(history[:,1:])
        df.columns = header.split(',')[1:-1]
        summary = df.describe()

        # Set Paths
        assert self.output_dir is not None
        if kfold_i is not None:
            history_file = os.path.join(self.output_dir, 'history_k'+str(kfold_i)+'.csv')
            summary_file = os.path.join(self.output_dir, 'summary_k'+str(kfold_i)+'.csv')
        else:
            history_file = os.path.join(self.output_dir, 'history.csv')
            summary_file = os.path.join(self.output_dir, 'summary.csv')
        self.logger.info('Saving History to %s' % history_file)
        self.logger.info('Saving Summary to %s' % summary_file)

        # Save Training History and Summary
        np.savetxt(history_file, history, fmt= '%1.6f', delimiter=', ',header=header,footer=test)
        summary.to_csv(summary_file, float_format='%1.6f', sep=',')

    def get_best_score(self,):
        '''
        '''
        for _ in self.summary:
            if 'test' in _: test += str(self.summary[_][0]) + ', '

    def build_model(self):
        '''
        Abstract method defining model for training. Must be defined for each trainer.

        '''
        raise NotImplementedError

    def train_epoch(self, data_loader):
        '''
        Abstract method defining training epoch of trainer. Must be defined for
        each trainer.

        '''
        raise NotImplementedError

    def evaluate(self, data_loader):
        '''
        Abstract method defining evaluation epoc of trainer. Must be defined for
        each trainer.

        '''
        raise NotImplementedError

    def train(self, train_data_loader, valid_data_loader, test_data_loader,
                    nb_epochs=10, early_stop_metric='val_loss',
                    early_stop_epochs=0, load_prev=False, save_best=False):
        '''
        Method defines training loop. If save_best, saves best model according to
        the lowest loss during training.

        Params:
            train_data_loader - DataLoader; training examples generator
            valid_data_loader - DataLoader; validation examples generator
            test_data_loader - DataLoader;  test examples generator
            nb_epochs - int; number of training epochs
            save_best - bool; whether to save best TF model according to loss.

        '''
        # Start the session
        self.saver = tf.train.Saver()
        with tf.Session() as sess:

            # Setup the variable initialisation
            sess.run(tf.global_variables_initializer())

            if load_prev:
                if os.path.exists(self.output_dir+"/model/model.ckpt"):
                    self.saver.restore(sess, self.output_dir+"/model/model.ckpt")

            # Loop over epochs
            best_metric = None
            epochs_since_best = 0
            for i in range(nb_epochs):
                # Log Epoch
                self.logger.info('Epoch %i:' % i)
                summary = dict(epoch=i)

                # Train on this epoch
                if train_data_loader is not None:
                    sess.run(self.running_vars_initializer)
                    summary.update(self.train_epoch(train_data_loader, sess))

                # Evaluate on this epoch
                if valid_data_loader is not None:
                    sess.run(self.running_vars_initializer)
                    summary.update(self.evaluate(valid_data_loader, sess, mode='valid'))

                metric = summary[early_stop_metric]

                # Save summary, checkpoint
                self.save_summary(summary)
                if not os.path.exists(self.output_dir+'/model'): os.mkdir(self.output_dir+'/model')
                if save_best:
                    if 'loss' in early_stop_metric:
                        if best_metric == None or metric <= best_metric:
                            self.logger.info('Current best according to '+early_stop_metric+'. Model Saved.')
                            best_metric = metric
                            epochs_since_best = 0
                            self.saver.save(sess, self.output_dir+"/model/model.ckpt")
                        else: epochs_since_best +=1
                    else:
                        if best_metric == None or metric >= best_metric:
                            self.logger.info('Current best according to '+early_stop_metric+'. Model Saved.')
                            best_metric = metric
                            epochs_since_best = 0
                            self.saver.save(sess, self.output_dir+"/model/model.ckpt")
                        else: epochs_since_best +=1

                # Stop training if no progress have been made
                if early_stop_epochs > 0:
                    if epochs_since_best == early_stop_epochs:
                        self.logger.info('Stopping training early due to no progress within the last '+ str(early_stop_epochs)+' epochs.')
                        break

            # Load best model
            if save_best:
                self.saver.restore(sess, self.output_dir+"/model/model.ckpt")

            # Evaluate on this epoch
            summary = dict()
            if test_data_loader is not None:
                sess.run(self.running_vars_initializer)
                summary.update(self.evaluate(test_data_loader, sess, mode='test'))
            self.final_score = summary['test_loss']

            # Save summary, checkpoint
            self.save_summary(summary)

        return self.summary
