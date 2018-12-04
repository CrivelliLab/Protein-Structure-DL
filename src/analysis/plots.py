'''
plots.py

'''
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('plots.py')
    add_arg = parser.add_argument
    add_arg('outfolder', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--acc', action='store_true')
    add_arg('--loss', action='store_true')
    add_arg('--prec', action='store_true')
    add_arg('--recall', action='store_true')
    add_arg('--f1', action='store_true')
    add_arg('--auc', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    out_folder = args.outfolder
    plot_loss = args.loss
    plot_acc = args.acc
    plot_prec = args.prec
    plot_recall = args.recall
    plot_f1 = args.f1
    plot_auc = args.auc
    verbose = args.verbose
    if out_folder[-1] != '/': out_folder += '/'
    plot_path = out_folder+'plots/'
    if not os.path.exists(plot_path): os.mkdir(plot_path)
    title = ' '.join(out_folder.split('/')[-2].split('_')).upper()

    # Load training history
    history_df = pd.read_csv(out_folder+'history.csv')[:-1].astype('float')

    # Plot loss
    if plot_loss:
        f = plt.figure()
        labels = history_df.filter(like='loss').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"loss.pdf", bbox_inches='tight')
        f.savefig(plot_path+"loss.png", bbox_inches='tight')

    # Plot accuracy
    if plot_acc:
        f = plt.figure()
        labels = history_df.filter(like='acc').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 1.1)
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"accuracy.pdf", bbox_inches='tight')
        f.savefig(plot_path+"accuracy.png", bbox_inches='tight')

    # Plot precision
    if plot_prec:
        f = plt.figure()
        labels = history_df.filter(like='prec').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.ylim(0.5, 1.1)
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"precision.pdf", bbox_inches='tight')
        f.savefig(plot_path+"precision.png", bbox_inches='tight')

    # Plot recall
    if plot_recall:
        f = plt.figure()
        labels = history_df.filter(like='recall').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.ylim(0.5, 1.1)
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"recall.pdf", bbox_inches='tight')
        f.savefig(plot_path+"recall.png", bbox_inches='tight')

    # Plot f1
    if plot_f1:
        f = plt.figure()
        labels = history_df.filter(like='f1').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.ylim(0.5, 1.1)
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"f1.pdf", bbox_inches='tight')
        f.savefig(plot_path+"f1.png", bbox_inches='tight')

    # Plot auc
    if plot_auc:
        f = plt.figure()
        labels = history_df.filter(like='auc').columns
        train = history_df[labels[0]].values
        val = history_df[labels[1]].values
        plt.plot(list(range(len(train))),train, label='Training')
        plt.plot(list(range(len(val))),val, label='Validation')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.ylim(0.5, 1.1)
        plt.legend(loc='upper right')
        if verbose: plt.show()
        f.savefig(plot_path+"auc.pdf", bbox_inches='tight')
        f.savefig(plot_path+"auc.png", bbox_inches='tight')
