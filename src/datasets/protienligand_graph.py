import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ProtienLigandGraphDataset():

    def __init__(self, data, nb_nodes, task_type, nb_classes, site_path):
        '''
        '''
        self.data = data
        self.nb_nodes = nb_nodes
        self.nb_classes = nb_classes
        self.task_type = task_type
        self.site_path = site_path


    def __getitem__(self, index):
        '''
        '''
        # Parse Site
        atoms = ['C','H','O','N','S','UNK']
        body = ['L','R','UNK']
        v = []
        c = []
        with open(self.site_path, 'r')as f:
            for i, _ in enumerate(f):
                row = _[:-1].split()
                a = row[0]
                b = row[1]

                if a not in atoms: a = 'UNK'
                a_ = [0 for _ in range(len(atoms))]
                a_[atoms.index(a)] = 1

                if b not in body: b = 'UNK'
                b_ = [0 for _ in range(len(body))]
                b_[body.index(b)] = 1

                v.append(a_+b_)
                c.append(row[2:5])
        v = np.array(v, dtype=float)
        c = np.array(c, dtype=float)
        site = [v,c]

        # Parse Protein Graph
        v = []
        c = []
        atoms = ['C','H','O','N','S','UNK']
        body = ['L','R','UNK']
        with open(self.data[index][0], 'r')as f:
            for i, _ in enumerate(f):
                row = _[:-1].split()
                a = row[0]
                b = row[1]

                if a not in atoms: a = 'UNK'
                a_ = [0 for _ in range(len(atoms))]
                a_[atoms.index(a)] = 1

                if b not in body: b = 'UNK'
                b_ = [0 for _ in range(len(body))]
                b_[body.index(b)] = 1

                v.append(a_+b_)
                c.append(row[2:5])
        v = np.array(v, dtype=float)
        c = np.array(c, dtype=float)
        v = np.concatenate([site[0], v], axis=0)
        c = np.concatenate([site[1], c], axis=0)

        # Zero Padding
        if v.shape[0] < self.nb_nodes:
            v_ = np.zeros((self.nb_nodes, v.shape[1]))
            v_[:v.shape[0],:v.shape[1]] = v
            c_ = np.zeros((self.nb_nodes, c.shape[1]))
            c_[:c.shape[0],:c.shape[1]] = c
            v = v_
            c = c_

        if self.task_type == 'classification':
            y = [0 for _ in range(self.nb_classes)]
            y[int(self.data[index][1])] = 1
        elif self.task_type == 'regression': y = [float(self.data[index][1]),]
        else: raise Exception('Task Type %s unknown' % self.task_type)

        data_ = [v,c,y]

        return data_

    def __len__(self):
        '''
        '''
        return len(self.data)

def get_datasets(data_path, nb_nodes, task_type, nb_classes, split=[0.7,0.1,0.2], seed=1234):
    '''
    '''
    # Load examples
    X = []
    Y = []
    with open(data_path+'/data.csv', 'r')as f:
        for i, _ in enumerate(f):
            row = _[:-1].split(',')
            file_ = row[0]
            filename = file_ + '.txt'
            if not os.path.exists(data_path+'/ligands/'+filename): continue
            X.append(data_path+'/ligands/'+filename)
            Y.append(row[1])
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    # Split Examples
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split[2], random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=split[1]/(split[0]+split[1]), random_state=seed)
    data_train = np.concatenate([x_train,y_train],axis=-1)
    data_test = np.concatenate([x_test,y_test],axis=-1)
    data_valid = np.concatenate([x_valid,y_valid],axis=-1)

    # Initialize Dataset Iterators
    site_path = data_path + '/site.txt'
    train_dataset = ProtienLigandGraphDataset(data_train, nb_nodes, task_type, nb_classes, site_path)
    valid_dataset = ProtienLigandGraphDataset(data_valid, nb_nodes, task_type, nb_classes, site_path)
    test_dataset = ProtienLigandGraphDataset(data_test, nb_nodes, task_type, nb_classes, site_path)

    return train_dataset, valid_dataset, test_dataset
