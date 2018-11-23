import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ProtienPairwiseDataset():

    def __init__(self, data, task_type, nb_classes):
        '''
        '''
        self.data = data
        self.nb_classes = nb_classes
        self.task_type = task_type

    def __getitem__(self, index):
        '''
        '''
        # Load Image
        x = np.load(self.data[index][0])
        x = x[x.keys()[0]]

        if self.task_type == 'classification':
            y = [0 for _ in range(self.nb_classes)]
            y[int(self.data[index][1])] = 1
        elif self.task_type == 'regression': y = [float(self.data[index][1]),]
        else: raise Exception('Task Type %s unknown' % self.task_type)

        data_ = [x,y]

        return data_

    def __len__(self):
        '''
        '''
        return len(self.data)

def get_datasets(data_path, task_type, nb_classes, split=[0.7,0.1,0.2], seed=1234):
    '''
    '''
    # Load examples
    X = []
    Y = []
    with open(data_path+'/data.csv', 'r')as f:
        for i, _ in enumerate(f):
            row = _[:-1].split(',')
            pdb_id = row[0].lower()
            if not os.path.exists(data_path+'/pairwise2d/'+pdb_id+'.npz'): continue
            X.append(data_path+'/pairwise2d/'+pdb_id+'.npz')
            Y.append(row[2])
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    # Split Examples
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split[2], random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=split[1]/(split[0]+split[1]), random_state=seed)
    data_train = np.concatenate([x_train,y_train],axis=-1)
    data_test = np.concatenate([x_test,y_test],axis=-1)
    data_valid = np.concatenate([x_valid,y_valid],axis=-1)

    # Initialize Dataset Iterators
    train_dataset = ProtienPairwiseDataset(data_train, task_type, nb_classes)
    valid_dataset = ProtienPairwiseDataset(data_valid, task_type, nb_classes)
    test_dataset = ProtienPairwiseDataset(data_test, task_type, nb_classes)

    return train_dataset, valid_dataset, test_dataset
