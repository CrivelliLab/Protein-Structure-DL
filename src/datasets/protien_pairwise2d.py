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
        try:
            x = np.load(self.data[index][0])
            x = x['arr_0']
        except:
            x = None
            print("Error: Reading File:", self.data[index][0])

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

def get_datasets(data_path, task_type, nb_classes, split=[0.7,0.1,0.2], k_fold=None, seed=1234):
    '''
    '''
    # Load examples
    X = []
    Y = []
    with open(data_path+'/data.csv', 'r')as f:
        for i, _ in enumerate(f):
            row = _[:-1].split(',')
            pdb_id = row[0].lower()
            chain_id = row[1].lower()
            filename = pdb_id + '_' + chain_id + '.npz'
            if not os.path.exists(data_path+'/pairwise2d/'+filename): continue
            X.append(data_path+'/pairwise2d/'+filename)
            Y.append(row[2])
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)

    if k_fold is not None:
        # Split into K Folds and return training, validation and test
        np.random.seed(seed)
        data = np.concatenate([X,Y],axis=-1)
        np.random.shuffle(data)
        fs = len(data)//int(k_fold[0])
        ind = [fs*(i+1) for i in range(len(data)//fs)]
        remainder = len(data)%fs
        for i in range(remainder):
            for j in range(i%len(ind)+1):
                ind[-(j+1)] += 1
        folds = np.split(data.copy(), ind, axis=0)
        data_test = folds.pop(int(k_fold[1]))
        data_train = np.concatenate(folds,axis=0)
        x_train, x_valid, y_train, y_valid = train_test_split(data_train[:,0:1], data_train[:,1:], test_size=float(k_fold[-1]), random_state=seed)
        data_train = np.concatenate([x_train,y_train],axis=-1)
        data_valid = np.concatenate([x_valid,y_valid],axis=-1)
    else:
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
