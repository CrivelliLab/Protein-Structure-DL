'''
protienligand_graph.py

'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ProtienLigandGraphDataset():

    def __init__(self, data, nb_nodes, task_type, nb_classes, site_path, curve_order, diameter):
        '''
        '''
        self.data = data
        self.nb_nodes = nb_nodes
        self.nb_classes = nb_classes
        self.task_type = task_type
        self.site_path = site_path

        # Generate SPC
        self.curve_order = curve_order
        self.diameter = diameter
        self.curve = self.__hilbert_3d(curve_order)

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
        self.site_com = c.mean(axis=0)
        c = c - self.site_com

        # Spatial Ordering Using SPC
        data = np.concatenate([v,c],axis=-1)
        sorted_data = self.__hilbert_sort(data, self.curve, self.diameter, 2**self.curve_order)
        v = sorted_data[:,:-3]
        c = sorted_data[:,-3:]

        self.site = [v,c]
        self.ident = np.eye(nb_nodes)

    def __getitem__(self, index):
        '''
        '''
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

                v.append(a_ + b_)
                c.append(row[2:5])
        v = np.array(v, dtype=float)
        c = np.array(c, dtype=float)
        c = c - self.site_com

        # Spatial Ordering Using SPC
        #data = np.concatenate([v,c],axis=-1)
        #sorted_data = self.__hilbert_sort(data, self.curve, self.diameter, 2**self.curve_order)
        #v = sorted_data[:,:-3]
        #c = sorted_data[:,-3:]

        # Zero Padding
        v_ = np.zeros((self.nb_nodes, v.shape[1]))
        vs = self.site[0]
        v_[:vs.shape[0],:vs.shape[1]] = vs
        v_[-v.shape[0]:,-v.shape[1]:] = v
        v = v_
        c_ = np.zeros((self.nb_nodes, c.shape[1]))
        cs = self.site[1]
        c_[:cs.shape[0],:cs.shape[1]] = cs
        c_[-c.shape[0]:,-c.shape[1]:] = c
        c = c_

        # Set MasK
        m = np.repeat(np.expand_dims(v[:,-3], axis=-1), len(v), axis=-1)
        m = (m + m.T) + self.ident
        m[m>1] = 1
        m = self.ident

        if self.task_type == 'classification':
            y = [0 for _ in range(self.nb_classes)]
            y[int(self.data[index][1])] = 1
        elif self.task_type == 'regression': y = [float(self.data[index][1]),]
        else: raise Exception('Task Type %s unknown' % self.task_type)

        data_ = [v,c,m,y]

        return data_

    def __len__(self):
        '''
        '''
        return len(self.data)

    def __hilbert_3d(self, order):
        '''
        Method generates 3D hilbert curve of desired order.
        Param:
            order - int ; order of curve
        Returns:
            np.array ; list of (x, y, z) coordinates of curve
        '''

        def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
            if order == 0:
                xx = x + (xi + yi + zi)/3
                yy = y + (xj + yj + zj)/3
                zz = z + (xk + yk + zk)/3
                array.append((xx, yy, zz))
            else:
                gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

                gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                           yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                           xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                           -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                           -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                           -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                           -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                           -zk/2, -xi/2, -xj/2, -xk/2, array)

        n = pow(2, order)
        hilbert_curve = []
        gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

        return np.array(hilbert_curve).astype('int')

    def __hilbert_sort(self, data, curve, diameter, bins):
        '''
        '''
        # Bin points
        binned = [[[[] for k in range(bins)] for j in range(bins)] for i in range(bins)]
        bin_interval = (diameter / bins)
        offset = int((diameter/2.0)/bin_interval)
        for i, _ in enumerate(data):
            x = int(_[-3]/bin_interval) + offset
            y = int(_[-2]/bin_interval) + offset
            z = int(_[-1]/bin_interval) + offset
            #print(x,y,z, _)
            if (x > bins-1) or (x < 0): continue
            if (y > bins-1) or (y < 0): continue
            if (z > bins-1) or (z < 0): continue
            binned[x][y][z].append(_)

        # Traverse and Assemble
        sorted_data = []
        for _ in curve:
            x = binned[_[0]][_[1]][_[2]]
            if len(x) > 0:
                sorted_data.append(np.array(x))

        sorted_data = np.concatenate(sorted_data, axis=0)

        return sorted_data

def get_datasets(data_path, nb_nodes, task_type, nb_classes, curve_order=5, diameter=50.0, split=[0.7,0.1,0.2], k_fold=None, seed=1234):
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
    site_path = data_path + '/site.txt'
    train_dataset = ProtienLigandGraphDataset(data_train[:300], nb_nodes, task_type, nb_classes, site_path, curve_order, diameter)
    valid_dataset = ProtienLigandGraphDataset(data_valid, nb_nodes, task_type, nb_classes, site_path, curve_order, diameter)
    test_dataset = ProtienLigandGraphDataset(data_test, nb_nodes, task_type, nb_classes, site_path, curve_order, diameter)

    return train_dataset, valid_dataset, test_dataset
