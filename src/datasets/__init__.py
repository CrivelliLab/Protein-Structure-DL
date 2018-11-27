import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def get_datasets(name, **data_args):
    if name == 'protien_graphs':
        from .protien_graph import get_datasets
        return get_datasets(**data_args)
    elif name == 'protienligand_graphs':
        pass
    elif name == 'protien_volume_images':
        from .protien_volume3d import get_datasets
        return get_datasets(**data_args)
    elif name == 'protien_pairwise_images':
        from .protien_pairwise2d import get_datasets
        return get_datasets(**data_args)
    else:
        raise Exception('Dataset %s unknown' % name)

class DataLoader(object):

    def __init__(self, dataset, batch_size=1, cores=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cores = cores

    def __iter__(self):
        if self.cores > 1: self.pool = Pool(self.cores)
        xx = len(self.dataset.__getitem__(0))
        batch_x = [[] for i in range(xx)]
        fetch_i = []
        for i in tqdm(range(self.dataset.__len__())):
            fetch_i.append(i)
            if self.cores > 1:
                if len(fetch_i) == self.cores:
                    data = self.pool.map(self.gather_data, fetch_i)
                    fetch_i = []
                elif i+1 == self.dataset.__len__():
                    data = [self.gather_data(fi) for fi in fetch_i]
                    fetch_i = []
                else: continue
            else:
                data = [self.gather_data(fetch_i[0]),]
                fetch_i = []
            for x in data:
                for ii in range(len(x)):batch_x[ii].append(x[ii])
                if len(batch_x[0]) == self.batch_size or i+1 == self.dataset.__len__():
                    batch = [np.array(_) for _ in batch_x]
                    yield batch
                    batch_x = [[] for i in range(xx)]
        if self.cores > 1: self.pool.close()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def gather_data(self, i):
        return self.dataset.__getitem__(i)
