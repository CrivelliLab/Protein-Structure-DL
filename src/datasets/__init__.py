import numpy as np
from tqdm import tqdm

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

    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        xx = len(self.dataset.__getitem__(0))
        batch_x = [[] for i in range(xx)]
        for i in tqdm(range(self.dataset.__len__())):
            x = self.dataset.__getitem__(i)
            for ii in range(len(x)): batch_x[ii].append(x[ii])
            if len(batch_x[0]) == self.batch_size or i+1 == self.dataset.__len__():
                batch_x = [np.array(_) for _ in batch_x]
                yield batch_x
                batch_x = [[] for i in range(xx)]
