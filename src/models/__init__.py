def get_model(name, **model_args):
    '''
    Top-level factory function for getting your models.
    '''
    if name == 'gcnn':
        from .gcnn import GCNN
        return GCNN(**model_args)
    elif name == 'cnn_2d':
        from .cnn_2d import CNN2D
        return CNN2D(**model_args)
    elif name == 'cnn_3d':
        from .cnn_3d import CNN3D
        return CNN3D(**model_args)
    else:
        raise Exception('Model %s unknown' % name)

class Model(object):

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.model_name = None

    def get_inputs_outputs(self):
        '''
        '''
        return self.inputs, self.outputs

    def get_model_name(self):
        '''
        '''
        return self.model_name

    def define_model(self):
        '''
        '''
        raise NotImplementedError
