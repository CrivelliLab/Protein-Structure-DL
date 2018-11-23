def get_trainer(name, **trainer_args):
    '''
    Factory function for retrieving a trainer.
    '''
    if name == 'classifier':
        from .classifier_trainer import ClassifierTrainer
        return ClassifierTrainer(**trainer_args)
    elif name == 'regressor':
        from .regressor_trainer import RegressorTrainer
        return RegressorTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)
