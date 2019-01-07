def get_trainer(name, **trainer_args):
    '''
    Factory function for retrieving a trainer.
    '''
    if 'k_folds' in trainer_args.keys(): trainer_args.pop('k_folds')
    if 'k_fold_valid' in trainer_args.keys(): trainer_args.pop('k_fold_valid')
    if name == 'classifier':
        from .classifier_trainer import ClassifierTrainer
        return ClassifierTrainer(**trainer_args)
    elif name == 'regressor':
        from .regressor_trainer import RegressorTrainer
        return RegressorTrainer(**trainer_args)
    elif name == 'scoring':
        from .scoring_trainer import ScoringTrainer
        return ScoringTrainer(**trainer_args)
    else:
        raise Exception('Trainer %s unknown' % name)
