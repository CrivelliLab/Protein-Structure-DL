'''
main.py

README:

Script runs training experiments defined by .yaml files located in src/configs.

'''
import os
import argparse
import logging
import yaml
import numpy as np
from datasets import get_datasets, DataLoader
from trainers import get_trainer

################################################################################

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('main.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/hello.yaml')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--cores', nargs='?', default=1, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f: config = yaml.load(f)
    data_config = config['data_config']
    model_config = config.get('model_config', {})
    train_config = config['train_config']
    experiment_config = config['experiment_config']
    output_dir = experiment_config.pop('output_dir', None)

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(),
                        logging.FileHandler(output_dir+'/train.log')])
    logging.info('Initializing')
    if args.show_config: logging.info('Command line config: %s' % args)

    # Load the datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(**data_config)
    batch_size = train_config.pop('batch_size')

    if train_dataset.__len__() > 0:
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, cores=args.cores)
        logging.info('Loaded %g training samples', len(train_dataset))
    else: train_data_loader = None
    if valid_dataset.__len__() > 0:
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, cores=args.cores)
        logging.info('Loaded %g validation samples', len(valid_dataset))
    else: valid_data_loader = None
    if test_dataset.__len__() > 0:
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, cores=args.cores)
        logging.info('Loaded %g test samples', len(test_dataset))
    else: test_data_loader = None

    # Load the trainer
    trainer = get_trainer(output_dir=output_dir, **experiment_config)

    # Build the model
    trainer.build_model(**model_config)
    trainer.print_model_summary()

    # Run the training
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            **train_config)
    trainer.write_summary()

    # Print some conclusions
    logging.info('All done!')
