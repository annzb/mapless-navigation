import os
import wandb

from utils.logger import Logger


def get_local_params():
    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/all-mar11.h5'
        device_name = 'cuda:1'
        dataset_part = 1.0
        logger = Logger(print_log=True, loggers=(wandb, ))
        batch_size = 8
        n_epochs = 50
    else:
        dataset_path = '/home/arpg/coloradar/one-mar11.h5'
        device_name = 'cpu'
        dataset_part = 0.1
        logger = Logger(print_log=True)
        batch_size = 2
        n_epochs = 2
    return {
        'dataset_path': dataset_path,
        'device_name': device_name,
        'dataset_part': dataset_part,
        'logger': logger,
        'batch_size': batch_size,
        'n_epochs': n_epochs
    }
