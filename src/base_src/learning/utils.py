import os
import wandb

from model_manager import Logger


def get_local_params():
    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset2.h5'
        device_name = 'cuda:1'
        dataset_part = 1.0
        logger = Logger(print_log=True, loggers=(wandb, ))
        batch_size = 8
    else:
        dataset_path = '/home/arpg/projects/coloradar_plus_processing_tools/dataset2.h5'
        device_name = 'cpu'
        dataset_part = 0.05
        logger = Logger(print_log=True)
        batch_size = 2
    return {
        'dataset_path': dataset_path,
        'device_name': device_name,
        'dataset_part': dataset_part,
        'logger': logger,
        'batch_size': batch_size
    }
