import os
import platform
import wandb

from utils.logger import Logger


def get_local_params():
    if os.path.isdir('/media/giantdrive'):
        dataset_path = '/media/giantdrive/coloradar/dataset_may2_all.h5'
        device_name = 'cuda:1'
        dataset_part = 1.0
        logger = Logger(print_log=True, loggers=(wandb, ))
        batch_size = 8
        n_epochs = 100
        model_save_directory = '/media/giantdrive/coloradar/models'
    elif platform.system() == "Darwin":
        dataset_path = '/Users/anna/data/coloradar/dataset_may2_one.h5'
        device_name = 'mps'
        dataset_part = 0.1
        logger = Logger(print_log=True)
        batch_size = 4
        n_epochs = 10
        model_save_directory = '/Users/anna/data/coloradar/models'
    else:
        dataset_path = '/home/arpg/coloradar/one-mar11.h5'
        device_name = 'cpu'
        dataset_part = 0.1
        logger = Logger(print_log=True)
        batch_size = 2
        n_epochs = 10
        model_save_directory = '/home/arpg/projects/mapless-navigation/trained_models'

    return {
        'dataset_path': dataset_path,
        'device_name': device_name,
        'dataset_part': dataset_part,
        'logger': logger,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'model_save_directory': model_save_directory
    }
