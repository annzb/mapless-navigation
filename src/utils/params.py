import os
import platform
import wandb

from utils.logger import Logger


def get_params():
    logger = Logger(print_log=True)
    device_name = 'cpu'
    model_save_directory = '.'
    random_seed = 42
    n_epochs = 1
    batch_size = 1
    dataset_params = {
        'dataset_file_path': None,
        'partial': 1.0, 
        'shuffle_runs': True,
        'intensity_threshold': 0.0,
        'grid_voxel_size': 1.0
    }
    optimizer_params = {
        'learning_rate': 1e-3
    }
    loss_params = {
        'occupied_only': True,
        'occupancy_threshold': 0.6,
        'max_point_distance': 1.0,
        'unmatched_weight': 1.0,
        'fn_fp_weight': 1.0,
        'fn_weight': 1.0,
        'fp_weight': 1.0,
        'spatial_weight': 1.0,
        'occupancy_weight': 1.0
    }
    metric_params = {

    }
        

    if os.path.isdir('/media/giantdrive'):
        logger = Logger(print_log=True, loggers=(wandb, ))
        device_name = 'cuda:1'
        model_save_directory = '/home/annz/mapping/models'
        n_epochs = 100
        batch_size = 8

        dataset_params['dataset_file_path'] = '/media/giantdrive/coloradar/dataset_may2_all.h5'
        dataset_params['partial'] = 0.4

    elif platform.system() == "Darwin":
        device_name = 'mps'
        model_save_directory = '/Users/anna/data/coloradar/models'
        n_epochs = 3
        batch_size = 4

        dataset_params['dataset_file_path'] = '/Users/anna/data/coloradar/dataset_may2_one.h5'
        dataset_params['partial'] = 0.1
    
    else:
        model_save_directory = '/home/arpg/projects/mapless-navigation/trained_models'
        n_epochs = 5
        batch_size = 4

        dataset_params['dataset_file_path'] = '/home/arpg/coloradar/one-mar11.h5'
        dataset_params['partial'] = 0.1

    return {
        'logger': logger,
        'device_name': device_name,
        'model_save_directory': model_save_directory,
        'random_seed': random_seed,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'dataset_params': dataset_params,
        'optimizer_params': optimizer_params,
        'loss_params': loss_params,
        'metric_params': metric_params
    }
