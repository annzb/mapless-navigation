import os
import platform
import wandb
from typing import Optional

from utils.logger import Logger


def get_params():
    logger = Logger(print_log=True)
    device_name = 'cpu'
    model_save_directory = '.'
    random_seed = 42
    training_params = {
        'n_epochs': 500,
        'checkpoint_interval': 10,
        'batch_size': 8
    }
    dataset_params = {
        'dataset_file_path': None,
        'partial': 1.0, 
        'shuffle_runs': True,
        'intensity_threshold': 5000.0,
        'grid_voxel_size': 0.25
    }
    model_params = {
        'encoder_cloud_size': 1024,
        'encoder_num_features': 128,
        'encoder_batch_norm': True,
        'encoder_dropout': 0.2,
        'predicted_cloud_size': 4096,
        'decoder_layer_norm': True,
        'decoder_dropout': 0.2
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
    metric_params = {}

    if os.path.isdir('/media/giantdrive'):
        host_name = 'brute'
        logger = Logger(print_log=True, loggers=(wandb, ))
        device_name = 'cuda:1'
        model_save_directory = '/home/annz/mapping/models'
        training_params['n_epochs'] = 1000
        training_params['batch_size'] = 8

        dataset_params['dataset_file_path'] = '/media/giantdrive/coloradar/dataset_may2_all.h5'
        dataset_params['partial'] = 0.05
        # model_params['encoder_batch_norm'] = False
        # model_params['decoder_layer_norm'] = False
        # model_params['encoder_dropout'] = None
        # model_params['decoder_dropout'] = None

    elif platform.system() == "Darwin":
        host_name = 'mac'
        logger = Logger(print_log=True)
        device_name = 'mps'
        model_save_directory = '/Users/anna/data/coloradar/models'
        training_params['n_epochs'] = 2
        training_params['batch_size'] = 2

        dataset_params['dataset_file_path'] = '/Users/anna/data/coloradar/dataset_may2_one.h5'
        dataset_params['partial'] = 0.01
        model_params['encoder_batch_norm'] = False
        model_params['decoder_layer_norm'] = False
        model_params['encoder_dropout'] = None
        model_params['decoder_dropout'] = None
    
    else:
        host_name = 'lab_pc'
        logger = Logger(print_log=True)
        model_save_directory = '/home/arpg/projects/mapless-navigation/trained_models'
        training_params['n_epochs'] = 10
        training_params['batch_size'] = 4

        dataset_params['dataset_file_path'] = '/home/arpg/coloradar/dataset_may2_one.h5'
        dataset_params['partial'] = 0.01
        # model_params['encoder_batch_norm'] = False
        # model_params['decoder_layer_norm'] = False
        # model_params['encoder_dropout'] = None
        # model_params['decoder_dropout'] = None

    return {
        'host_name': host_name,
        'logger': logger,
        'device_name': device_name,
        'model_save_directory': model_save_directory,
        'random_seed': random_seed,
        'training_params': training_params,
        'model_params': model_params,
        'dataset_params': dataset_params,
        'optimizer_params': optimizer_params,
        'loss_params': loss_params,
        'metric_params': metric_params
    }
