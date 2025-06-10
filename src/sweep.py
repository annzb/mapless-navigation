import wandb
from train_points_generative import run


def run_wrapper():
    wandb.init()
    sweep_params = dict(wandb.config)
    grouped_params = {
        'training_params': {},
        'model_params': {},
        'dataset_params': {},
        'optimizer_params': {},
        'loss_params': {}
    }
    for flat_key, value in sweep_params.items():
        if '.' in flat_key:
            group, key = flat_key.split('.', 1)
            if group in grouped_params:
                grouped_params[group][key] = value
    run(
        training_params=grouped_params['training_params'],
        model_params=grouped_params['model_params'],
        dataset_params=grouped_params['dataset_params'],
        optimizer_params=grouped_params['optimizer_params'],
        loss_params=grouped_params['loss_params'],
    )


def sweep():
    config = {
        'project': 'radar-occupancy',
        'name': 'small-sample-param-search',
        'method': 'bayes',
        'metric': {'name': 'val_metric_dbscanf1', 'goal': 'maximize'},
        'parameters': {
            'training_params.n_epochs': {
                'values': [50, 100, 150, 200, 250, 300]
            },

            'optimizer_params.learning_rate': {
                'min': 1e-6,
                'max': 1e-2,
                'distribution': 'log_uniform_values'
            },

            'loss_params.occupancy_threshold': {
                'values': [0.5, 0.6, 0.7, 0.8]
            },
            'loss_params.max_point_distance': {
                'min': 0.1,
                'max': 5.0,
                'distribution': 'uniform'
            },
            'loss_params.fn_fp_weight': {
                'min': 0.1,
                'max': 10.0,
                'distribution': 'log_uniform_values'
            },
            'loss_params.fn_weight': {
                'min': 0.1,
                'max': 10.0,
                'distribution': 'log_uniform_values'
            },
            'loss_params.fp_weight': {
                'min': 0.1,
                'max': 10.0,
                'distribution': 'log_uniform_values'
            },

            'model_params.encoder_cloud_size': {
                'values': [512, 1024, 2048, 4096]
            },
            'model_params.encoder_num_features': {
                'values': [64, 128, 256, 512]
            },
            'model_params.encoder_batch_norm': {
                'values': [True, False]
            },
            'model_params.encoder_dropout': {
                'min': 0.0,
                'max': 0.5,
                'distribution': 'uniform'
            },
            'model_params.predicted_cloud_size': {
                'values': [2048, 4096, 8192]
            },
            'model_params.decoder_dropout': {
                'min': 0.0,
                'max': 0.5,
                'distribution': 'uniform'
            },
            'model_params.decoder_layer_norm': {
                'values': [True, False]
            }
        }
    }
    sweep_id = wandb.sweep(config)
    wandb.agent(sweep_id, function=run_wrapper)


if __name__ == '__main__':
    sweep()
