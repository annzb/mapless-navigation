import wandb
from train_points_generative import run


def run_wrapper():
    wandb.init()
    sweep_params = dict(wandb.config)
    global_params = {}
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
        else:
            global_params[flat_key] = value
    run(
        training_params=grouped_params['training_params'],
        model_params=grouped_params['model_params'],
        dataset_params=grouped_params['dataset_params'],
        optimizer_params=grouped_params['optimizer_params'],
        loss_params=grouped_params['loss_params'],
        **global_params
    )


def sweep():
    config = {
        'project': 'radar-occupancy',
        'name': 'XS-norm-dropout-loss',
        'method': 'grid',
        'metric': {'name': 'best_train_loss', 'goal': 'minimize'},
        'parameters': {
            'dataset_params.partial': {
                'values': [0.02]  # around 10 train samples from the One dataset
            },
            'training_params.n_epochs': {
                'values': [300]
            },
            'optimizer_params.learning_rate': {
                'values': [1e-3]
            },
            'loss_params.occupancy_threshold': {
                'values': [0.6]
            },
            'model_params.encoder_batch_norm': {
                'values': [True, False]
            },
            'model_params.encoder_dropout': {
                'values': [0.2, 0.5]
            },
            'model_params.decoder_dropout': {
                'values': [0.2, 0.5]
            },
            'model_params.decoder_layer_norm': {
                'values': [True, False]
            },
            'random_seed': {
                'values': [1, 2, 3]  # to see if consistent across sample sets
            }
        }
    }
    sweep_id = wandb.sweep(config)
    wandb.agent(sweep_id, function=run_wrapper)


if __name__ == '__main__':
    sweep()
