import wandb
import logging

logger = logging.getLogger(__name__)


def execute_sweep(model, dataloader, config, device, log_level):
    sweep_config = {
        'name': model.__class__.__name__ + '_sweep',
        'description': 'Hyperparameter sweep for ' + model.__class__.__name__,
            'method': 'random',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'epochs': {
                'values': 10
            },
            'hidden_size': {
                'values': [64, 128, 256, 512]
            },
            'num_layers': {
                'values': [2, 3, 4]
            },
            'learning_rate': {
                'values': [1e-6, 1e-5, 1e-4, 1e-3]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            }
        }
    }