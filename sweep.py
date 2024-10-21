from enum import Enum

import torch
import wandb
import logging

from tqdm import tqdm

from data import metrics

logger = logging.getLogger(__name__)

class ModelEnum(Enum):
    SIREN = 'siren'
    MFN = 'mfn'
    FFB = 'fourier'
    KAN = 'kan'
    BASIC = 'basic'


def get_optimizer(model, wandb_config):
    if wandb_config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb_config.learning_rate)
    elif wandb_config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb_config.learning_rate)
    else:
        logger.error(f"Optimizer {wandb_config.optimizer} not recognized")
        raise ValueError(f"Optimizer {wandb_config.optimizer} not recognized")
    return optimizer


class Sweeper:

    def __init__(self, model_name, dataloader, config, device, log_level, sweep_runs = 25):
        self.model_name = model_name
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.log_level = log_level

        logger.setLevel(log_level)

        self.sweep_config = {
            'name': model_name + '_' + self.config["datatype"] + '_sweep',
            'description': 'Hyperparameter sweep for ' + model_name + ' with datatype ' + self.config["datatype"],
                'method': 'random',
            'metric': {
                'name': 'avg_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'epochs': {
                    'values': [1000, 1500, 2000]
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

        logger.debug(f"Creating sweep with config {self.sweep_config}")

        self.sweep_id = wandb.sweep(self.sweep_config, project=model_name + '_' + self.config["datatype"] + '_sweep')

        logger.info(f"Created sweep with id {self.sweep_id}")

        wandb.agent(self.sweep_id, function=self.train, count=sweep_runs)


    def get_model(self, wandb_config):
        match self.model_name:
            case ModelEnum.MFN.value:
                from models.mfn import GaborNet
                model = GaborNet(in_size=self.config["in_features"],
                                 hidden_size=self.config["hidden_dim"],
                                 out_size=self.dataloader.dataset.dataset.channels,
                                 n_layers=wandb_config.num_layers,
                                 input_scale=wandb_config.hidden_size,
                                 weight_scale=1)
            case ModelEnum.FFB.value:
                from models.NFFB.img.NFFB_2d import NFFB
                model = NFFB(self.config["in_features"], self.dataloader.dataset.dataset.channels)
            case ModelEnum.KAN.value:
                from models.kan import KAN
                model = KAN(layers_hidden=[self.config["in_features"], *[wandb_config.hidden_size] * (wandb_config.num_layers + 2),
                                           self.dataloader.dataset.dataset.channels])
            case ModelEnum.SIREN.value:
                from models.siren import SIREN
                model = SIREN(in_features=self.config["in_features"],
                              out_features=self.dataloader.dataset.dataset.channels,
                              hidden_features=wandb.config.hidden_size,
                              num_hidden_layers=wandb.config.num_layers)
            case _:
                logger.error(f"Model {self.model_name} not recognized")
                raise ValueError(f"Model {self.model_name} not recognized")
        return model

    def train(self, config=None):
        with wandb.init(config=config):
            # Set up logging, checkpoints and summaries
            logger.setLevel(self.log_level)
            logger.info(f"Training {self.model_name}")

            wandb_config = wandb.config

            model = self.get_model(wandb_config).to(self.device)
            optimizer = get_optimizer(model, wandb_config)

            with tqdm(total=len(self.dataloader) * wandb_config.epochs) as pbar:
                train_losses = []
                for epoch in range(wandb_config.epochs):
                    for step, (model_input, ground_truth) in enumerate(self.dataloader):
                        model_input = {key: value.to(self.device) for key, value in model_input.items()}
                        ground_truth = {key: value.to(self.device) for key, value in ground_truth.items()}

                        logging.debug(f"Model input: {model_input}")
                        model_output = model(model_input)
                        losses = self.config["loss_fn"](model_output, ground_truth)

                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            single_loss = loss.mean()
                            wandb.log({loss_name: single_loss, 'epoch': epoch})
                            train_loss += single_loss

                        train_losses.append(train_loss.item())
                        if config["datatype"] != "sdf":
                            wandb.log({'total_loss': train_loss,
                                       "avg_loss": self.config["loss_fn"](model_output, ground_truth)['loss'],
                                       'psnr': metrics.peak_signal_to_noise_ratio(model_output, ground_truth),
                                       'epoch': epoch
                                       })
                        else:
                            chamfer, hausdorff, _, _, _, _ = metrics.chamfer_hausdorff_distance(
                                model_output['model_out'], ground_truth['sdf']
                            )

                            wandb.log(losses + {
                                'epoch': epoch,
                                'iou': metrics.intersection_over_union(model_output, ground_truth),
                                'chamfer': chamfer,
                                'hausdorff': hausdorff
                            })

                        # Backpropagation
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                        pbar.update(1)

            logger.info("Training complete")