import sys
import os
from argparse import ArgumentParser
from enum import Enum

import torch
import wandb

from framework.dataloader_manager import get_dataloader
import subprocess
import logging

from framework.test import test
from framework.train import train
from sweep import Sweeper

logger = logging.getLogger(__name__)

class ModelEnum(Enum):
    SIREN = 'siren'
    MFN = 'mfn'
    FFB = 'fourier'
    KAN = 'kan'
    FKAN = 'fkan'

hidden_dimensions = {
    'siren': {
        'images': 256,
        'audio': 512,
        'shapes': 256
    },
    'mfn': {
        'images': 16,
        'shapes': 16
        'audio': 512,
    },
    'fourier': {
        'images': 16,
        'audio': 512,
        'shapes': 16
    },
    'kan': {
        'images': [16, 16],
        'audio': [64,64],
        'shapes': [16, 16]
    },
    'fkan': {
        'images': [16],
        'audio': [64,64],
        'shapes': [16, 16]
    }

}

hidden_layers = {
    'siren': {
        'images': 3,
        'audio': 4,
        'shapes': 3
    },
    'mfn': {
        'images': 3,
        'audio': 3,
        'shapes': 3
    },
    'fourier': {
        'images': 3,
        'audio': 3,
        'shapes': 3
    },
    'kan': {
        'images': len(hidden_dimensions['kan']['images']),
        'audio': len(hidden_dimensions['kan']['audio']),
        'shapes': len(hidden_dimensions['kan']['shapes'])
    },
    'fkan': {
        'images': len(hidden_dimensions['fkan']['images']),
        'audio': len(hidden_dimensions['fkan']['audio']),
        'shapes': len(hidden_dimensions['fkan']['shapes'])
    }
}

grid_size = {
    'kan': {
        'images': 5,
        'audio': 5
    },
    'fkan': {
        'images': 5,
        'audio': 5
    }
}

spline_order = {
    'kan': {
        'images': 2,
        'audio': 3
    },
    'fkan': {
        'images': 2,
        'audio': 3
    }
}

def parse_args():
    parser = ArgumentParser(description='Train and test a neural fields model on a chosen dataset with certain parameters')
    parser.add_argument('--data',
                        type=str,
                        choices=['images', 'audio', 'shapes'],
                        default='images',
                        help='Type of data to train and test on. Default is images')
    parser.add_argument('--data_point',
                        type=int,
                        default=0,
                        help='Choose the index of the data_point to train on.')
    parser.add_argument('--data_fidelity',
                        type=str,
                        choices=['low', 'medium', 'high'],
                        default='low',
                        help='Choose the fidelity of the data point to train on.')
    parser.add_argument('--model',
                        type=str,
                        choices=[model.value for model in ModelEnum],
                        default='siren',
                        help='Type of model to use. Options are for SIREN, Multiplicative Filter Networks, ' +
                             'Fourier Filter Banks, Kolmogorov-Arnold Networks, and a basic coordinate-MLP, ' +
                             'respectively. Default is SIREN')
    parser.add_argument('--sweep',
                        action='store_true',
                        help='Run a hyperparameter sweep. Default is False. Note: This will override ' +
                        'any arguments passed related to sweep parameters')
    parser.add_argument('--sweep_runs',
                        type=int,
                        default=25,
                        help='Number of random runs to perform in the hyperparameter sweep. Default is 25')
    parser.add_argument('--epochs',
                        type=int,
                        default=1001,
                        help='Number of epochs to train for. Default is 1001')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size for training. Default is 1')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for training. Default is 1e-3')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for random number generation. Default is 42')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='PyTorch device to train on. Default is cuda')
    parser.add_argument('--verbose',
                        type=int,
                        default=logging.INFO,
                        choices=[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                        help='Verbosity level for logging. Options are for DEBUG, INFO, WARNING, and ERROR, ' +
                             'respectively. Default is INFO')
    parser.add_argument('--save',
                        action='store_true',
                        help='Save the model and optimizer state_dicts (if applicable) after training. ' +
                             'Default is False')
    parser.add_argument('--load',
                        action='store_true',
                        help='Load the stored model and optimizer state_dicts (if applicable) ' +
                             'before training and skip training. Default is False')
    parser.add_argument('--skip_train',
                        action='store_true',
                        help='Skip training and only evaluate the model. Default is False')
    parser.add_argument('--skip_test',
                        action='store_true',
                        help='Skip testing and only train the model. Default is False')
    parser.add_argument('--wandb_api_key',
                        type=str,
                        default=None,
                        help='Your personal API key for Weights and Biases. Default is None. Alternatively, you can ' +
                             'leave this empty and store the key in a file in the project root called "wandb.login". ' +
                             'This file will be ignored by git. ' +
                             'NOTE: Make sure to keep this key private and secure. Do not share it or upload it to ' +
                             'a public repository.')
    return parser.parse_args()

def validate_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = [req.replace('\n', '') for req in f.readlines()]
        installed = subprocess.check_output(['pip', 'freeze']).decode().replace('\r', '').split('\n')
        pytorch_wheel = None
        missing = False
        for req in requirements:
            if req.startswith("--extra-index-url"):
                pytorch_wheel = req.split('/')[-1].strip()
                continue
            if req not in '\t'.join(installed):
                logger.warning(f"Missing requirement: {req}")
                missing = True
            elif req.startswith('torch') and pytorch_wheel not in '\t'.join(installed):
                logger.warning(f"Missing or mismatched PyTorch wheel: {pytorch_wheel}. " +
                      "Check that your PyTorch installation matches the wheel in requirements.txt")
                missing = True
        if pytorch_wheel is None:
            logger.warning("PyTorch wheel not found in requirements.txt. " +
                  "Make sure you have the correct PyTorch version installed")
            missing = True

    if missing:
        logger.warning("Some requirements are missing. Please run 'pip install -r requirements.txt' to install them")
        exit(1)
    logger.info("All requirements satisfied")

def get_configuration(args):
    match args.data:
        case "images":
            from data.metrics import mean_squared_error
            from data import image_summary
            from functools import partial
            resolution = (500, 500)
            return {
                "datatype": "images",
                "loss_fn": mean_squared_error, 
                "summary_fn": partial(image_summary, resolution),
                "resolution": resolution,
                "in_features": 2,
                "hidden_dim": hidden_dimensions[args.model][args.data],
                "hidden_layers": hidden_layers[args.model][args.data],
                "grid_size": grid_size[args.model][args.data],
                "spline_order": spline_order[args.model][args.data]
            }
        case "audio":
            from data.metrics import mean_squared_error
            from data import audio_summary
            return {
                "datatype": "audio",
                "loss_fn": mean_squared_error,
                "summary_fn": audio_summary,
                "in_features": 1,
                "hidden_dim": hidden_dimensions[args.model][args.data],
                "hidden_layers": hidden_layers[args.model][args.data],
                "grid_size": grid_size[args.model][args.data],
                "spline_order": spline_order[args.model][args.data]
            }
        case "shapes":
            from data.metrics import sdf
            from data import shape_summary
            return {
                "datatype": "shapes",
                "loss_fn": sdf,
                "summary_fn": shape_summary,
                "in_features": 3,
                "hidden_dim": hidden_dimensions[args.model][args.data],
                "hidden_layers": hidden_layers[args.model][args.data],
                "clip_grad": True
            }
        case _:
            logger.error(f"Data {args.data} not recognized")
            raise ValueError(f"Data {args.data} not recognized")

def get_model(args, dataloader, config):
    match args.model:
        case ModelEnum.MFN.value:
            from models.mfn import GaborNet
            model = GaborNet(in_size=config["in_features"],
                             hidden_size=config["hidden_dim"],
                             out_size=dataloader.dataset.output_dimensionality,
                             n_layers=config["hidden_layers"],
                             input_scale=256,
                             weight_scale=1
                             )
        case ModelEnum.FFB.value:
            from models.NFFB.img.NFFB_2d import NFFB
            model = NFFB(config["in_features"], dataloader.dataset.output_dimensionality)
        case ModelEnum.KAN.value:
            from models.kan import KAN
            model = KAN(layers_hidden=[config["in_features"],
                                       *config["hidden_dim"],
                                       dataloader.dataset.dataset.output_dimensionality],
                                       grid_size=config["grid_size"],
                                       spline_order=config["spline_order"])
        case ModelEnum.FKAN.value:
            from models.kan import KAN
            model = KAN(layers_hidden=[config["in_features"],
                                        *config["hidden_dim"],
                                        dataloader.dataset.dataset.output_dimensionality],
                                        grid_size=config["grid_size"],
                                        spline_order=config["spline_order"],
                                        selection='fourier')
        case ModelEnum.SIREN.value:
            from models.siren import SIREN
            model = SIREN(in_features=config["in_features"],
                          out_features=dataloader.dataset.output_dimensionality,
                          hidden_features=config["hidden_dim"],
                          num_hidden_layers=config["hidden_layers"])
            if config["datatype"] == "shapes":
                model.clip_gradients = True 
        case _:
            logger.error(f"Model {args.model} not recognized")
            raise ValueError(f"Model {args.model} not recognized")
    return model

def main():
    args = parse_args()
    logging.basicConfig(
        filename='run.log',
        level=args.verbose,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(args.verbose)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"))
    logger.addHandler(console_handler)
    # validate_requirements()

    wandb_config = {
        "data": args.data,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "verbose": args.verbose,
        "save": args.save,
        "load": args.load
    }

    logger.debug(f"Arguments: {args}")

    dataloader = get_dataloader(args)

    logger.debug(f"Dataloaders: {dataloader}")

    logger.info("Loading model")
    config = get_configuration(args)
    logger.info("Configuration loaded")
    model = get_model(args, dataloader, config)


    if args.load:
        model.load_state_dict(torch.load(f"{args.save_dir}/{args.model}.pt"))
    logger.info("Model loaded")

    if args.sweep and not args.wandb_api_key and not os.path.exists('wandb.login'):
        logger.error("Hyperparameter sweep requested but Weights and Biases API key not provided. " +
              "Please provide an API key with the --wandb_api_key argument or in a file called 'wandb.login'")
        exit(1)
    elif args.sweep:
        logger.info("Running hyperparameter sweep, skipping regular training and testing")
        if args.wandb_api_key is not None:
            wandb.login(key=args.wandb_api_key.strip())
        elif os.path.exists('wandb.login'):
            with open('wandb.login', 'r') as f:
                wandb.login(key=f.read().strip())
        else:
            logger.error("Something went wrong logging in to Weights and Biases")
            exit(1)
        Sweeper(args.model, dataloader, config, args.device, args.verbose, args.sweep_runs)
        exit(0)

    use_wandb = True

    if args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)
    elif os.path.exists('wandb.login'):
        with open('wandb.login', 'r') as f:
            wandb.login(key=f.read())
    else:
        logger.warning("Weights and Biases API key not provided. Logging to Weights and Biases will be disabled")
        use_wandb = False

    if use_wandb:
        wandb.init(project=args.model + "_" + args.data, config=wandb_config)
        logger.info("Weights and Biases initialized")
    elif not os.path.exists('out'):
        os.makedirs('out')

    if not args.skip_train:
        train(
            model=model,
            dataloader=dataloader,
            epochs=args.epochs,
            lr=args.lr,
            config=config,
            device=args.device,
            log_level=args.verbose,
            use_wandb=use_wandb
        )

    if not args.skip_test:
        test(
            model=model,
            test_dataloader=dataloader,
            config=config,
            device=args.device,
            log_level=args.verbose,
            use_wandb=use_wandb
        )

    if use_wandb:
        wandb.finish()
    logger.info("Run complete. Logs saved in run.log")

if __name__ == '__main__':
    main()