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

logger = logging.getLogger(__name__)

class ModelEnum(Enum):
    SIREN = 'siren'
    MFN = 'mfn'
    FFB = 'fourier'
    KAN = 'kan'
    BASIC = 'basic'

input_dimensions = {
    'images': 2,
}

output_dimensions = {
    'images': 1,
}

def parse_args():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--data',
                        type=str,
                        choices=['images'],
                        default='images',
                        help='Type of data to train and test on. Default is images')
    parser.add_argument('--data_point',
                        type=int,
                        default=0,
                        help='Choose the index of the data_point to train on.')
    parser.add_argument('--data_fidelity',
                        type=str,
                        choices=['low', 'medium', 'high'],\
                        default='low',
                        help='Choose the fidelity of the data point to train on.')
    parser.add_argument('--model',
                        type=str,
                        choices=[model.value for model in ModelEnum],
                        default='basic',
                        help='Type of model to use. Options are for SIREN, Multiplicative Filter Networks, ' +
                             'Fourier Filter Banks, Kolmogorov-Arnold Networks, and a basic coordinate-MLP, ' +
                             'respectively. Default is basic')
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
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test',
                        help='Unique name of this experiment.')
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

def get_model(args, dataloader):
    match args.model:
        # case ModelEnum.MFN.value:
        #     from models.mfn import MFN
        #     model = MFN()
        # case ModelEnum.FFB.value:
        #     from models.fourier import FourierFilterBank
        #     model = FourierFilterBank()
        # case ModelEnum.KAN.value:
        #     from models.kan import KAN
        #     model = KAN()
        case ModelEnum.BASIC.value:
            from models.basic.basic import Basic
            model = Basic(input_dimensions[args.data], output_dimensions[args.data])
        case ModelEnum.SIREN.value:
            from models.siren import SIREN
            # Initialize the model with out_features matching the input image channel dimension
            model = SIREN(out_features=dataloader.dataset.dataset.img_channels,
                          hidden_features=64)
        case _:
            logger.error(f"Model {args.model} not recognized")
            raise ValueError(f"Model {args.model} not recognized")
    return model

def get_configuration(args):
    match args.data:
        case "images":
            from data.images.metrics import mean_squared_error
            from data.images.summary import summary
            from functools import partial
            resolution = (500, 500)
            return {
                "loss_fn": mean_squared_error, 
                "summary_fn": partial(summary, resolution),
                "resolution": resolution
            }
        case _:
            logger.error(f"Data {args.data} not recognized")
            raise ValueError(f"Data {args.data} not recognized")

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
    validate_requirements()

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
        wandb.init(project=args.model, config=wandb_config)
        logger.info("Weights and Biases initialized")

    logger.debug(f"Arguments: {args}")

    dataloader = get_dataloader(args)

    logger.debug(f"Dataloaders: {dataloader}")

    logger.info("Loading model")
    model = get_model(args, dataloader)
    config = get_configuration(args)

    if args.load:
        model.load_state_dict(torch.load(f"{args.save_dir}/{args.model}.pt"))
    logger.info("Model loaded")

    train(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        model_dir=os.path.join('./logs', args.experiment_name),
        config=config,
        device=args.device,
        log_level=args.verbose,
        use_wandb=use_wandb
    )

    test(
        model=model,
        test_dataloader=dataloader,
        model_dir=os.path.join('./logs', args.experiment_name),
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
