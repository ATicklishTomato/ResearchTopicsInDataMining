import sys
import os
from argparse import ArgumentParser
from enum import Enum

import torch

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

hidden_dimensions = {
    'images': 16,
}

hidden_layers = {
    'images': [16,16]
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
    parser.add_argument('--skip_train',
                        action='store_true',
                        help='Skip training and only evaluate the model. Default is False')
    parser.add_argument('--skip_test',
                        action='store_true',
                        help='Skip testing and only train the model. Default is False')
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
            from data.images.metrics import mean_squared_error
            from data.images.summary import summary
            from functools import partial
            resolution = (500, 500)
            return {
                "loss_fn": mean_squared_error, 
                "summary_fn": partial(summary, resolution),
                "resolution": resolution,
                "in_features": input_dimensions[args.data],
                "out_features": output_dimensions[args.data],
                "hidden_dim": hidden_dimensions[args.data],
                "hidden_layers": hidden_layers[args.data]
            }
        case _:
            logger.error(f"Data {args.data} not recognized")
            raise ValueError(f"Data {args.data} not recognized")

def get_model(args, dataloader, config):
    match args.model:
        case ModelEnum.MFN.value:
            from models.mfn import GaborNet
            model = GaborNet(in_size=config["in_features"], hidden_size=config["hidden_dim"], out_size=dataloader.dataset.dataset.img_channels, n_layers=3, input_scale=256, weight_scale=1)
        case ModelEnum.FFB.value:
            from models.NFFB.img.NFFB_2d import NFFB
            model = NFFB(config["in_features"], dataloader.dataset.dataset.img_channels)
        case ModelEnum.KAN.value:
            from models.kan import KAN, KANLinear
            model = KAN(layers_hidden=[config["in_features"], *config["hidden_layers"], dataloader.dataset.dataset.img_channels])
        case ModelEnum.BASIC.value:
            from models.basic.basic import Basic
            model = Basic(input_dimensions[args.data], output_dimensions[args.data])
        case ModelEnum.SIREN.value:
            from models.siren import SIREN
            model = SIREN(in_features=config["in_features"], out_features=dataloader.dataset.dataset.img_channels, hidden_features=config["hidden_dim"])
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



    if not args.skip_train:
        train(
            model=model,
            dataloader=dataloader,
            epochs=args.epochs,
            lr=args.lr,
            model_dir=os.path.join('./logs', args.experiment_name),
            config=config,
            device=args.device,
            log_level=args.verbose
        )

    if not args.skip_test:
        test(model, args.data, dataloader, args.device, args.verbose)

    logger.info("Run complete. Logs saved in run.log")

if __name__ == '__main__':
    main()