import sys
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

def parse_args():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--data',
                        type=str,
                        choices=['images'],
                        default='images',
                        help='Type of data to train and test on. Default is images')
    parser.add_argument('--model',
                        type=str,
                        choices=[model.value for model in ModelEnum],
                        default='basic',
                        help='Type of model to use. Options are for SIREN, Multiplicative Filter Networks, ' +
                             'Fourier Filter Banks, Kolmogorov-Arnold Networks, and a basic coordinate-MLP, ' +
                             'respectively. Default is basic')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train for. Default is 100')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for training. Default is 64')
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
    parser.add_argument('--save_dir',
                        type=str,
                        default='saved_models',
                        help='Directory to save models in. Default is saved_models')
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

def get_model(args):
    match args.model:
        # case ModelEnum.SIREN.value:
        #     from models.siren import Siren
        #     model = Siren()
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
        case _:
            logger.error(f"Model {args.model} not recognized")
            raise ValueError(f"Model {args.model} not recognized")

    return model

def main():
    args = parse_args()
    logging.basicConfig(filename='run.log',
                        level=args.verbose,
                        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(args.verbose)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"))
    logger.addHandler(console_handler)
    validate_requirements()

    logger.debug(f"Arguments: {args}")

    logger.info("Loading model")
    model = get_model(args)
    if args.load:
        model.load_state_dict(torch.load(f"{args.save_dir}/{args.model}.pt"))
    logger.info("Model loaded")

    dataloaders = get_dataloader(args)

    logger.debug(f"Dataloaders: {dataloaders}")

    if not args.skip_train:
        model = train(model, args.data, dataloaders['train'], dataloaders['val'], args.epochs, args.lr, args.device, args.verbose)

    if args.save:
        logger.info("Saving model")
        torch.save(model.state_dict(), f"{args.save_dir}/{args.model}.pt")
        logger.info("Model saved")


    if not args.skip_test:
        test(model, args.data, dataloaders['test'], args.device, args.verbose)

    logger.info("Run complete. Logs saved in run.log")



if __name__ == '__main__':
    main()
