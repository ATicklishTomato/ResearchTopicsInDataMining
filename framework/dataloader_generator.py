import logging
import torch

logger = logging.getLogger(__name__)

def get_dataloaders(data, batch_size, log_level):
    logger.setLevel(log_level)
    logger.info("Getting dataloaders")

    pass