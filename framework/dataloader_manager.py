import logging
import os

from framework.image_reconstruction_dataloader import ImageFitting

logger = logging.getLogger(__name__)

def get_dataloader(args):
    logger.setLevel(args.verbose)
    logger.info("Getting relevant dataloaders")

    if args.data == "images":
        logger.debug(f"Getting dataloader for image reconstruction")
        if os.path.isfile("data/images/reconstruction/train.jpg"):
            dataloader = ImageFitting("data/images/reconstruction/train.jpg", 256, args.verbose)
        else:
            dataloader = ImageFitting("", 256, args.verbose)
        return {"train": dataloader, "val": None, "test": dataloader}
    else:
        logger.error(f"Data type {args.data} not recognized")
        raise ValueError(f"Data type {args.data} not recognized")