import logging
import os

from framework.image_reconstruction_dataloader import ImageFitting
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def get_dataloader(args):
    logger.setLevel(args.verbose)
    logger.info("Getting relevant dataloaders")

    if args.data == "images":
        logger.debug(f"Getting dataloader for image reconstruction")
        # if os.path.isfile("data/images/reconstruction/train.jpg"):
        #     dataloader = ImageFitting("data/images/reconstruction/train.jpg", 256, args.verbose)
        # else:
        #     dataloader = ImageFitting("", 256, args.verbose)
        # return {"train": dataloader, "val": None, "test": dataloader}
    
        from data.image.data import Camera
        from data.image.utils import Implicit2DWrapper
        img_dataset = Camera()
        coord_dataset = Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
        return {
            "train": DataLoader(coord_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=0)
        }

    else:
        logger.error(f"Data type {args.data} not recognized")
        raise ValueError(f"Data type {args.data} not recognized")