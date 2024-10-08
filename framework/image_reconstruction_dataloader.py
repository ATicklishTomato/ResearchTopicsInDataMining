import logging
import skimage
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

logger = logging.getLogger(__name__)


class ImageFitting(Dataset):
    """Borrowed and adapted from the SIREN repository."""

    def __init__(self, image, sidelength, verbose):
        super().__init__()
        logger.setLevel(verbose)
        logger.info(f"Preparing dataloader for image {image}")
        img = get_image_tensor(image, sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        logger.debug(f"Pixels shape: {self.pixels.shape}, Coords shape: {self.coords.shape}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


def get_image_tensor(image, sidelength):
    '''Loads an image and resizes it to a square image of sidelength.
    Borrowed from the SIREN repository.
    image: str
    sidelength: int'''
    if image == "":
        img = Image.fromarray(skimage.data.camera())
    else:
        img = Image.open(image)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    Borrowed from the SIREN repository.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid