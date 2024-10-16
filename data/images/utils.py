
import torch
import numpy as np
import scipy.ndimage
import scipy.special
import math
import matplotlib.colors as colors
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


class Implicit2DWrapper(Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Tensor of shape (c, x, y)
        img = self.transform(self.dataset[idx])

        # Apply the sobel filter to the x and y axes
        if self.compute_diff == 'gradients':
            img *= 1e1
            # Compute gradients and sum to get numpy array to shape (x, y)
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).sum(axis=0)  # Sum over channels
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).sum(axis=0)  # Sum over channels
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            # Compute laplacian and sum to get numpy array to shape (x, y)
            laplace = scipy.ndimage.laplace(img.numpy()).sum(axis=0)
        elif self.compute_diff == 'all':
            # Compute gradient & laplacian and sum to get numpy array to shape (x, y)
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).sum(axis=0)  # Sum over channels
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).sum(axis=0)  # Sum over channels
            laplace = scipy.ndimage.laplace(img.numpy()).sum(axis=0)

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)
        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.stack(
                (
                    torch.from_numpy(gradx),  # (x, y)
                    torch.from_numpy(grady)   # (x, y)
                ),
                dim=-1  # Concatenate along the last dimension to get shape (x, y, 2)
            )
            gradients = gradients.view(-1, 2)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.stack(
                (
                    torch.from_numpy(gradx),  # (x, y)
                    torch.from_numpy(grady)   # (x, y)
                ),
                dim=-1  # Concatenate along the last dimension to get shape (x, y, 2)
            )
            gradients = gradients.view(-1, 2)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict


    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict
    
def lin2img(tensor, image_resolution=None):
    # print("tensor.shape", tensor.shape)
    batch_size, num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    # Reshape the tensor into (batch_size, channels, height, width)
    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    # If there are multiple channels, we process each channel individually
    if gradients.size(0) > 1:
        grad_imgs = []
        for channel in range(gradients.size(0)):
            mG = gradients[channel].detach().permute(-2, -1, -3).cpu()

            # assumes mG is [row, cols, 2]
            nRows = mG.shape[0]
            nCols = mG.shape[1]
            mGr = mG[:, :, 0]
            mGc = mG[:, :, 1]
            mGa = np.arctan2(mGc, mGr)
            mGm = np.hypot(mGc, mGr)
            mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
            mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
            mGhsv[:, :, 1] = 1.

            nPerMin = np.percentile(mGm, 5)
            nPerMax = np.percentile(mGm, 95)
            mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
            mGm = np.clip(mGm, 0, 1)

            mGhsv[:, :, 2] = mGm
            mGrgb = colors.hsv_to_rgb(mGhsv)
            grad_imgs.append(torch.from_numpy(mGrgb).permute(2, 0, 1))

        # Stack the images for each channel back together
        return torch.stack(grad_imgs, dim=0)

    else:
        # For single-channel gradients (e.g., grayscale)
        mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

        # assumes mG is [row, cols, 2]
        nRows = mG.shape[0]
        nCols = mG.shape[1]
        mGr = mG[:, :, 0]
        mGc = mG[:, :, 1]
        mGa = np.arctan2(mGc, mGr)
        mGm = np.hypot(mGc, mGr)
        mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
        mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
        mGhsv[:, :, 1] = 1.

        nPerMin = np.percentile(mGm, 5)
        nPerMax = np.percentile(mGm, 95)
        mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
        mGm = np.clip(mGm, 0, 1)

        mGhsv[:, :, 2] = mGm
        mGrgb = colors.hsv_to_rgb(mGhsv)
        return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)