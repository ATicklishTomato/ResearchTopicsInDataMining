import skimage
from PIL import Image
from torch.utils.data import Dataset

class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.output_dimensionality = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img