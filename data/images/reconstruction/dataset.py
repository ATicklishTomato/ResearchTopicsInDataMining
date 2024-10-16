from PIL import Image
from torch.utils.data import Dataset

class Reconstruction(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.img = Image.open(fp=path)
        
        # Determine the number of channels based on the image mode
        if self.img.mode == 'L':  # Grayscale image
            self.img_channels = 1
        elif self.img.mode == 'RGB':  # RGB image
            self.img_channels = 3
        elif self.img.mode == 'RGBA':  # RGBA image
            self.img_channels = 4
        else:
            raise ValueError(f"Unsupported image mode: {self.img.mode}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img