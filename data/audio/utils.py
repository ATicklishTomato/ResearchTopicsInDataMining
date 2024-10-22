import numpy as np
import torch

class ImplicitAudioWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)
        self.grid = self.grid.astype(np.float32)
        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rate, data = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = (data / scale)
        data = torch.Tensor(data).view(-1, 1)
        return {'idx': idx, 'coords': self.grid}, {'func': data, 'rate': rate, 'scale': scale}