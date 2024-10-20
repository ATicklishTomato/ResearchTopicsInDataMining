import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset

class AudioFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.rate, self.data = wavfile.read(filename)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        self.channels = 1
        print("Rate: %d" % self.rate)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data

