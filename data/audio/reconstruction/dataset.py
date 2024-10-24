from torch.utils.data import Dataset
import scipy.io.wavfile as wavfile
import numpy as np

class Reconstruction(Dataset):
    def __init__(self, path):
        super().__init__()
        self.rate, self.data = wavfile.read(path)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)
        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)
        print("Rate: %d" % self.rate)

        self.output_dimensionality = 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.rate, self.data