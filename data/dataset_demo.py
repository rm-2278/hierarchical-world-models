import numpy as np
from torch.utils.data import Dataset

class DummyImageDataset(Dataset):
    """Random images 64x64x3 and scalar reward (for demo)."""
    def __init__(self, size=1000, img_size=64, seed=0):
        self.rng = np.random.RandomState(seed)
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = (self.rng.rand(self.img_size, self.img_size, 3) * 255).astype('uint8')
        reward = float(self.rng.rand() * 2 - 1)
        return img, reward