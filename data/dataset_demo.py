# filepath: data/dataset_demo.py
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.CIFAR10(
            root="data/processed",
            train=train,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert to numpy uint8 for compatibility with Agent
        img_np = (img.permute(1,2,0).numpy() * 255).astype('uint8')
        return img_np, label