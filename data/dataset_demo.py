# filepath: data/dataset_demo.py
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, img_size=64):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
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
        # Return tensor in [0, 1] range as (H, W, C) for compatibility
        img_np = img.permute(1, 2, 0).numpy()
        return img_np, label