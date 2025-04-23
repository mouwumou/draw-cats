from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import random
import os

class Pix2PixDataset(Dataset):
    def __init__(self, root, img_size):
        super().__init__()
        self.files = list(Path(root).glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size*2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.transform(img)
        # split pair
        c, h, w = img.shape
        real = img[:, :, :w//2]
        fake = img[:, :, w//2:]
        return {'real': real, 'fake': fake}

class UnpairedDataset(Dataset):
    def __init__(self, root, img_size):
        super().__init__()
        self.files = list(Path(root).glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.transform(img)


def make_dataloader(cfg):
    if cfg.mode == 'pix2pix':
        dataset = Pix2PixDataset(os.path.join(cfg.data_root, 'pix2pix', 'train'), cfg.img_size)
    else:
        dsA = UnpairedDataset(os.path.join(cfg.data_root, 'A'), cfg.img_size)
        dsB = UnpairedDataset(os.path.join(cfg.data_root, 'B'), cfg.img_size)
        dataset = list(zip(dsA, dsB))  # returns tuple of (A,B)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
