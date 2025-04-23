from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import random
import os

class Pix2PixDataset(Dataset):
    """
    Dataset for paired image-to-image translation (Pix2Pix).
    Expects directory structure:
      root/train/*.png (concatenated real|fake images)
      root/val/*.png
    """
    def __init__(self, root, img_size):
        super().__init__()
        self.root = Path(root)
        self.files = list(self.root.glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size*2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        _, h, w = img.shape
        real = img[:, :, :w//2]
        fake = img[:, :, w//2:]
        return {'real': real, 'fake': fake}

class UnpairedDataset(Dataset):
    """
    Dataset for unpaired images (CycleGAN).
    Expects directory structure:
      root/train/*.png
      root/val/*.png
    """
    def __init__(self, root, img_size):
        super().__init__()
        self.root = Path(root)
        self.files = list(self.root.glob('*.png'))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)

class PairedUnpairedDataset(Dataset):
    """
    Helper to zip two unpaired datasets for CycleGAN training.
    Returns tuples (real_A, real_B).
    """
    def __init__(self, dsA, dsB):
        super().__init__()
        self.dsA = dsA
        self.dsB = dsB
        self.len = min(len(dsA), len(dsB))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.dsA[idx], self.dsB[idx]


def make_dataloader(cfg, split='train'):
    """
    Create DataLoader for training or validation.

    Args:
      cfg: Config object with mode, data_root, img_size, batch_size, num_workers.
      split: 'train' or 'val'.

    Returns:
      DataLoader instance.
    """
    shuffle = split == 'train'
    drop_last = split == 'train'
    if cfg.mode == 'pix2pix':
        root = os.path.join(cfg.data_root, split)
        dataset = Pix2PixDataset(root, cfg.img_size)
    else:
        rootA = os.path.join(cfg.data_root, 'A', split)
        rootB = os.path.join(cfg.data_root, 'B', split)
        dsA = UnpairedDataset(rootA, cfg.img_size)
        dsB = UnpairedDataset(rootB, cfg.img_size)
        dataset = PairedUnpairedDataset(dsA, dsB)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=getattr(cfg, 'num_workers', 4),
        drop_last=drop_last
    )
