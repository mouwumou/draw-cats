import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from pathlib import Path
from PIL import Image
import random
import os

class Pix2PixDataset(Dataset):
    """
    Paired dataset for Pix2Pix with separate geometric and color augmentations.
    Expects:
      data_root/train/*.png  (pairs concatenated as [real|fake])
      data_root/val/*.png
    """
    def __init__(self, root, img_size, split='train'):
        self.files = list(Path(root).glob('*.png'))
        self.img_size = img_size
        self.split = split
        # geometric transforms applied equally to real & fake
        self.geo = transforms.Compose([
            transforms.Resize((img_size, img_size * 2)),
        ])
        # color jitter on input and weaker jitter on target
        self.real_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.fake_jitter = transforms.ColorJitter(0.05, 0.05, 0.05, 0.02)
        # final to tensor & normalize
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        # random erasing on input tensor
        self.random_erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.geo(img)
        w,h = img.size
        real_img = img.crop((0,0,w//2,h))
        fake_img= img.crop((w//2,0,w,h))

        if self.split=='train':
            # 1) 随机裁剪
            i,j,h0,w0 = transforms.RandomResizedCrop.get_params(
                real_img, scale=(0.8,1.0), ratio=(0.9,1.1))
            real_img = F.resized_crop(real_img,i,j,h0,w0,(self.img_size,self.img_size))
            fake_img= F.resized_crop(fake_img,i,j,h0,w0,(self.img_size,self.img_size))
            # 2) 翻转/旋转
            if random.random()<0.5:
                real_img, fake_img = F.hflip(real_img), F.hflip(fake_img)
            ang = random.uniform(-10,10)
            real_img, fake_img = F.rotate(real_img,ang), F.rotate(fake_img,ang)
            # 3) 色彩抖动
            real_img = self.real_jitter(real_img)
            fake_img= self.fake_jitter(fake_img)
        
        # ToTensor & Normalize
        real = self.norm(self.to_tensor(real_img))
        fake= self.norm(self.to_tensor(fake_img))

        if self.split=='train':
            # 4) RandomErasing + GaussianNoise
            real = self.random_erasing(real)
            real = real + torch.randn_like(real)*0.02
            real = real.clamp(-1,1)

        return {'real':real, 'fake':fake}
    

class UnpairedDataset(Dataset):
    """
    Unpaired dataset for CycleGAN with augmentation.
    Expects:
      data_root/A/train, data_root/B/train, etc.
    """
    def __init__(self, root, img_size, split='train'):
        self.files = list(Path(root).glob('*.png'))
        self.split = split
        self.base = transforms.Resize((img_size, img_size))
        self.color = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5,) * 3, (0.5,) * 3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.base(img)
        if self.split == 'train':
            if random.random() < 0.5:
                img = F.hflip(img)
            img = F.rotate(img, random.uniform(-10, 10))
            img = self.color(img)
        img = self.norm(self.to_tensor(img))
        return img

class PairedUnpairedDataset(Dataset):
    """Zip two unpaired sets for CycleGAN"""
    def __init__(self, dsA, dsB):
        self.dsA = dsA
        self.dsB = dsB
        self.length = min(len(dsA), len(dsB))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dsA[idx], self.dsB[idx]


def make_dataloader(cfg, split='train', pin_memory=False):
    shuffle = split == 'train'
    drop_last = split == 'train'
    if cfg.mode == 'pix2pix':
        path = os.path.join(cfg.data_root, split)
        dataset = Pix2PixDataset(path, cfg.img_size, split)
    else:
        pathA = os.path.join(cfg.data_root, 'A', split)
        pathB = os.path.join(cfg.data_root, 'B', split)
        dsA = UnpairedDataset(pathA, cfg.img_size, split)
        dsB = UnpairedDataset(pathB, cfg.img_size, split)
        dataset = PairedUnpairedDataset(dsA, dsB)
    return DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=getattr(cfg, 'num_workers', 8),
        drop_last=drop_last, 
        pin_memory=pin_memory,
    )
