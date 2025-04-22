import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CatStyleDataset(Dataset):
    """Dataset for loading preprocessed cat images for style transfer"""

    def __init__(self, data_dir, domain, mode='train', transform=None):
        self.data_dir = data_dir
        self.domain = domain
        self.mode = mode

        if mode == 'train':
            self.img_dir = os.path.join(data_dir, 'processed', domain)
        else:
            self.img_dir = os.path.join(data_dir, 'processed', f'{domain}_val')

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print(f"[{mode}] Loaded {len(self.img_files)} {domain} images from {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image}  # ✅ NOW real_batch['image'] works!




class UnpairedDataLoader:
    """DataLoader for CycleGAN that loads unpaired images from two domains"""
    
    def __init__(self, data_dir, batch_size=1, image_size=256, num_workers=4, mode='train'):
        """
        Args:
            data_dir (str): Base directory for the dataset
            batch_size (int): Batch size
            image_size (int): Size of the images
            num_workers (int): Number of worker threads for dataloader
            mode (str): 'train' or 'val'
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mode = mode
        
        # Define transforms
        if mode == 'train':
            # For training: include data augmentation
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            # For validation: only normalize
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        # Create datasets for both domains
        self.real_dataset = CatStyleDataset(
            data_dir=data_dir,
            domain='real',
            mode=mode,
            transform=self.transform
        )
        
        self.artistic_dataset = CatStyleDataset(
            data_dir=data_dir,
            domain='artistic',
            mode=mode,
            transform=self.transform
        )
        
        # Create dataloaders
        self.real_dataloader = DataLoader(
            self.real_dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers
        )
        
        self.artistic_dataloader = DataLoader(
            self.artistic_dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers
        )
        
        # Define iterators
        self.real_iter = iter(self.real_dataloader)
        self.artistic_iter = iter(self.artistic_dataloader)
        
        self.real_size = len(self.real_dataloader)
        self.artistic_size = len(self.artistic_dataloader)
    
    def next_batch(self):
        try:
            real = next(self.real_iter)
        except StopIteration:
            self.real_iter = iter(self.real_dataloader)
            real = next(self.real_iter)

        try:
            artistic = next(self.artistic_iter)
        except StopIteration:
            self.artistic_iter = iter(self.artistic_dataloader)
            artistic = next(self.artistic_iter)

        return real, artistic  # ✅ must return tensors, not file paths
    
    def __len__(self):
        """Length is determined by the maximum of the two domain sizes"""
        return max(self.real_size, self.artistic_size)

    def next_batch(self):
        try:
            real = next(self.real_iter)
        except StopIteration:
            self.real_iter = iter(self.real_dataloader)
            real = next(self.real_iter)

        try:
            artistic = next(self.artistic_iter)
        except StopIteration:
            self.artistic_iter = iter(self.artistic_dataloader)
            artistic = next(self.artistic_iter)

        # real and artistic are dicts like {'image': tensor of shape [1, 3, 256, 256]}
        # we want to extract the image and squeeze out the batch dim
        return {'image': real['image'].squeeze(0)}, {'image': artistic['image'].squeeze(0)}






# Example usage
if __name__ == "__main__":
    data_dir = 'data'
    batch_size = 4
    
    # Create dataloader
    dataloader = UnpairedDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=256,
        num_workers=4,
        mode='train'
    )
    
    # Get a batch of images
    real_batch, artistic_batch = dataloader.next_batch()
    
    print(f"Real batch shape: {real_batch['image'].shape}")
    print(f"Artistic batch shape: {artistic_batch['image'].shape}")