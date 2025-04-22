import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class PairedCatDataset(Dataset):
    """Dataset class for paired cat style transfer (Pix2Pix)"""
    
    def __init__(self, real_dir, artistic_dir, mode='train', transform=None, 
                 paired_transform=None, direction='real2artistic'):
        """
        Args:
            real_dir (str): Directory with real cat images
            artistic_dir (str): Directory with artistic cat images
            mode (str): 'train' or 'val' mode
            transform (callable): Transform to apply to individual images
            paired_transform (callable): Transform to apply to paired images together
            direction (str): 'real2artistic' or 'artistic2real'
        """
        self.real_dir = real_dir
        self.artistic_dir = artistic_dir
        self.mode = mode
        self.transform = transform
        self.paired_transform = paired_transform
        self.direction = direction
        
        # Get all image files - assuming one-to-one correspondence by filename
        self.real_files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG'))])
        self.artistic_files = sorted([f for f in os.listdir(artistic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG'))])
        
        # Verify pairs
        self.pairs = []
        for real_file in self.real_files:
            # Extract number from filename (cat1.PNG -> 1)
            try:
                real_num = int(''.join(filter(str.isdigit, real_file)))
                
                # Find matching artistic file
                matching_artistic = None
                for art_file in self.artistic_files:
                    art_num = int(''.join(filter(str.isdigit, art_file)))
                    if real_num == art_num:
                        matching_artistic = art_file
                        break
                
                if matching_artistic:
                    self.pairs.append((real_file, matching_artistic))
            except ValueError:
                continue
        
        print(f"Found {len(self.pairs)} valid paired images")
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        real_file, artistic_file = self.pairs[idx]
        
        # Load images
        real_path = os.path.join(self.real_dir, real_file)
        artistic_path = os.path.join(self.artistic_dir, artistic_file)
        
        real_img = Image.open(real_path).convert('RGB')
        artistic_img = Image.open(artistic_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            real_img = self.transform(real_img)
            artistic_img = self.transform(artistic_img)
        
        # Apply paired transforms if available
        if self.paired_transform:
            real_img, artistic_img = self.paired_transform(real_img, artistic_img)
        
        # Determine input and target based on direction
        if self.direction == 'real2artistic':
            input_img = real_img
            target_img = artistic_img
        else:  # artistic2real
            input_img = artistic_img
            target_img = real_img
        
        return {
            'input': input_img,
            'target': target_img,
            'real_path': real_path,
            'artistic_path': artistic_path,
        }

    
def create_side_by_side_image(input_img, target_img):
    """
    Create a combined side-by-side image for Pix2Pix training.
    
    Args:
        input_img (PIL.Image): Input image
        target_img (PIL.Image): Target image
        
    Returns:
        PIL.Image: Combined side-by-side image
    """
    width, height = input_img.width, input_img.height
    combined = Image.new('RGB', (width * 2, height))
    combined.paste(input_img, (0, 0))
    combined.paste(target_img, (width, 0))
    return combined


def prepare_pix2pix_data(real_dir, artistic_dir, output_dir, target_size=256, val_ratio=0.1):
    """
    Prepare data for Pix2Pix training by creating side-by-side images.
    
    Args:
        real_dir (str): Directory with real cat images
        artistic_dir (str): Directory with artistic cat images
        output_dir (str): Directory to save combined images
        target_size (int): Size to resize images to
        val_ratio (float): Ratio of images to use for validation
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get pairs
    dataset = PairedCatDataset(
        real_dir=real_dir,
        artistic_dir=artistic_dir,
        transform=None  # No transforms yet
    )
    
    # Split into train and validation
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    for i, (real_file, artistic_file) in enumerate(dataset.pairs):
        # Load and resize images
        real_path = os.path.join(real_dir, real_file)
        artistic_path = os.path.join(artistic_dir, artistic_file)
        
        real_img = Image.open(real_path).convert('RGB')
        artistic_img = Image.open(artistic_path).convert('RGB')
        
        # Resize
        real_img = real_img.resize((target_size, target_size), Image.LANCZOS)
        artistic_img = artistic_img.resize((target_size, target_size), Image.LANCZOS)
        
        # Create combined image
        combined = create_side_by_side_image(real_img, artistic_img)
        
        # Save to appropriate directory
        if i < train_size:
            out_path = os.path.join(train_dir, f"pair_{i:04d}.png")
        else:
            out_path = os.path.join(val_dir, f"pair_{i - train_size:04d}.png")
        
        combined.save(out_path)
    
    print(f"Processed {train_size} training pairs and {val_size} validation pairs")


# Example usage
if __name__ == "__main__":
    # Test dataset
    dataset = PairedCatDataset(
        real_dir='data/paired_real_cats',
        artistic_dir='data/paired_artistic_cats'
    )
    
    # Example preparing data for Pix2Pix
    prepare_pix2pix_data(
        real_dir='data/paired_real_cats',
        artistic_dir='data/paired_artistic_cats',
        output_dir='data/processed/pix2pix',
        target_size=256,
        val_ratio=0.1
    )