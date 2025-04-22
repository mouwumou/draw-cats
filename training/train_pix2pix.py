import argparse
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pix2pix import Pix2PixGAN, Pix2PixLoss, weights_init_normal
from config import Config  # Changed from training.config
from logger import Logger  # Changed from training.logger


torch.autograd.set_detect_anomaly(True)


class Pix2PixDataset(Dataset):
    """Dataset for loading Pix2Pix side-by-side images"""
    
    def __init__(self, data_dir, mode='train', transform=None, direction='AtoB'):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Base directory for the processed data
            mode (str): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied
            direction (str): 'AtoB' or 'BtoA' for direction of translation
        """
        self.data_dir = os.path.join(data_dir, mode)
        self.direction = direction
        self.transform = transform
        
        # Get all images
        self.image_files = sorted([f for f in os.listdir(self.data_dir) 
                                 if f.endswith('.png') or f.endswith('.jpg')])
        
        print(f"[{mode}] Found {len(self.image_files)} paired images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        
        # Load image
        image = plt.imread(img_path)
        
        # Input and target images are side by side
        width = image.shape[1] // 2
        
        # Split into A and B
        img_A = image[:, :width, :]
        img_B = image[:, width:, :]
        
        # Convert to [0, 1] if in [0, 255]
        if img_A.dtype == np.uint8:
            img_A = img_A.astype(np.float32) / 255.0
            img_B = img_B.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        img_A = torch.from_numpy(img_A.transpose((2, 0, 1))).float()
        img_B = torch.from_numpy(img_B.transpose((2, 0, 1))).float()
        
        # Apply any additional transforms
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        # Normalize to [-1, 1]
        img_A = (img_A - 0.5) * 2
        img_B = (img_B - 0.5) * 2
        
        # Return based on direction
        if self.direction == 'AtoB':
            return {'input': img_A, 'target': img_B, 'path': img_path}
        else:  # BtoA
            return {'input': img_B, 'target': img_A, 'path': img_path}


def create_dataloader(config):
    """Create train and validation dataloaders"""
    data_dir = config.get('DATA', 'processed_dir')
    batch_size = config.get('TRAINING', 'batch_size')
    num_workers = config.get('DATA', 'num_workers')
    
    # Directory where Pix2Pix processed data is stored
    pix2pix_dir = data_dir
    
    # Create train dataset
    train_dataset = Pix2PixDataset(
        data_dir=pix2pix_dir,
        mode='train',
        direction=config.get('MODEL', 'direction')
    )
    
    # Create validation dataset
    val_dataset = Pix2PixDataset(
        data_dir=pix2pix_dir,
        mode='val',
        direction=config.get('MODEL', 'direction')
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader


def train_pix2pix(config, logger):
    """Train Pix2Pix GAN model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not config.get('TRAINING', 'no_cuda') else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    checkpoint_dir = 'checkpoints_pix2pix'
    results_dir = 'results_pix2pix'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'val'), exist_ok=True)
    
    # Create dataloader
    train_dataloader, val_dataloader = create_dataloader(config)
    
    # Initialize model
    model = Pix2PixGAN(
        input_channels=3,
        output_channels=3,
        ngf=config.get('MODEL', 'ngf'),
        ndf=config.get('MODEL', 'ndf'),
        device=device
    )
    
    # Initialize weights
    model.G.apply(weights_init_normal)
    model.D.apply(weights_init_normal)
    
    # Initialize loss function
    criterion = Pix2PixLoss(
        lambda_l1=config.get('TRAINING', 'lambda_l1'),
        device=device
    )
    
    # Initialize optimizers
    optimizer_G = optim.Adam(
        model.G.parameters(),
        lr=config.get('TRAINING', 'lr'),
        betas=(config.get('TRAINING', 'beta1'), config.get('TRAINING', 'beta2'))
    )
    
    optimizer_D = optim.Adam(
        model.D.parameters(),
        lr=config.get('TRAINING', 'lr'),
        betas=(config.get('TRAINING', 'beta1'), config.get('TRAINING', 'beta2'))
    )
    
    # Learning rate schedulers
    # Get decay epoch, defaulting to half the total epochs if not specified
    decay_epoch = config.get('TRAINING', 'decay_epoch') if hasattr(config.config['TRAINING'], 'decay_epoch') else (config.get('TRAINING', 'n_epochs') // 2)

    # Make sure we don't have a division by zero
    if decay_epoch >= config.get('TRAINING', 'n_epochs'):
        decay_epoch = config.get('TRAINING', 'n_epochs') - 1

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=lambda epoch: 1.0 - max(0, epoch - decay_epoch) / max(1, (config.get('TRAINING', 'n_epochs') - decay_epoch))
    )
    
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(
    optimizer_D,
    lr_lambda=lambda epoch: 1.0 - max(0, epoch - decay_epoch) / max(1, (config.get('TRAINING', 'n_epochs') - decay_epoch))
    )
    
    # Start training from checkpoint if provided
    start_epoch = 0
    if config.get('CHECKPOINT', 'resume'):
        resume_path = config.get('CHECKPOINT', 'resume')
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.G.load_state_dict(checkpoint['G_state_dict'])
            model.D.load_state_dict(checkpoint['D_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            print(f"Loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{resume_path}'")
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    print(f"Starting Pix2Pix training for {config.get('TRAINING', 'n_epochs')} epochs...")
    for epoch in range(start_epoch, config.get('TRAINING', 'n_epochs')):
        start_time = time.time()
        
        # Set models to training mode
        model.G.train()
        model.D.train()
        
        # Training metrics
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Training loop
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.get('TRAINING', 'n_epochs')}")):
            # Get batch data
            input_images = batch['input'].to(device)
            target_images = batch['target'].to(device)
            model.last_input = input_images
            # Generate fake images
            fake_images = model.G(input_images)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            model.set_requires_grad(model.D, True)
            optimizer_D.zero_grad()
            
            # Calculate discriminator loss
            # Calculate discriminator loss
            loss_D, fake_preds, d_metrics = criterion.get_discriminator_loss(
                target_images, fake_images, input_images, model
            )

            # Backward pass and optimize
            loss_D.backward(retain_graph=True)  # Add retain_graph=True here
            optimizer_D.step()

            # Train Generator
            model.set_requires_grad(model.D, False)
            optimizer_G.zero_grad()

            # Calculate generator loss
            loss_G, g_metrics = criterion.get_generator_loss(
                fake_images, target_images, input_images, model
            )

            with torch.autograd.set_detect_anomaly(True):
                loss_G.backward()  # No need for retain_graph here as it's the last backward
            optimizer_G.step()

            # Clear gradients explicitly
            optimizer_G.zero_grad(set_to_none=True)
            
            # Track losses
            epoch_g_loss = epoch_g_loss + loss_G.item()
            epoch_d_loss = epoch_d_loss + loss_D.item()
            
            # Save sample images
            if i % config.get('TRAINING', 'sample_interval') == 0:
                save_sample_images(input_images, fake_images, target_images, epoch, i, 
                                  os.path.join(results_dir, 'train'))
                
                # Log batch metrics
                batch_metrics = {**g_metrics, **d_metrics}
                logger.log_metrics(batch_metrics, epoch, i, prefix='train_batch')
        
        # Calculate average epoch losses
        epoch_g_loss /= len(train_dataloader)
        epoch_d_loss /= len(train_dataloader)
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        
        # Log training metrics
        train_metrics = {
            'loss_G': epoch_g_loss,
            'loss_D': epoch_d_loss,
            'lr': optimizer_G.param_groups[0]['lr']
        }
        logger.log_metrics(train_metrics, epoch, prefix='train')
        
        # Validate model
        if (epoch + 1) % config.get('TRAINING', 'val_interval') == 0:
            val_g_loss, val_d_loss = validate_model(model, val_dataloader, criterion, device, epoch, 
                                                  os.path.join(results_dir, 'val'))
            
            # Log validation metrics
            val_metrics = {
                'loss_G': val_g_loss,
                'loss_D': val_d_loss
            }
            logger.log_metrics(val_metrics, epoch, prefix='val')
            
            # Check if this is the best model
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                # Save best model
                save_checkpoint(model, optimizer_G, optimizer_D, epoch, 
                               os.path.join(checkpoint_dir, 'best_model.pth'), best=True)
                print(f"New best model saved with val_loss_G = {val_g_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % config.get('TRAINING', 'save_interval') == 0:
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, 
                           os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
            
            # Also save as latest
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, 
                         os.path.join(checkpoint_dir, 'latest_model.pth'))
        
        # Print epoch stats
        time_elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{config.get('TRAINING', 'n_epochs')}] "
              f"[G loss: {epoch_g_loss:.4f}] [D loss: {epoch_d_loss:.4f}] "
              f"[Time: {time_elapsed:.2f}s]")
    
    # Save final model
    save_checkpoint(model, optimizer_G, optimizer_D, config.get('TRAINING', 'n_epochs') - 1, 
                   os.path.join(checkpoint_dir, 'final_model.pth'))
    
    print("Pix2Pix training completed!")


def validate_model(model, val_dataloader, criterion, device, epoch, save_dir):
    """Validate model on validation set"""
    # Set models to evaluation mode
    model.G.eval()
    model.D.eval()
    
    val_g_loss = 0
    val_d_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validation")):
            # Get batch data
            input_images = batch['input'].to(device)
            target_images = batch['target'].to(device)

            model.last_input = input_images

            # Generate fake images
            fake_images = model.G(input_images)

            # Resize input to match fake/target image shape
            if input_images.shape[2:] != fake_images.shape[2:]:
                input_images = F.interpolate(input_images, size=fake_images.shape[2:], mode='bilinear', align_corners=False)

            # Calculate discriminator loss
            fake_preds = model.D(input_images, fake_images)
            loss_D, _, _ = criterion.get_discriminator_loss(
                target_images, fake_images, input_images, model
            )

            # Calculate generator loss
            loss_G, _ = criterion.get_generator_loss(fake_images, target_images, fake_preds, model)

            # Track losses
            val_g_loss += loss_G.item()
            val_d_loss += loss_D.item()

            # Save sample images (first batch only)
            if i == 0:
                save_sample_images(input_images, fake_images, target_images, epoch, i, save_dir)

    # Calculate average validation losses
    val_g_loss /= len(val_dataloader)
    val_d_loss /= len(val_dataloader)

    # Print validation stats
    print(f"[Validation] [G loss: {val_g_loss:.4f}] [D loss: {val_d_loss:.4f}]")

    # Set models back to training mode
    model.G.train()
    model.D.train()

    return val_g_loss, val_d_loss



def save_sample_images(input_images, fake_images, target_images, epoch, batch_i, save_dir):
    """Save sample images during training"""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalize images
    def denorm(tensor):
        return (tensor + 1) / 2
    
    # Take only the first few images
    n_samples = min(5, input_images.size(0))
    input_images = input_images[:n_samples]
    fake_images = fake_images[:n_samples]
    target_images = target_images[:n_samples]
    
    # Denormalize
    input_images = denorm(input_images)
    fake_images = denorm(fake_images)
    target_images = denorm(target_images)
    
    # Create image grid
    image_grid = torch.cat([input_images, fake_images, target_images], dim=3)
    image_grid = make_grid(image_grid, nrow=1, normalize=False)
    
    # Save grid
    save_path = f"{save_dir}/epoch_{epoch+1}_batch_{batch_i}.png"
    save_image(image_grid, save_path, normalize=False)


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, filename, best=False):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'G_state_dict': model.G.state_dict(),
        'D_state_dict': model.D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'is_best': best
    }
    torch.save(state, filename)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Pix2Pix GAN')
    
    # Arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/processed/pix2pix',
                        help='Directory with processed paired data')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam optimizer beta1')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                        help='Weight for L1 loss')
    parser.add_argument('--direction', type=str, default='AtoB',
                        choices=['AtoB', 'BtoA'],
                        help='Direction of translation (AtoB: real->artistic, BtoA: artistic->real)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config(config_file=args.config)
    else:
        # Create default config and override with args
        config = Config()
        
        # Update config with command line arguments
        if args.data_dir:
            config.config['DATA']['processed_dir'] = args.data_dir
        if args.n_epochs:
            config.config['TRAINING']['n_epochs'] = args.n_epochs
        if args.batch_size:
            config.config['TRAINING']['batch_size'] = args.batch_size
        if args.lr:
            config.config['TRAINING']['lr'] = args.lr
        if args.beta1:
            config.config['TRAINING']['beta1'] = args.beta1
        if args.lambda_l1:
            config.config['TRAINING']['lambda_l1'] = args.lambda_l1
        if args.direction:
            config.config['MODEL']['direction'] = args.direction
        if args.resume:
            config.config['CHECKPOINT']['resume'] = args.resume
        if args.no_cuda:
            config.config['TRAINING']['no_cuda'] = args.no_cuda
    
    # Add Pix2Pix specific configuration
    if 'lambda_l1' not in config.config['TRAINING']:
        config.config['TRAINING']['lambda_l1'] = 100.0
    if 'direction' not in config.config['MODEL']:
        config.config['MODEL']['direction'] = 'AtoB'
    
    # Create logger
    logger = Logger(config)
    
    # Save configuration
    config.save()
    
    # Train model
    train_pix2pix(config, logger)
    
    # Close logger
    logger.close()


if __name__ == "__main__":
    main()