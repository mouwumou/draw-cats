#!/usr/bin/env python
"""
Script to continue training Pix2Pix with improved parameters to increase validation performance.
This script resumes from an existing checkpoint and applies various improvements:
1. Reduced learning rate
2. Increased dropout
3. Enhanced data augmentation
4. Adjusted loss weights
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import time
import numpy as np
import random

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from model.pix2pix import Pix2PixGAN, Pix2PixLoss, weights_init_normal
from config import Config
from logger import Logger

class EnhancedPix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode='train', direction='AtoB'):
        import matplotlib.pyplot as plt
        from PIL import Image

        self.data_dir = os.path.join(data_dir, mode)
        self.direction = direction
        self.image_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.png') or f.endswith('.jpg')])

        print(f"[{mode}] Found {len(self.image_files)} paired images")

        self.transform = None
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        import matplotlib.pyplot as plt
        from PIL import Image

        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = plt.imread(img_path)

        if self.transform and np.random.random() > 0.5:
            pil_image = Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float32 else image)
            image = np.array(self.transform(pil_image)).astype(np.float32) / 255.0

        width = image.shape[1] // 2
        img_A = image[:, :width, :]
        img_B = image[:, width:, :]

        if img_A.dtype == np.uint8:
            img_A = img_A.astype(np.float32) / 255.0
            img_B = img_B.astype(np.float32) / 255.0

        if np.random.random() > 0.5 and self.transform:
            img_A = np.fliplr(img_A).copy()
            img_B = np.fliplr(img_B).copy()

        img_A = torch.from_numpy(img_A.transpose((2, 0, 1))).float()
        img_B = torch.from_numpy(img_B.transpose((2, 0, 1))).float()

        img_A = (img_A - 0.5) * 2
        img_B = (img_B - 0.5) * 2

        return {'input': img_A, 'target': img_B, 'path': img_path} if self.direction == 'AtoB' else {'input': img_B, 'target': img_A, 'path': img_path}

def create_enhanced_dataloader(config):
    data_dir = config.get('DATA', 'processed_dir')

    # Safe fallback for batch_size
    try:
        batch_size = config.get('TRAINING', 'batch_size')
    except:
        batch_size = 2

    # Safe fallback for num_workers
    try:
        num_workers = config.get('DATA', 'num_workers')
    except:
        num_workers = 4

    # Safe fallback for direction
    try:
        direction = config.get('MODEL', 'direction')
    except:
        direction = 'AtoB'

    pix2pix_dir = data_dir if 'pix2pix' in data_dir else os.path.join(data_dir, 'pix2pix')

    train_dataset = EnhancedPix2PixDataset(pix2pix_dir, mode='train', direction=direction)
    val_dataset = EnhancedPix2PixDataset(pix2pix_dir, mode='val', direction=direction)

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True), \
           torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



def increase_dropout(model, dropout_rate=0.7):
    for name, module in model.G.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"Updating dropout layer {name} from {module.p} to {dropout_rate}")
            module.p = dropout_rate
    return model


def continue_training(checkpoint_path, output_dir, additional_epochs=100, new_lr=0.00002, 
                     lambda_l1=200.0, dropout_rate=0.7, batch_size=2):
    """Continue training with improved parameters"""
    # Create configuration with improved settings
    config = Config()
    
    # Update configuration for continued training
    config.config['TRAINING']['n_epochs'] = additional_epochs
    config.config['TRAINING']['lr'] = new_lr
    config.config['TRAINING']['lambda_l1'] = lambda_l1
    config.config['CHECKPOINT']['resume'] = checkpoint_path
    config.config['CHECKPOINT']['output_dir'] = output_dir
    
    # Add batch_size to the config
    if 'batch_size' not in config.config['TRAINING']:
        config.config['TRAINING']['batch_size'] = batch_size
    
    # Create new output directories
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join('results_continued', 'pix2pix')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'val'), exist_ok=True)
    
    # Create logger
    logger = Logger(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders with enhanced augmentation
    train_dataloader, val_dataloader = create_enhanced_dataloader(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Initialize model
    model = Pix2PixGAN(
        input_channels=3,
        output_channels=3,
        ngf=config.get('MODEL', 'ngf') if 'ngf' in config.config['MODEL'] else 64,
        ndf=config.get('MODEL', 'ndf') if 'ndf' in config.config['MODEL'] else 64,
        device=device
    )
    
    # Load state from checkpoint
    model.G.load_state_dict(checkpoint['G_state_dict'])
    model.D.load_state_dict(checkpoint['D_state_dict'])
    
    # Increase dropout rate
    model = increase_dropout(model, dropout_rate)
    
    # Initialize loss function with increased L1 weight
    criterion = Pix2PixLoss(
        lambda_l1=lambda_l1,
        device=device
    )
    
    # Initialize optimizers with reduced learning rate
    optimizer_G = optim.Adam(
        model.G.parameters(),
        lr=new_lr,
        betas=(config.get('TRAINING', 'beta1') if 'beta1' in config.config['TRAINING'] else 0.5, 
               config.get('TRAINING', 'beta2') if 'beta2' in config.config['TRAINING'] else 0.999)
    )
    
    optimizer_D = optim.Adam(
        model.D.parameters(),
        lr=new_lr,
        betas=(config.get('TRAINING', 'beta1') if 'beta1' in config.config['TRAINING'] else 0.5,
               config.get('TRAINING', 'beta2') if 'beta2' in config.config['TRAINING'] else 0.999)
    )
    
    # Learning rate schedulers with gentler decay
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=lambda epoch: 1.0 - (epoch / additional_epochs) ** 1.5  # Slower decay
    )
    
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(
        optimizer_D,
        lr_lambda=lambda epoch: 1.0 - (epoch / additional_epochs) ** 1.5  # Slower decay
    )
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    print(f"Starting continued Pix2Pix training for {additional_epochs} epochs...")
    for epoch in range(additional_epochs):
        start_time = time.time()
        
        # Set models to training mode
        model.G.train()
        model.D.train()
        
        # Training metrics
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Training loop
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{additional_epochs}")):
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
            loss_D, fake_preds, d_metrics = criterion.get_discriminator_loss(
                target_images, fake_images, input_images, model
            )
            
            # Apply gradient penalty to improve stability
            if random.random() < 0.1:  # Apply sporadically to save computation
                # Interpolate between real and fake
                alpha = torch.rand(target_images.size(0), 1, 1, 1, device=device)
                interpolates = alpha * target_images + (1 - alpha) * fake_images.detach()
                interpolates.requires_grad_(True)
                
                # Calculate gradients
                disc_interpolates = model.D(input_images, interpolates)
                gradients = torch.autograd.grad(
                    outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones_like(disc_interpolates),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                
                # Calculate gradient penalty
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
                loss_D = loss_D + gradient_penalty
            
            # Backward pass and optimize
            loss_D.backward(retain_graph=True)
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            model.set_requires_grad(model.D, False)
            optimizer_G.zero_grad()
            
            # Calculate generator loss
            loss_G, g_metrics = criterion.get_generator_loss(
                fake_images, target_images, fake_preds, model
            )
            
            # Add feature matching loss to improve quality
            if hasattr(model.D, 'get_intermediate_features'):
                real_features = model.D.get_intermediate_features(input_images, target_images)
                fake_features = model.D.get_intermediate_features(input_images, fake_images)
                feature_matching_loss = 0
                for real_feat, fake_feat in zip(real_features, fake_features):
                    feature_matching_loss += nn.L1Loss()(fake_feat, real_feat.detach()) * 10.0
                loss_G += feature_matching_loss
            
            # Backward pass and optimize
            loss_G.backward()
            optimizer_G.step()
            
            # Track losses
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
            # Save sample images
            if i % 50 == 0:  # Save samples every 50 batches
                from torchvision.utils import save_image
                
                # Denormalize images
                def denorm(tensor):
                    return (tensor + 1) / 2
                
                # Save images
                imgs = [denorm(input_images), denorm(fake_images), denorm(target_images)]
                img_grid = torch.cat(imgs, dim=3)  # Side by side
                save_image(img_grid[0], f"{results_dir}/train/epoch_{epoch+1}_batch_{i}.png")
                
                # Log metrics for this batch
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
        val_g_loss, val_d_loss = validate_model(model, val_dataloader, criterion, device, epoch, 
                                                results_dir)
        
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
                           os.path.join(output_dir, 'best_model.pth'), best=True)
            print(f"New best model saved with val_loss_G = {val_g_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == additional_epochs - 1:
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, 
                           os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            
            # Also save as latest
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, 
                         os.path.join(output_dir, 'latest_model.pth'))
        
        # Print epoch stats
        time_elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{additional_epochs}] "
              f"[G loss: {epoch_g_loss:.4f}] [D loss: {epoch_d_loss:.4f}] "
              f"[Val G loss: {val_g_loss:.4f}] [Val D loss: {val_d_loss:.4f}] "
              f"[Time: {time_elapsed:.2f}s]")
    
    # Save final model
    save_checkpoint(model, optimizer_G, optimizer_D, additional_epochs - 1, 
                   os.path.join(output_dir, 'final_model.pth'))
    
    print("Continued Pix2Pix training completed!")
    

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
            
            # Calculate discriminator loss
            fake_preds = model.D(input_images, fake_images)
            loss_D, _, _ = criterion.get_discriminator_loss(
                target_images, fake_images, input_images, model
            )
            
            # Calculate generator loss
            loss_G, _ = criterion.get_generator_loss(
                fake_images, target_images, fake_preds, model
            )
            
            # Track losses
            val_g_loss += loss_G.item()
            val_d_loss += loss_D.item()
            
            # Save sample images (first batch only)
            if i == 0:
                from torchvision.utils import save_image
                
                # Denormalize images
                def denorm(tensor):
                    return (tensor + 1) / 2
                
                # Save images
                imgs = [denorm(input_images), denorm(fake_images), denorm(target_images)]
                img_grid = torch.cat(imgs, dim=3)  # Side by side
                save_image(img_grid[0], f"{save_dir}/val/epoch_{epoch+1}_batch_{i}.png")
    
    # Calculate average validation losses
    val_g_loss /= len(val_dataloader)
    val_d_loss /= len(val_dataloader)
    
    # Print validation stats
    print(f"[Validation] [G loss: {val_g_loss:.4f}] [D loss: {val_d_loss:.4f}]")
    
    # Set models back to training mode
    model.G.train()
    model.D.train()
    
    return val_g_loss, val_d_loss


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, filename, best=False):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='checkpoints_improved')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    config = Config()
    config.config['TRAINING'].update({'n_epochs': args.epochs, 'lr': args.lr, 'lambda_l1': args.lambda_l1, 'batch_size': args.batch_size})
    config.config['CHECKPOINT'].update({'resume': args.checkpoint, 'output_dir': args.output_dir})
    logger = Logger(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader = create_enhanced_dataloader(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Pix2PixGAN(3, 3, 64, 64, device=device)
    model.G.load_state_dict(checkpoint['G_state_dict'])
    model.D.load_state_dict(checkpoint['D_state_dict'])
    model = increase_dropout(model, args.dropout)

    criterion = Pix2PixLoss(lambda_l1=args.lambda_l1, device=device)
    optimizer_G = optim.Adam(model.G.parameters(), lr=args.lr * 1.0, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.D.parameters(), lr=args.lr * 4.0, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda e: 1.0 - (e / args.epochs))
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda e: 1.0 - (e / args.epochs))

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.G.train(), model.D.train()
        epoch_g, epoch_d = 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_images = batch['input'].to(device)
            target_images = batch['target'].to(device)
            fake_images = model.G(input_images)

            model.set_requires_grad(model.D, True)
            optimizer_D.zero_grad()
            loss_D, fake_preds, _ = criterion.get_discriminator_loss(target_images, fake_images, input_images, model)
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            model.set_requires_grad(model.D, False)
            optimizer_G.zero_grad()
            loss_G, _ = criterion.get_generator_loss(fake_images, target_images, input_images, model)
            loss_G.backward()
            optimizer_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()

        scheduler_G.step()
        scheduler_D.step()

        val_g_loss, val_d_loss = 0, 0
        model.G.eval(), model.D.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_images = batch['input'].to(device)
                target_images = batch['target'].to(device)
                fake_images = model.G(input_images)
                fake_preds = model.D(input_images, fake_images)
                d_loss, _, _ = criterion.get_discriminator_loss(target_images, fake_images, input_images, model)
                g_loss, _ = criterion.get_generator_loss(fake_images, target_images, input_images, model)
                val_d_loss += d_loss.item()
                val_g_loss += g_loss.item()

        val_g_loss /= len(val_loader)
        val_d_loss /= len(val_loader)
        logger.log_metrics({'loss_G': val_g_loss, 'loss_D': val_d_loss}, epoch, prefix='val')

        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, os.path.join(args.output_dir, 'best_model.pth'), best=True)
            print(f"New best model saved with val_loss_G = {val_g_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()