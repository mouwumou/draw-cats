import argparse
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules 
from model.gan import CycleGAN
from data.dataset import CatStyleDataset, UnpairedDataLoader

class ReplayBuffer:
    """Buffer for storing previously generated images to update discriminators"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []
    
    def push_and_pop(self, data):
        """Push data into buffer and pop older data if buffer is full"""
        result = []
        for element in data.detach():
            element = element.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                result.append(element)
            else:
                if np.random.random() > 0.5:
                    i = np.random.randint(0, self.max_size)
                    temp = self.buffer[i].clone()
                    self.buffer[i] = element
                    result.append(temp)
                else:
                    result.append(element)
        return torch.cat(result)

class LambdaLR:
    """Learning rate scheduler with linear decay"""
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    """Initialize network weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train_cycle_gan(config):
    """Main training function for CycleGAN"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not config.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/train', exist_ok=True)
    os.makedirs('results/validation', exist_ok=True)
    
    # Create dataloader
    dataloader = UnpairedDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        mode='train'
    )
    
    val_dataloader = UnpairedDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        mode='val'
    )
    
    # Initialize model
    model = CycleGAN(device=device)
    
    # Weight initialization
    model.G_real_to_artistic.apply(weights_init_normal)
    model.G_artistic_to_real.apply(weights_init_normal)
    model.D_real.apply(weights_init_normal)
    model.D_artistic.apply(weights_init_normal)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(model.G_real_to_artistic.parameters()) + 
        list(model.G_artistic_to_real.parameters()),
        lr=config.lr, betas=(0.5, 0.999)
    )
    
    optimizer_D_real = optim.Adam(
        model.D_real.parameters(), lr=config.lr, betas=(0.5, 0.999)
    )
    
    optimizer_D_artistic = optim.Adam(
        model.D_artistic.parameters(), lr=config.lr, betas=(0.5, 0.999)
    )
    
    # Learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / (config.n_epochs - config.decay_epoch)
    )
    
    lr_scheduler_D_real = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_real, lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / (config.n_epochs - config.decay_epoch)
    )
    
    lr_scheduler_D_artistic = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_artistic, lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / (config.n_epochs - config.decay_epoch)
    )
    
    # Buffers for updating discriminators
    fake_real_buffer = ReplayBuffer()
    fake_artistic_buffer = ReplayBuffer()
    
    # Track losses
    g_losses = []
    d_losses = []
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if config.resume:
        if os.path.isfile(config.resume):
            print(f"Loading checkpoint '{config.resume}'")
            checkpoint = torch.load(config.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.G_real_to_artistic.load_state_dict(checkpoint['G_real_to_artistic_state_dict'])
            model.G_artistic_to_real.load_state_dict(checkpoint['G_artistic_to_real_state_dict'])
            model.D_real.load_state_dict(checkpoint['D_real_state_dict'])
            model.D_artistic.load_state_dict(checkpoint['D_artistic_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D_real.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
            optimizer_D_artistic.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
            g_losses = checkpoint.get('g_losses', [])
            d_losses = checkpoint.get('d_losses', [])
            print(f"Loaded checkpoint '{config.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{config.resume}'")
    
    # Start training
    print(f"Starting training for {config.n_epochs} epochs...")
    
    for epoch in range(start_epoch, config.n_epochs):
        start_time = time.time()
        
        # Training metrics
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Set models to training mode
        model.G_real_to_artistic.train()
        model.G_artistic_to_real.train()
        model.D_real.train()
        model.D_artistic.train()
        
        # Training loop
        for i in tqdm(range(len(dataloader)), desc=f"Epoch {epoch+1}/{config.n_epochs}"):
            # Get batch of images
            real_batch, artistic_batch = dataloader.next_batch()
            real_A = real_batch['image'].to(device) 
            real_B = artistic_batch['image'].to(device)

            # Get batch size for current batch
            batch_size = real_A.size(0)

            # Dynamically match label shapes
            with torch.no_grad():
                pred_real_shape_A = model.D_real(real_A)
                pred_real_shape_B = model.D_artistic(real_B)
            real_label_A = torch.ones_like(pred_real_shape_A).to(device)
            fake_label_A = torch.zeros_like(pred_real_shape_A).to(device)
            real_label_B = torch.ones_like(pred_real_shape_B).to(device)
            fake_label_B = torch.zeros_like(pred_real_shape_B).to(device)

            # ------------------
            # Train Generators
            # ------------------
            optimizer_G.zero_grad()

            if config.use_identity_loss:
                identity_B = model.G_real_to_artistic(real_B)
                loss_identity_B = criterion_identity(identity_B, real_B) * 5.0
                identity_A = model.G_artistic_to_real(real_A)
                loss_identity_A = criterion_identity(identity_A, real_A) * 5.0
                loss_identity = loss_identity_A + loss_identity_B
            else:
                loss_identity = 0

            fake_B = model.G_real_to_artistic(real_A)
            pred_fake_B = model.D_artistic(fake_B)
            loss_GAN_A = criterion_GAN(pred_fake_B, real_label_B)

            fake_A = model.G_artistic_to_real(real_B)
            pred_fake_A = model.D_real(fake_A)
            loss_GAN_B = criterion_GAN(pred_fake_A, real_label_A)

            recovered_A = model.G_artistic_to_real(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * 10.0
            recovered_B = model.G_real_to_artistic(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * 10.0

            loss_G = loss_GAN_A + loss_GAN_B + loss_cycle_A + loss_cycle_B + loss_identity
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            # Train Discriminator A
            # -----------------------
            optimizer_D_real.zero_grad()
            pred_real = model.D_real(real_A)
            loss_D_real_real = criterion_GAN(pred_real, real_label_A)

            fake_A_ = fake_real_buffer.push_and_pop(fake_A)
            pred_fake = model.D_real(fake_A_.detach())
            loss_D_real_fake = criterion_GAN(pred_fake, fake_label_A)

            loss_D_real = (loss_D_real_real + loss_D_real_fake) * 0.5
            loss_D_real.backward()
            optimizer_D_real.step()

            # -----------------------
            # Train Discriminator B
            # -----------------------
            optimizer_D_artistic.zero_grad()
            pred_real = model.D_artistic(real_B)
            loss_D_artistic_real = criterion_GAN(pred_real, real_label_B)

            fake_B_ = fake_artistic_buffer.push_and_pop(fake_B)
            pred_fake = model.D_artistic(fake_B_.detach())
            loss_D_artistic_fake = criterion_GAN(pred_fake, fake_label_B)

            loss_D_artistic = (loss_D_artistic_real + loss_D_artistic_fake) * 0.5
            loss_D_artistic.backward()
            optimizer_D_artistic.step()

            loss_D = loss_D_real + loss_D_artistic
            epoch_g_loss = epoch_g_loss + loss_G.item()
            epoch_d_loss = epoch_d_loss + loss_D.item()
            
            # Save generated samples
            if i % config.sample_interval == 0:
                save_sample_images(epoch, i, real_A, real_B, fake_A, fake_B, recovered_A, recovered_B)
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_real.step()
        lr_scheduler_D_artistic.step()
        
        # Calculate average epoch losses
        epoch_g_loss /= len(dataloader)
        epoch_d_loss /= len(dataloader)
        
        # Record losses
        g_losses.append(epoch_g_loss)
        d_losses.append(epoch_d_loss)
        
        # Print epoch stats
        time_elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{config.n_epochs}] "
              f"[G loss: {epoch_g_loss:.4f}] [D loss: {epoch_d_loss:.4f}] "
              f"[Time: {time_elapsed:.2f}s]")
        
        # Validate model
        if (epoch + 1) % config.val_interval == 0:
            val_g_loss, val_d_loss = validate_model(model, val_dataloader, device, criterion_GAN, 
                                                  criterion_cycle, criterion_identity, epoch, config)
            print(f"[Validation] [G loss: {val_g_loss:.4f}] [D loss: {val_d_loss:.4f}]")
        
        # Save model checkpoint
        if (epoch + 1) % config.save_interval == 0 or epoch == config.n_epochs - 1:
            save_checkpoint(model, optimizer_G, optimizer_D_real, optimizer_D_artistic, 
                           epoch, g_losses, d_losses, config)
    
    # Save final loss curve
    plot_losses(g_losses, d_losses)
    
    print("Training completed!")

def validate_model(model, val_dataloader, device, criterion_GAN, criterion_cycle, 
                 criterion_identity, epoch, config):
    """Validate model on validation set"""
    model.G_real_to_artistic.eval()
    model.G_artistic_to_real.eval()
    model.D_real.eval()
    model.D_artistic.eval()
    
    val_g_loss = 0
    val_d_loss = 0
    
    with torch.no_grad():
        for i in tqdm(range(min(len(val_dataloader), 50)), desc="Validation"):
            # Get batch of images
            real_batch, artistic_batch = val_dataloader.next_batch()
            real_A = real_batch['image'].to(device)
            real_B = artistic_batch['image'].to(device)


            batch_size = real_A.size(0)

    # Dynamically shaped validation labels
    with torch.no_grad():
        pred_real_shape_A = model.D_real(real_A)
        pred_real_shape_B = model.D_artistic(real_B)
    real_label_A = torch.ones_like(pred_real_shape_A).to(device)
    fake_label_A = torch.zeros_like(pred_real_shape_A).to(device)
    real_label_B = torch.ones_like(pred_real_shape_B).to(device)
    fake_label_B = torch.zeros_like(pred_real_shape_B).to(device)

    # Forward pass
    fake_B = model.G_real_to_artistic(real_A)
    fake_A = model.G_artistic_to_real(real_B)

    recovered_A = model.G_artistic_to_real(fake_B)
    recovered_B = model.G_real_to_artistic(fake_A)

    # Calculate generator losses
    pred_fake_B = model.D_artistic(fake_B)
    pred_fake_A = model.D_real(fake_A)

    loss_GAN_A = criterion_GAN(pred_fake_B, real_label_B)
    loss_GAN_B = criterion_GAN(pred_fake_A, real_label_A)

    # Cycle loss
    loss_cycle_A = criterion_cycle(recovered_A, real_A) * 10.0
    loss_cycle_B = criterion_cycle(recovered_B, real_B) * 10.0

    # Identity loss
    if config.use_identity_loss:
        identity_B = model.G_real_to_artistic(real_B)
        identity_A = model.G_artistic_to_real(real_A)
        loss_identity_B = criterion_identity(identity_B, real_B) * 5.0
        loss_identity_A = criterion_identity(identity_A, real_A) * 5.0
        loss_identity = loss_identity_A + loss_identity_B
    else:
        loss_identity = 0

    # Total generator loss
    loss_G = loss_GAN_A + loss_GAN_B + loss_cycle_A + loss_cycle_B + loss_identity

    # Calculate discriminator losses
    # Real loss for D_A
    pred_real_A = model.D_real(real_A)
    loss_D_real_real = criterion_GAN(pred_real_A, real_label_A)

    # Fake loss for D_A
    pred_fake_A = model.D_real(fake_A)
    loss_D_real_fake = criterion_GAN(pred_fake_A, fake_label_A)

    loss_D_real = (loss_D_real_real + loss_D_real_fake) * 0.5

    # Real loss for D_B
    pred_real_B = model.D_artistic(real_B)
    loss_D_artistic_real = criterion_GAN(pred_real_B, real_label_B)

    # Fake loss for D_B
    pred_fake_B = model.D_artistic(fake_B)
    loss_D_artistic_fake = criterion_GAN(pred_fake_B, fake_label_B)

    loss_D_artistic = (loss_D_artistic_real + loss_D_artistic_fake) * 0.5

    # Total discriminator loss
    loss_D = loss_D_real + loss_D_artistic

    # Track validation losses
    val_g_loss = val_g_loss + loss_G.item()
    val_d_loss = val_d_loss + loss_D.item()

    # Save validation samples
    if i == 0:
        save_validation_images(epoch, real_A, real_B, fake_A, fake_B)

    
    # Calculate average validation losses
    val_g_loss /= min(len(val_dataloader), 50)
    val_d_loss /= min(len(val_dataloader), 50)
    
    # Set models back to training mode
    model.G_real_to_artistic.train()
    model.G_artistic_to_real.train()
    model.D_real.train()
    model.D_artistic.train()
    
    return val_g_loss, val_d_loss

def save_sample_images(epoch, batch_i, real_A, real_B, fake_A, fake_B, recovered_A, recovered_B):
    """Save training samples"""
    result_dir = 'results/train'
    os.makedirs(result_dir, exist_ok=True)
    
    # Denormalize
    def denorm(x):
        return (x + 1) / 2
    
    # Save individual samples
    imgs = [real_A, fake_B, recovered_A, real_B, fake_A, recovered_B]
    img_names = ['real_A', 'fake_B', 'recovered_A', 'real_B', 'fake_A', 'recovered_B']
    
    # Create a batch grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (img, name) in enumerate(zip(imgs, img_names)):
        img = denorm(img)
        img = img.cpu().detach().permute(1, 2, 0).numpy()  # Take first sample in batch
        axes[i].imshow(img)
        axes[i].set_title(name)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/epoch_{epoch+1}_batch_{batch_i}.png')
    plt.close()

def save_validation_images(epoch, real_A, real_B, fake_A, fake_B):
    """Save validation images"""
    result_dir = 'results/validation'
    os.makedirs(result_dir, exist_ok=True)
    
    # Denormalize
    def denorm(x):
        return (x + 1) / 2
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Real images (original domain)
    img = denorm(real_A)[0].cpu().detach().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Real Cat')
    axes[0, 0].axis('off')
    
    # Fake images (target domain)
    img = denorm(fake_B)[0].cpu().detach().permute(1, 2, 0).numpy()
    axes[0, 1].imshow(img)
    axes[0, 1].set_title('Generated Drawing')
    axes[0, 1].axis('off')
    
    # Real images (target domain)
    img = denorm(real_B)[0].cpu().detach().permute(1, 2, 0).numpy()
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Real Drawing')
    axes[1, 0].axis('off')
    
    # Fake images (original domain)
    img = denorm(fake_A)[0].cpu().detach().permute(1, 2, 0).numpy()
    axes[1, 1].imshow(img)
    axes[1, 1].set_title('Generated Cat')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/validation_epoch_{epoch+1}.png')
    plt.close()

def save_checkpoint(model, optimizer_G, optimizer_D_A, optimizer_D_B, epoch, g_losses, d_losses, config):
    """Save model checkpoint"""
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'G_real_to_artistic_state_dict': model.G_real_to_artistic.state_dict(),
        'G_artistic_to_real_state_dict': model.G_artistic_to_real.state_dict(),
        'D_real_state_dict': model.D_real.state_dict(),
        'D_artistic_state_dict': model.D_artistic.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
        'config': vars(config)
    }
    
    # Save checkpoint
    torch.save(checkpoint, f'{checkpoint_dir}/model_epoch_{epoch+1}.pth')
    
    # Also save as latest
    torch.save(checkpoint, f'{checkpoint_dir}/latest_model.pth')
    
    print(f"Checkpoint saved at epoch {epoch+1}")

def plot_losses(g_losses, d_losses):
    """Plot and save loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    plt.savefig('results/loss_curve.png')
    plt.close()

def main():
    """Main function for training script"""
    parser = argparse.ArgumentParser(description='Train CycleGAN for cat style transfer')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='base directory for data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--image_size', type=int, default=256,
                        help='image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of worker threads for dataloader')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start learning rate decay')
    parser.add_argument('--use_identity_loss', action='store_true',
                        help='use identity loss')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many epochs to wait before saving model')
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='how many batches to wait before saving sample images')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='how many epochs to wait before validating')
    
    config = parser.parse_args()
    
    # Train model
    train_cycle_gan(config)

if __name__ == "__main__":
    main()