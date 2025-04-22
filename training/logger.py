"""
Logging module for cat style transfer project.
Handles tracking and visualization of training progress.
"""

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import csv

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Logger class to track training progress and visualize results.
    
    Features:
    - Track and save metrics (loss values, learning rates)
    - Save generated images during training
    - Create visualizations of training progress
    - TensorBoard integration (optional)
    """
    
    def __init__(self, config):
        """
        Initialize logger
        
        Args:
            config: Configuration object containing logging settings
        """
        self.config = config
        
        # Set up log directories
        self.log_dir = Path(config.get('LOGGING', 'log_dir'))
        self.run_name = config.get('CHECKPOINT', 'name')
        
        # Create run-specific log directory
        self.run_dir = self.log_dir / self.run_name
        self.img_dir = self.run_dir / 'images'
        self.checkpoint_dir = Path(config.get('CHECKPOINT', 'output_dir'))
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {}
        self.best_metric = float('inf')  # For tracking best model (lower is better)
        
        # Set up CSV logging
        self.csv_path = self.run_dir / 'metrics.csv'
        self.csv_file = None
        self.csv_writer = None
        
        # Initialize TensorBoard if available and enabled
        self.writer = None
        if config.get('LOGGING', 'use_tensorboard') and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
        
        # Log the configuration
        self.log_config(config)
        
        print(f"Logger initialized. Logs will be saved to {self.run_dir}")
    
    def log_config(self, config):
        """Save configuration to log directory"""
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config.config, f, indent=4)
    
    def log_metrics(self, metrics, epoch, step=None, prefix='train'):
        """
        Log training metrics
        
        Args:
            metrics (dict): Dictionary of metric values
            epoch (int): Current epoch
            step (int, optional): Current step within epoch
            prefix (str): Prefix for metric names (e.g., 'train', 'val')
        """
        # Initialize CSV file if needed
        if self.csv_file is None:
            is_new_file = not self.csv_path.exists()
            self.csv_file = open(self.csv_path, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if new file
            if is_new_file:
                header = ['epoch', 'step']
                for key in metrics.keys():
                    header.append(f"{prefix}_{key}")
                self.csv_writer.writerow(header)
        
        # Add to CSV
        row = [epoch]
        row.append(step if step is not None else '')
        row.extend(metrics.values())
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        # Add to metrics tracking
        for key, value in metrics.items():
            metric_key = f"{prefix}_{key}"
            if metric_key not in self.metrics:
                self.metrics[metric_key] = []
            self.metrics[metric_key].append((epoch, step, value))
        
        # Log to TensorBoard if available
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, 
                                      step if step is not None else epoch)
        
        # Print metrics if at print frequency
        print_freq = self.config.get('LOGGING', 'print_freq')
        should_print = step is None or (step % print_freq == 0)
        
        if should_print:
            metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            epoch_info = f"Epoch {epoch}"
            step_info = f"Step {step}" if step is not None else ""
            print(f"[{timestamp}] {epoch_info} {step_info} | {prefix} | {metrics_str}")
    
    def log_learning_rate(self, lr, epoch):
        """Log learning rate"""
        if self.writer:
            self.writer.add_scalar('training/learning_rate', lr, epoch)
        
        # Add to metrics
        if 'learning_rate' not in self.metrics:
            self.metrics['learning_rate'] = []
        self.metrics['learning_rate'].append((epoch, None, lr))
    
    def save_images(self, images, epoch, batch=None, normalize=True, nrow=3):
        """
        Save a grid of images during training
        
        Args:
            images (dict): Dictionary of image tensors
            epoch (int): Current epoch
            batch (int, optional): Current batch
            normalize (bool): Whether to normalize images from [-1, 1] to [0, 1]
            nrow (int): Number of images per row in the grid
        """
        try:
            import torchvision
            
            # Determine output filename
            if batch is not None:
                filepath = self.img_dir / f'epoch_{epoch}_batch_{batch}.png'
            else:
                filepath = self.img_dir / f'epoch_{epoch}.png'
            
            # Create a list of tensors in the order we want to display
            img_list = []
            for key in ['real', 'fake_artistic', 'recovered_real', 'artistic', 'fake_real', 'recovered_artistic']:
                if key in images:
                    # Add a label to each image
                    img = images[key].detach().cpu()
                    # Only take the first few images if there are many
                    if img.size(0) > 2:
                        img = img[:2]
                    img_list.append(img)
            
            # Make grid
            grid = torchvision.utils.make_grid(
                torch.cat(img_list), nrow=nrow, normalize=normalize, padding=2
            )
            
            # Save grid
            torchvision.utils.save_image(grid, filepath)
            
            # Log to TensorBoard if available
            if self.writer and (batch is None or batch % 100 == 0):
                self.writer.add_image(f'images/grid', grid, epoch)
                
                # Also log individual images
                for key, img in images.items():
                    if img.size(0) > 0:  # Make sure tensor isn't empty
                        # Only log first image in batch
                        self.writer.add_image(f'images/{key}', 
                                           img[0].detach().cpu(), 
                                           epoch,
                                           dataformats='CHW')
            
            return filepath
        except Exception as e:
            print(f"Error saving images: {e}")
            return None
    
    def plot_losses(self, save=True):
        """
        Plot training and validation losses
        
        Args:
            save (bool): Whether to save the plot to disk
        
        Returns:
            Path to saved plot if save=True, None otherwise
        """
        # Filter metrics to get loss values
        loss_metrics = {k: v for k, v in self.metrics.items() if 'loss' in k.lower()}
        
        if not loss_metrics:
            return None
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for name, values in loss_metrics.items():
            # Extract epochs and values (ignoring step)
            epochs = [x[0] for x in values]
            metric_values = [x[2] for x in values]
            
            plt.plot(epochs, metric_values, label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if save:
            plot_path = self.run_dir / 'loss_plot.png'
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        
        plt.show()
        plt.close()
        return None
    
    def check_best_model(self, metric_value, model, optimizer_G, optimizer_D_A, optimizer_D_B, 
                         epoch, metric_name='val_loss_G'):
        """
        Check if current model is the best and save if it is
        
        Args:
            metric_value (float): Current metric value
            model: Model to save
            optimizer_G: Generator optimizer
            optimizer_D_A: Discriminator A optimizer
            optimizer_D_B: Discriminator B optimizer
            epoch (int): Current epoch
            metric_name (str): Name of metric to track for best model
        
        Returns:
            bool: True if this is the best model so far, False otherwise
        """
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'G_real_to_artistic_state_dict': model.G_real_to_artistic.state_dict(),
                'G_artistic_to_real_state_dict': model.G_artistic_to_real.state_dict(),
                'D_real_state_dict': model.D_real.state_dict(),
                'D_artistic_state_dict': model.D_artistic.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
                f'{metric_name}': metric_value
            }
            
            # Save checkpoint
            best_path = self.checkpoint_dir / f'{self.run_name}_best.pth'
            torch.save(checkpoint, best_path)
            
            print(f"New best model saved with {metric_name} = {metric_value:.4f}")
            return True
        
        return False
    
    def close(self):
        """Close the logger and finalize"""
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            self.writer = None
        
        # Create final plots
        self.plot_losses(save=True)