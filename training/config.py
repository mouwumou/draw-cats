"""
Configuration module for cat style transfer project.
This file centralizes all parameters used throughout the training process.
"""

import os
import yaml
import argparse
from datetime import datetime


class Config:
    """Configuration class that loads and manages training settings."""
    
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize configuration with default values or from YAML file
        
        Args:
            config_file (str, optional): Path to YAML configuration file
            **kwargs: Override parameters
        """
        # Default configuration
        self.config = {
            # Data settings
            'DATA': {
                'data_dir': 'data',
                'real_cats_dir': 'data/real_cats',
                'artistic_cats_dir': 'data/artistic_cats',
                'processed_dir': 'data/processed',
                'examples_dir': 'data/examples',
                'image_size': 256,
                'batch_size': 1,
                'num_workers': 4,
                'unpaired': True,  # Indicates CycleGAN (unpaired) vs Pix2Pix (paired)
            },
            
            # Model settings
            'MODEL': {
                'model_type': 'cycle_gan',  # 'cycle_gan' or 'pix2pix'
                'input_channels': 3,
                'output_channels': 3,
                'ngf': 64,  # Number of generator filters
                'ndf': 64,  # Number of discriminator filters
                'n_residual_blocks': 9,
                'dropout': 0.0,
                'norm_type': 'instance',  # 'batch', 'instance', 'none'
                'init_type': 'normal',  # 'normal', 'xavier', 'kaiming', 'orthogonal'
                'init_gain': 0.02,
            },
            
            # Training settings
            'TRAINING': {
                'n_epochs': 50,
                'lr': 0.0002,
                'beta1': 0.5,
                'beta2': 0.999,
                'decay_epoch': 50,
                'lambda_identity': 5.0,  # Identity loss weight
                'lambda_cycle': 10.0,    # Cycle consistency loss weight
                'lambda_gan': 1.0,       # GAN loss weight
                'use_identity_loss': True,
                'lr_policy': 'linear',   # 'linear', 'step', 'plateau', 'cosine'
                'save_interval': 10,     # Save checkpoints every N epochs
                'sample_interval': 100,  # Generate samples every N batches
                'val_interval': 5,       # Validate every N epochs 
                'save_best_only': False, # Only save best model based on validation
            },
            
            # Checkpoint & Logging
            'CHECKPOINT': {
                'resume': '',  # Path to checkpoint to resume from
                'output_dir': 'checkpoints',
                'name': f'catGAN_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            },
            
            # Logging settings
            'LOGGING': {
                'log_dir': 'logs',
                'print_freq': 100,       # Print stats every N batches
                'use_tensorboard': True, # Use TensorBoard for logging
            }
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Override with any kwargs
        self._override_with_kwargs(kwargs)
        
        # Create output directories
        self._create_dirs()
    
    def _load_from_file(self, config_file):
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update config with file values
        for section, params in yaml_config.items():
            if section in self.config:
                self.config[section].update(params)
    
    def _override_with_kwargs(self, kwargs):
        """Override configuration with provided keyword arguments"""
        for key, value in kwargs.items():
            # Check if key uses dot notation (e.g., "DATA.batch_size")
            if '.' in key:
                section, param = key.split('.', 1)
                if section in self.config and param in self.config[section]:
                    self.config[section][param] = value
            # Otherwise, assume it's a direct parameter name
            else:
                for section in self.config:
                    if key in self.config[section]:
                        self.config[section][key] = value
    
    def _create_dirs(self):
        """Create necessary output directories"""
        dirs = [
            self.config['DATA']['processed_dir'],
            self.config['CHECKPOINT']['output_dir'],
            self.config['LOGGING']['log_dir']
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def get(self, section, param=None):
        """Get configuration value"""
        if param is None:
            return self.config[section]
        return self.config[section][param]
    
    def save(self, filename=None):
        """Save configuration to YAML file"""
        if filename is None:
            filename = os.path.join(
                self.config['CHECKPOINT']['output_dir'], 
                f"{self.config['CHECKPOINT']['name']}_config.yaml"
            )
        
        with open(filename, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getattr__(self, name):
        """Allow direct access to config sections as attributes"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __str__(self):
        """String representation of configuration"""
        import json
        return json.dumps(self.config, indent=4)


def parse_args():
    """Parse command line arguments and create config"""
    parser = argparse.ArgumentParser(description='Cat Style Transfer Training')
    
    # Basic arguments
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to YAML configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of images')
    parser.add_argument('--use_identity_loss', action='store_true',
                        help='Use identity loss')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config with args
    config = Config(
        config_file=args.config,
        **vars(args)
    )
    
    return config


# Allow import of Config class or directly create from args
if __name__ == "__main__":
    config = parse_args()
    print(config)