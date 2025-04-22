#!/usr/bin/env python
"""
Batch preprocessing script for cat style transfer project.

This script performs one-time preprocessing of all images:
- Resizes and centers images while maintaining aspect ratio
- Creates training and validation splits
- Analyzes dataset statistics
- Visualizes sample processed images

Usage:
    python preprocess.py --real_dir data/real_cats --artistic_dir data/artistic_cats
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from project modules
try:
    from data.preprocessing import preprocess_centralized_image
    print("Successfully imported preprocessing module")
except ImportError as e:
    print(f"Error importing preprocessing module: {e}")
    
    # Fallback implementation if import fails
    def preprocess_centralized_image(image_path, target_size=256):
        """Fallback implementation if import fails"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                return None
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize while maintaining aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h, new_w = target_size, int(w * target_size / h)
            else:
                new_h, new_w = int(h * target_size / w), target_size
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create a square canvas and center the resized image
            square_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
            offset_h = (target_size - new_h) // 2
            offset_w = (target_size - new_w) // 2
            square_img[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = img
            
            return square_img
        except Exception as e:
            print(f"Error in fallback preprocess_centralized_image: {e}")
            return None

# Try to import Config, but provide fallback
try:
    from training.config import Config
except ImportError:
    print("Could not import Config class, using dictionary instead")
    
    class Config:
        def __init__(self, config_file=None, **kwargs):
            self.config = {
                'DATA': {
                    'data_dir': 'data',
                    'real_cats_dir': 'data/real_cats',
                    'artistic_cats_dir': 'data/artistic_cats',
                    'processed_dir': 'data/processed',
                    'image_size': 256,
                }
            }
            
            # Override with any kwargs
            for key, value in kwargs.items():
                if key == 'data_dir':
                    self.config['DATA']['data_dir'] = value
                elif key == 'real_dir':
                    self.config['DATA']['real_cats_dir'] = value
                elif key == 'artistic_dir':
                    self.config['DATA']['artistic_cats_dir'] = value
                elif key == 'target_size':
                    self.config['DATA']['image_size'] = value

        def get(self, section, param=None):
            if param is None:
                return self.config[section]
            return self.config[section][param]


def create_directories(config):
    """Create necessary directories for processed images"""
    # Base processed directory
    processed_dir = Path(config.get('DATA', 'processed_dir'))
    
    # Training directories
    real_train_dir = processed_dir / 'real'
    artistic_train_dir = processed_dir / 'artistic'
    
    # Validation directories
    real_val_dir = processed_dir / 'real_val'
    artistic_val_dir = processed_dir / 'artistic_val'
    
    # Visualization directory
    vis_dir = processed_dir / 'visualization'
    
    # Create all directories
    for directory in [processed_dir, real_train_dir, artistic_train_dir, 
                     real_val_dir, artistic_val_dir, vis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return {
        'processed_dir': processed_dir,
        'real_train_dir': real_train_dir,
        'artistic_train_dir': artistic_train_dir,
        'real_val_dir': real_val_dir,
        'artistic_val_dir': artistic_val_dir,
        'vis_dir': vis_dir
    }


def get_image_files(directory):
    """Get all image files from a directory"""
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return []
    
    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.PNG']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    print(f"Looking in {directory} for images")
    if image_files:
        print(f"Found extensions: {[os.path.splitext(f)[1] for f in image_files[:5]]}")
    else:
        print(f"No image files found in {directory}")
        print(f"Directory content: {os.listdir(directory) if os.path.exists(directory) else 'Directory not found'}")
    
    return sorted(image_files)


def process_images(image_files, output_dir, target_size, overwrite=False):
    """Process all images in a directory and save to output directory"""
    print(f"Processing {len(image_files)} images to {output_dir}...")
    
    # Track successful and failed images
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each image
    for img_path in tqdm(image_files):
        # Generate output path
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        # Skip if output exists and overwrite is False
        if os.path.exists(output_path) and not overwrite:
            skipped = skipped + 1
            continue
        
        try:
            # Process image using imported function from preprocessing.py
            processed_img = preprocess_centralized_image(img_path, target_size)
            
            if processed_img is not None:
                # Convert from RGB to BGR for OpenCV
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                
                # Save processed image
                cv2.imwrite(output_path, processed_img_bgr)
                successful = successful + 1
            else:
                failed = failed + 1
                print(f"Warning: Failed to process {img_path}")
        except Exception as e:
            failed = failed + 1
            print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Processing complete: {successful} successful, {failed} failed, {skipped} skipped")
    return successful, failed, skipped


def create_validation_split(dirs, val_ratio=0.1, random_seed=42):
    """Create validation split from processed images"""
    print(f"Creating validation split with ratio {val_ratio}...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all processed images
    real_images = list(Path(dirs['real_train_dir']).glob('*.png'))
    artistic_images = list(Path(dirs['artistic_train_dir']).glob('*.png'))
    
    print(f"Found {len(real_images)} real images and {len(artistic_images)} artistic images for creating validation split")
    
    # Check if we have enough images
    if len(real_images) == 0 or len(artistic_images) == 0:
        print("Not enough images to create validation split. Skipping.")
        return
    
    # Calculate validation set size
    num_real_val = max(1, int(len(real_images) * val_ratio))
    num_artistic_val = max(1, int(len(artistic_images) * val_ratio))
    
    # Make sure we don't try to sample more than we have
    num_real_val = min(num_real_val, len(real_images))
    num_artistic_val = min(num_artistic_val, len(artistic_images))
    
    # Randomly select validation images
    real_val_images = random.sample(real_images, num_real_val)
    artistic_val_images = random.sample(artistic_images, num_artistic_val)
    
    # Move real validation images
    for img_path in real_val_images:
        dest_path = Path(dirs['real_val_dir']) / img_path.name
        # Copy instead of move to keep original files
        shutil.copy2(img_path, dest_path)
    
    # Move artistic validation images
    for img_path in artistic_val_images:
        dest_path = Path(dirs['artistic_val_dir']) / img_path.name
        # Copy instead of move to keep original files
        shutil.copy2(img_path, dest_path)
    
    print(f"Created validation set with {num_real_val} real images and {num_artistic_val} artistic images")


def analyze_datasets(dirs):
    """Analyze and print statistics about the datasets"""
    print("\nDataset Statistics:")
    
    # Get image counts
    real_train_count = len(list(Path(dirs['real_train_dir']).glob('*.png')))
    artistic_train_count = len(list(Path(dirs['artistic_train_dir']).glob('*.png')))
    real_val_count = len(list(Path(dirs['real_val_dir']).glob('*.png')))
    artistic_val_count = len(list(Path(dirs['artistic_val_dir']).glob('*.png')))
    
    # Print counts
    print(f"Training set - Real cats: {real_train_count}, Artistic cats: {artistic_train_count}")
    print(f"Validation set - Real cats: {real_val_count}, Artistic cats: {artistic_val_count}")
    
    # Calculate sample image statistics
    if real_train_count > 0 and artistic_train_count > 0:
        # Sample a few images to analyze
        real_sample = list(Path(dirs['real_train_dir']).glob('*.png'))
        artistic_sample = list(Path(dirs['artistic_train_dir']).glob('*.png'))
        
        if real_sample and artistic_sample:
            # Load sample images
            real_img = cv2.imread(str(real_sample[0]))
            artistic_img = cv2.imread(str(artistic_sample[0]))
            
            if real_img is not None and artistic_img is not None:
                # Print image dimensions and channels
                print(f"\nSample image dimensions:")
                print(f"Real cat image: {real_img.shape}")
                print(f"Artistic cat image: {artistic_img.shape}")
    
    # Check for dataset balance
    if real_train_count > 0 and artistic_train_count > 0:
        if abs(real_train_count - artistic_train_count) > max(real_train_count, artistic_train_count) * 0.2:  # More than 20% difference
            print("\nWarning: Training sets are imbalanced. This might affect training.")
            print("Consider adding more images to the smaller set or using data augmentation.")


def visualize_samples(dirs, num_samples=4):
    """Create and save visualization of sample processed images"""
    # Get sample images
    real_images = list(Path(dirs['real_train_dir']).glob('*.png'))
    artistic_images = list(Path(dirs['artistic_train_dir']).glob('*.png'))
    
    if len(real_images) == 0 or len(artistic_images) == 0:
        print("Not enough images for visualization")
        return
    
    # Limit number of samples
    num_samples = min(num_samples, len(real_images), len(artistic_images))
    
    if num_samples == 0:
        print("No images available for visualization")
        return
    
    # Select random samples
    random.seed(42)  # For reproducibility
    real_samples = random.sample(real_images, num_samples)
    artistic_samples = random.sample(artistic_images, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
    
    # Handle the case where num_samples=1
    if num_samples == 1:
        axes = np.array(axes).reshape(2, 1)
    
    # Plot real cat images
    for i, img_path in enumerate(real_samples):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Real Cat {i+1}")
            axes[0, i].axis('off')
    
    # Plot artistic cat images
    for i, img_path in enumerate(artistic_samples):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Artistic Cat {i+1}")
            axes[1, i].axis('off')
    
    # Add main title
    plt.suptitle("Sample Processed Images", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    vis_path = Path(dirs['vis_dir']) / 'sample_processed_images.png'
    plt.savefig(vis_path)
    plt.close()
    
    print(f"Sample visualization saved to {vis_path}")


def create_examples(dirs, num_examples=5):
    """Create example folder with selected images for Streamlit app"""
    # Create examples directory
    examples_dir = Path(dirs['processed_dir']).parent / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all processed images
    real_images = list(Path(dirs['real_train_dir']).glob('*.png'))
    
    if len(real_images) == 0:
        print("No images available for examples")
        return
    
    # Limit number of examples
    num_examples = min(num_examples, len(real_images))
    
    if num_examples == 0:
        print("No images available for examples")
        return
    
    # Select random examples
    random.seed(42)  # For reproducibility
    example_images = random.sample(real_images, num_examples)
    
    # Copy examples
    for i, img_path in enumerate(example_images):
        dest_path = examples_dir / f"example_cat_{i+1}.png"
        shutil.copy2(img_path, dest_path)
    
    print(f"Created {num_examples} example images in {examples_dir}")


def create_sample_data(real_dir, artistic_dir, target_size=256):
    """Create sample data if no images are found"""
    print("No images found in data directories. Creating sample data...")
    
    # Ensure directories exist
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(artistic_dir, exist_ok=True)
    
    # Create a simple sample image (256x256 colored square)
    real_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    # Draw a simple cat shape
    cv2.circle(real_img, (128, 128), 100, (200, 150, 150), -1)  # Face
    cv2.circle(real_img, (90, 90), 20, (100, 100, 100), -1)     # Left eye
    cv2.circle(real_img, (166, 90), 20, (100, 100, 100), -1)    # Right eye
    cv2.triangle = np.array([[128, 128], [108, 160], [148, 160]], np.int32)
    cv2.fillPoly(real_img, [cv2.triangle], (150, 100, 100))     # Nose
    
    # Create a simple artistic version
    artistic_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    cv2.rectangle(artistic_img, (78, 78), (178, 178), (150, 150, 200), -1)  # Face
    cv2.circle(artistic_img, (100, 100), 10, (50, 50, 50), -1)              # Left eye
    cv2.circle(artistic_img, (156, 100), 10, (50, 50, 50), -1)              # Right eye
    
    # Save sample images
    for i in range(5):
        cv2.imwrite(os.path.join(real_dir, f"sample_cat_{i+1}.png"), real_img)
        cv2.imwrite(os.path.join(artistic_dir, f"sample_drawing_{i+1}.png"), artistic_img)
    
    print(f"Created 5 sample images in each directory")


def main():
    """Main preprocessing function"""
    # Get current directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess cat images for style transfer')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to configuration file')
    parser.add_argument('--real_dir', type=str, default='data/real_cats',
                        help='Directory with real cat images')
    parser.add_argument('--artistic_dir', type=str, default='data/artistic_cats',
                        help='Directory with artistic cat images')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Target image size (square)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing processed images')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--create_samples', action='store_true',
                        help='Create sample data if no images found')
    
    args = parser.parse_args()
    
    # Create configuration
    try:
        if args.config:
            config = Config(config_file=args.config)
        else:
            config = Config()
            # Override config with args
            config.config['DATA']['real_cats_dir'] = args.real_dir
            config.config['DATA']['artistic_cats_dir'] = args.artistic_dir
            config.config['DATA']['image_size'] = args.target_size
    except Exception as e:
        print(f"Error creating config: {e}")
        # Fallback config
        config = Config(
            data_dir='data',
            real_dir=args.real_dir,
            artistic_dir=args.artistic_dir,
            target_size=args.target_size
        )
    
    # Print configuration
    print("\n--- Cat Style Transfer Preprocessing ---")
    print(f"Real cats directory: {config.get('DATA', 'real_cats_dir')}")
    print(f"Artistic cats directory: {config.get('DATA', 'artistic_cats_dir')}")
    print(f"Target image size: {config.get('DATA', 'image_size')}")
    print(f"Validation ratio: {args.val_ratio}")
    
    # Start timing
    start_time = time.time()
    
    # Create directories
    dirs = create_directories(config)
    
    # Get image files
    real_image_files = get_image_files(config.get('DATA', 'real_cats_dir'))
    artistic_image_files = get_image_files(config.get('DATA', 'artistic_cats_dir'))
    
    print(f"Found {len(real_image_files)} real cat images")
    print(f"Found {len(artistic_image_files)} artistic cat images")
    
    # Create sample data if requested and no images found
    if args.create_samples and (len(real_image_files) == 0 or len(artistic_image_files) == 0):
        create_sample_data(
            config.get('DATA', 'real_cats_dir'),
            config.get('DATA', 'artistic_cats_dir'),
            config.get('DATA', 'image_size')
        )
        # Refresh image lists
        real_image_files = get_image_files(config.get('DATA', 'real_cats_dir'))
        artistic_image_files = get_image_files(config.get('DATA', 'artistic_cats_dir'))
    
    # Process images
    target_size = config.get('DATA', 'image_size')
    real_success, real_failed, real_skipped = process_images(
        real_image_files, dirs['real_train_dir'], target_size, args.overwrite
    )
    
    artistic_success, artistic_failed, artistic_skipped = process_images(
        artistic_image_files, dirs['artistic_train_dir'], target_size, args.overwrite
    )
    
    # Only continue if we have processed images
    if real_success > 0 and artistic_success > 0:
        # Create validation split
        create_validation_split(dirs, args.val_ratio)
        
        # Analyze datasets
        analyze_datasets(dirs)
        
        # Visualize samples
        visualize_samples(dirs, args.num_samples)
        
        # Create examples for Streamlit app
        create_examples(dirs)
    else:
        print("\nNot enough processed images to continue with validation split and analysis.")
        print("Please check that your image directories contain valid image files.")
    
    # Report total time
    total_time = time.time() - start_time
    print(f"\nPreprocessing completed in {total_time:.1f} seconds")
    print(f"Processed images are ready in: {dirs['processed_dir']}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Verify your processed images in the data/processed directory")
    print("2. Run training: python training/train.py")
    print("3. After training, visualize results: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()