#!/usr/bin/env python
"""
Batch preprocessing script for paired images for Pix2Pix GAN.

This script performs one-time preprocessing of paired cat images:
- Matches real and artistic cat images by their numerical identifier
- Resizes images to a uniform size
- Creates side-by-side combined images for Pix2Pix training
- Splits data into training and validation sets

Usage:
    python preprocess_pix2pix.py --real_dir data/paired_real_cats --artistic_dir data/paired_artistic_cats
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
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_image_files(directory):
    """Get all image files from a directory"""
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return []
    
    # Find all image files with both lowercase and uppercase extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
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

def extract_number(filename):
    """Extract numerical identifier from filename"""
    # Extract all digits from the filename
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[0])
    return None

def find_matching_pairs(real_files, artistic_files):
    """Find matching pairs between real and artistic cat images"""
    pairs = []
    
    # Create dictionaries with numeric identifiers
    real_dict = {}
    for real_file in real_files:
        base_name = os.path.basename(real_file)
        num = extract_number(base_name)
        if num is not None:
            real_dict[num] = real_file
    
    artistic_dict = {}
    for artistic_file in artistic_files:
        base_name = os.path.basename(artistic_file)
        num = extract_number(base_name)
        if num is not None:
            artistic_dict[num] = artistic_file
    
    # Find matching pairs
    for num in real_dict:
        if num in artistic_dict:
            pairs.append((real_dict[num], artistic_dict[num]))
    
    return pairs

def create_side_by_side_image(real_path, artistic_path, target_size):
    """Create a side-by-side image for Pix2Pix training"""
    # Read images
    real_img = cv2.imread(real_path)
    artistic_img = cv2.imread(artistic_path)
    
    if real_img is None or artistic_img is None:
        print(f"Warning: Could not load images {real_path} or {artistic_path}")
        return None
    
    # Resize images
    real_img = cv2.resize(real_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    artistic_img = cv2.resize(artistic_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Create side-by-side image (real on left, artistic on right)
    combined_img = np.concatenate([real_img, artistic_img], axis=1)
    
    return combined_img

def process_paired_dataset(pairs, output_dir, target_size, val_ratio=0.1):
    """Process paired images into side-by-side format for Pix2Pix"""
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Shuffle and split pairs into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(pairs)
    
    val_size = max(1, int(len(pairs) * val_ratio))
    train_pairs = pairs[val_size:]
    val_pairs = pairs[:val_size]
    
    print(f"Processing {len(train_pairs)} training pairs...")
    for i, (real_path, artistic_path) in enumerate(tqdm(train_pairs)):
        # Create side-by-side image
        combined_img = create_side_by_side_image(real_path, artistic_path, target_size)
        
        if combined_img is not None:
            # Save combined image
            output_path = os.path.join(train_dir, f"pair_{i:04d}.png")
            cv2.imwrite(output_path, combined_img)
    
    print(f"Processing {len(val_pairs)} validation pairs...")
    for i, (real_path, artistic_path) in enumerate(tqdm(val_pairs)):
        # Create side-by-side image
        combined_img = create_side_by_side_image(real_path, artistic_path, target_size)
        
        if combined_img is not None:
            # Save combined image
            output_path = os.path.join(val_dir, f"pair_{i:04d}.png")
            cv2.imwrite(output_path, combined_img)
    
    return len(train_pairs), len(val_pairs)

def visualize_paired_samples(output_dir, num_samples=4):
    """Visualize sample processed paired images"""
    # Get sample images
    train_dir = os.path.join(output_dir, 'train')
    train_images = glob.glob(os.path.join(train_dir, '*.png'))
    
    if len(train_images) == 0:
        print("No processed images found for visualization")
        return
    
    # Limit number of samples
    num_samples = min(num_samples, len(train_images))
    
    # Select random samples
    random.seed(42)  # For reproducibility
    samples = random.sample(train_images, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 6, 6))
    
    # Handle the case where num_samples=1
    if num_samples == 1:
        axes = [axes]
    
    # Plot paired images
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Paired Sample {i+1}")
        axes[i].axis('off')
    
    # Add main title
    plt.suptitle("Sample Processed Paired Images (Left: Real, Right: Artistic)", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    vis_dir = os.path.join(os.path.dirname(output_dir), 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, 'sample_paired_images.png')
    plt.savefig(vis_path)
    plt.close()
    
    print(f"Sample visualization saved to {vis_path}")

def main():
    """Main preprocessing function for Pix2Pix GAN"""
    # Get current directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess paired cat images for Pix2Pix GAN')
    parser.add_argument('--real_dir', type=str, default='data/paired_real_cats',
                        help='Directory with real cat images')
    parser.add_argument('--artistic_dir', type=str, default='data/paired_artistic_cats',
                        help='Directory with artistic cat images')
    parser.add_argument('--output_dir', type=str, default='data/processed/pix2pix',
                        help='Directory to save processed images')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Target image size (square)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n--- Pix2Pix GAN Paired Image Preprocessing ---")
    print(f"Real cats directory: {args.real_dir}")
    print(f"Artistic cats directory: {args.artistic_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target image size: {args.target_size}")
    print(f"Validation ratio: {args.val_ratio}")
    
    # Start timing
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    real_files = get_image_files(args.real_dir)
    artistic_files = get_image_files(args.artistic_dir)
    
    print(f"Found {len(real_files)} real cat images")
    print(f"Found {len(artistic_files)} artistic cat images")
    
    # Find matching pairs
    pairs = find_matching_pairs(real_files, artistic_files)
    print(f"Found {len(pairs)} matching pairs")
    
    if len(pairs) == 0:
        print("No matching pairs found. Please check your file naming or directory structure.")
        return
    
    # Process paired dataset
    train_count, val_count = process_paired_dataset(
        pairs=pairs,
        output_dir=args.output_dir,
        target_size=args.target_size,
        val_ratio=args.val_ratio
    )
    
    # Visualize sample processed images
    visualize_paired_samples(args.output_dir, args.num_samples)
    
    # Report total time
    total_time = time.time() - start_time
    print(f"\nPreprocessing completed in {total_time:.1f} seconds")
    print(f"Created {train_count} training samples and {val_count} validation samples")
    print(f"Processed images are ready in: {args.output_dir}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Train the Pix2Pix model: python training/train_pix2pix.py")
    print("2. After training, visualize results with the Streamlit app")

if __name__ == "__main__":
    main()