import os
import argparse
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
import shutil
import random

def create_directories(base_dir):
    """Create necessary directories for processed images"""
    processed_dir = os.path.join(base_dir, 'processed')
    real_processed_dir = os.path.join(processed_dir, 'real')
    artistic_processed_dir = os.path.join(processed_dir, 'artistic')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(real_processed_dir, exist_ok=True)
    os.makedirs(artistic_processed_dir, exist_ok=True)
    
    return real_processed_dir, artistic_processed_dir

def preprocess_centralized_image(image_path, target_size=256):
    """Preprocess a single centralized image"""
    # Read image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Resize while maintaining aspect ratio
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a square canvas and center the resized image
    square_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255  # White background
    offset_h = (target_size - new_h) // 2
    offset_w = (target_size - new_w) // 2
    square_img[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = img
    
    return square_img

def process_all_images(src_dir, dest_dir, target_size=256):
    """Process all images in a directory and save to destination"""
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in glob.glob(os.path.join(src_dir, '*.*')) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Processing {len(image_files)} images from {src_dir} to {dest_dir}...")
    
    for idx, img_path in enumerate(tqdm(image_files)):
        # Generate output filename
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(dest_dir, f"{base_name}.png")
        
        # Process image
        processed_img = preprocess_centralized_image(img_path, target_size)
        
        if processed_img is not None:
            # Save processed image
            cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
    
    return len(image_files)

def create_validation_split(real_processed_dir, artistic_processed_dir, val_ratio=0.1):
    """Create a validation split from the processed images"""
    print(f"Creating validation split with ratio {val_ratio}...")
    
    # Create validation directories
    real_val_dir = os.path.join(os.path.dirname(real_processed_dir), 'real_val')
    artistic_val_dir = os.path.join(os.path.dirname(artistic_processed_dir), 'artistic_val')
    
    os.makedirs(real_val_dir, exist_ok=True)
    os.makedirs(artistic_val_dir, exist_ok=True)
    
    # Get all processed images
    real_images = glob.glob(os.path.join(real_processed_dir, '*.png'))
    artistic_images = glob.glob(os.path.join(artistic_processed_dir, '*.png'))
    
    # Random selection for validation
    num_real_val = max(1, int(len(real_images) * val_ratio))
    num_artistic_val = max(1, int(len(artistic_images) * val_ratio))
    
    real_val_indices = random.sample(range(len(real_images)), num_real_val)
    artistic_val_indices = random.sample(range(len(artistic_images)), num_artistic_val)
    
    # Move selected images to validation sets
    for idx in real_val_indices:
        if idx < len(real_images):
            real_path = real_images[idx]
            base_name = os.path.basename(real_path)
            shutil.copy(real_path, os.path.join(real_val_dir, base_name))
    
    for idx in artistic_val_indices:
        if idx < len(artistic_images):
            artistic_path = artistic_images[idx]
            base_name = os.path.basename(artistic_path)
            shutil.copy(artistic_path, os.path.join(artistic_val_dir, base_name))
    
    print(f"Moved {num_real_val} real images and {num_artistic_val} artistic images to validation split")

def analyze_dataset(data_dir):
    """Print dataset statistics after preprocessing"""
    real_train_dir = os.path.join(data_dir, 'processed', 'real')
    artistic_train_dir = os.path.join(data_dir, 'processed', 'artistic')
    real_val_dir = os.path.join(data_dir, 'processed', 'real_val')
    artistic_val_dir = os.path.join(data_dir, 'processed', 'artistic_val')
    
    real_train_count = len(glob.glob(os.path.join(real_train_dir, '*.png')))
    artistic_train_count = len(glob.glob(os.path.join(artistic_train_dir, '*.png')))
    real_val_count = len(glob.glob(os.path.join(real_val_dir, '*.png')))
    artistic_val_count = len(glob.glob(os.path.join(artistic_val_dir, '*.png')))
    
    print("\nDataset Statistics:")
    print(f"Training set - Real: {real_train_count}, Artistic: {artistic_train_count}")
    print(f"Validation set - Real: {real_val_count}, Artistic: {artistic_val_count}")

def visualize_samples(data_dir, num_samples=5):
    """Visualize sample processed images"""
    real_dir = os.path.join(data_dir, 'processed', 'real')
    artistic_dir = os.path.join(data_dir, 'processed', 'artistic')
    
    real_images = glob.glob(os.path.join(real_dir, '*.png'))
    artistic_images = glob.glob(os.path.join(artistic_dir, '*.png'))
    
    if len(real_images) == 0 or len(artistic_images) == 0:
        print("No processed images found to visualize.")
        return
    
    # Randomly select samples
    num_samples = min(num_samples, len(real_images), len(artistic_images))
    real_samples = random.sample(real_images, num_samples)
    artistic_samples = random.sample(artistic_images, num_samples)
    
    # Create visualization directory
    viz_dir = os.path.join(data_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a composite image
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i in range(num_samples):
        # Load real image
        real_img = cv2.imread(real_samples[i])
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        
        # Load artistic image
        artistic_img = cv2.imread(artistic_samples[i])
        artistic_img = cv2.cvtColor(artistic_img, cv2.COLOR_BGR2RGB)
        
        # Display images
        axes[0, i].imshow(real_img)
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(artistic_img)
        axes[1, i].set_title('Artistic')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_processed_images.png'))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(viz_dir, 'sample_processed_images.png')}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess cat images for style transfer')
    parser.add_argument('--real_dir', type=str, default='data/real_cats', 
                        help='Directory with real cat images')
    parser.add_argument('--artistic_dir', type=str, default='data/artistic_cats', 
                        help='Directory with artistic cat images')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Base data directory')
    parser.add_argument('--target_size', type=int, default=256, 
                        help='Target image size (square)')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                        help='Validation set ratio')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize sample processed images')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create directories
    real_processed_dir, artistic_processed_dir = create_directories(args.data_dir)
    
    # Process all images
    real_count = process_all_images(args.real_dir, real_processed_dir, args.target_size)
    artistic_count = process_all_images(args.artistic_dir, artistic_processed_dir, args.target_size)
    
    print(f"Processed {real_count} real images and {artistic_count} artistic images")
    
    # Create validation split
    create_validation_split(real_processed_dir, artistic_processed_dir, args.val_ratio)
    
    # Analyze final dataset
    analyze_dataset(args.data_dir)
    
    # Visualize samples if requested
    if args.visualize:
        visualize_samples(args.data_dir, args.num_samples)
    
    print("\nPreprocessing complete! Dataset is ready for training.")

if __name__ == "__main__":
    main()