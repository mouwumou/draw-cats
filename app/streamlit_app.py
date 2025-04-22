import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules 
from model.gan import CycleGAN
from model.pix2pix import Pix2PixGAN

# Set page configuration
st.set_page_config(
    page_title="Cat Style Transfer App",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4527A0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #673AB7;
    }
    .result-container {
        padding: 1rem;
        background-color: #F3E5F5;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .image-caption {
        text-align: center;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .model-selection {
        border: 1px solid #E1BEE7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #F8F0FC;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>Learning to Draw My Cat: Style Transfer App</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Transform your cat photos into artistic drawings using GAN models</p>", unsafe_allow_html=True)

# Function to load CycleGAN model
@st.cache_resource
def load_cyclegan_model(model_path):
    """Load the CycleGAN generator model from a checkpoint"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the CycleGAN model
        model = CycleGAN(device=device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load state dict for the real-to-artistic generator
        model.G_real_to_artistic.load_state_dict(checkpoint["G_real_to_artistic_state_dict"])
        
        # Set to evaluation mode
        model.G_real_to_artistic.eval()
        
        return model.G_real_to_artistic, device, "CycleGAN"
    except Exception as e:
        st.error(f"Error loading CycleGAN model: {str(e)}")
        return None, None, None

# Function to load Pix2Pix model
@st.cache_resource
def load_pix2pix_model(model_path):
    """Load the Pix2Pix generator model from a checkpoint"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the Pix2Pix model
        model = Pix2PixGAN(device=device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load state dict for the generator
        model.G.load_state_dict(checkpoint["G_state_dict"])
        
        # Set to evaluation mode
        model.G.eval()
        
        return model.G, device, "Pix2Pix"
    except Exception as e:
        st.error(f"Error loading Pix2Pix model: {str(e)}")
        return None, None, None

# Function to preprocess image
def preprocess_image(image, target_size=256):
    """Preprocess an image for the generator"""
    # Resize the image while preserving aspect ratio
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a white background canvas
    background = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # Paste the image in the center of the canvas
    offset = ((target_size - new_width) // 2, (target_size - new_height) // 2)
    background.paste(image, offset)
    
    # Apply transformations needed for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(background).unsqueeze(0)
    return image_tensor

# Function to postprocess tensor to image
def postprocess_tensor(tensor):
    """Convert output tensor back to PIL image"""
    # Move to CPU, remove batch dimension, and denormalize
    tensor = tensor.squeeze(0).cpu().detach()
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert from torch tensor (C, H, W) to numpy array (H, W, C)
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    return Image.fromarray(tensor.astype(np.uint8))

# Function to get available models
def get_available_models():
    """Get lists of available trained models"""
    # Check for CycleGAN models
    cyclegan_models = glob.glob('checkpoints/cycle_gan/*.pth') + glob.glob('checkpoints/*.pth')
    # Filter out latest_model.pth as it's a duplicate
    cyclegan_models = [f for f in cyclegan_models if 'latest_model.pth' not in f]
    
    # Check for Pix2Pix models
    pix2pix_models = glob.glob('checkpoints/pix2pix/*.pth')
    # Filter out latest_model.pth as it's a duplicate
    pix2pix_models = [f for f in pix2pix_models if 'latest_model.pth' not in f]
    
    # Return both model types
    return {
        'cyclegan': sorted(cyclegan_models),
        'pix2pix': sorted(pix2pix_models)
    }

# Sidebar
st.sidebar.markdown("<h2 class='sub-header'>Model Settings</h2>", unsafe_allow_html=True)

# Check if models are available
available_models = get_available_models()

# Add model type selection
model_type = st.sidebar.radio(
    "Select Model Type",
    ["CycleGAN (Unpaired)", "Pix2Pix (Paired)"],
    help="Choose between CycleGAN (trained on unpaired data) or Pix2Pix (trained on paired data)"
)

# Demo mode flag
demo_mode = False

# Model selection based on type
if model_type == "CycleGAN (Unpaired)":
    if not available_models['cyclegan']:
        st.sidebar.warning("No CycleGAN models found. Please train a model first.")
        demo_mode = True
        st.sidebar.info("Running in demo mode.")
    else:
        # Model selection dropdown
        model_path = st.sidebar.selectbox(
            "Select CycleGAN Model",
            available_models['cyclegan'],
            format_func=lambda x: f"Model from epoch {os.path.basename(x).replace('model_epoch_', '').replace('.pth', '')}"
        )
        
        # Load selected model
        with st.sidebar.spinner("Loading CycleGAN model..."):
            model, device, model_name = load_cyclegan_model(model_path)
        
        if model:
            st.sidebar.success("CycleGAN model loaded successfully!")
        else:
            st.sidebar.error("Failed to load CycleGAN model.")
            demo_mode = True
else:  # Pix2Pix
    if not available_models['pix2pix']:
        st.sidebar.warning("No Pix2Pix models found. Please train a model first.")
        demo_mode = True
        st.sidebar.info("Running in demo mode.")
    else:
        # Model selection dropdown
        model_path = st.sidebar.selectbox(
            "Select Pix2Pix Model",
            available_models['pix2pix'],
            format_func=lambda x: f"Model from {os.path.basename(x).replace('model_epoch_', '').replace('.pth', '')}"
        )
        
        # Load selected model
        with st.sidebar.spinner("Loading Pix2Pix model..."):
            model, device, model_name = load_pix2pix_model(model_path)
        
        if model:
            st.sidebar.success("Pix2Pix model loaded successfully!")
        else:
            st.sidebar.error("Failed to load Pix2Pix model.")
            demo_mode = True

# Image settings
st.sidebar.markdown("<h2 class='sub-header'>Image Settings</h2>", unsafe_allow_html=True)

# Image size slider
image_size = st.sidebar.slider(
    "Output Image Size",
    min_value=128,
    max_value=512,
    value=256,
    step=64,
    help="Size of the output image in pixels"
)

# Upload image
uploaded_file = st.sidebar.file_uploader(
    "Upload Cat Photo",
    type=["jpg", "jpeg", "png"],
    help="Upload a photo of your cat"
)

# Example images option
st.sidebar.markdown("<h2 class='sub-header'>Or Use Example Image</h2>", unsafe_allow_html=True)

# Check for examples directory
example_dir = "data/examples"
if os.path.exists(example_dir):
    example_images = glob.glob(os.path.join(example_dir, "*.jpg")) + \
                     glob.glob(os.path.join(example_dir, "*.png"))
    
    if example_images:
        example_option = st.sidebar.selectbox(
            "Select Example Image",
            ["None"] + example_images,
            format_func=lambda x: "None" if x == "None" else os.path.basename(x)
        )
    else:
        example_option = "None"
        st.sidebar.info("No example images found in 'data/examples' directory.")
else:
    example_option = "None"
    st.sidebar.info("Example images directory not found.")

# Main content area
col1, col2 = st.columns(2)

# Function to run style transfer
def run_style_transfer(input_image, model_type="CycleGAN"):
    """Process the input image through the model to create a stylized output"""
    # Create spinner animation while processing
    with st.spinner("Transforming your cat into art... 🎨"):
        if demo_mode:
            # In demo mode, just add a delay and apply simple filters
            time.sleep(2)
            
            # Create a simple fake "artistic" version for demo
            img_array = np.array(input_image.resize((image_size, image_size)))
            
            # Apply simple edge detection and color quantization for demo
            from scipy.ndimage import gaussian_filter
            
            # Convert to grayscale for edge detection
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = gaussian_filter(gray, sigma=1)
            
            # Simple edge detection
            edges = np.abs(gray - blurred)
            edges = (edges > 10).astype(np.uint8) * 255
            
            # Create a stylized version by combining edges with reduced colors
            stylized = img_array.copy()
            
            # Reduce colors (simple quantization)
            for c in range(3):
                channel = stylized[:,:,c]
                channel = (channel // 32) * 32
                stylized[:,:,c] = channel
            
            # Add edges
            for c in range(3):
                stylized[:,:,c] = np.maximum(0, stylized[:,:,c] - edges * 0.7)
            
            return Image.fromarray(stylized.astype(np.uint8))
        else:
            # Real model processing
            # Preprocess image
            input_tensor = preprocess_image(input_image, target_size=image_size)
            input_tensor = input_tensor.to(device)
            
            # Process through model
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Convert back to image
            output_image = postprocess_tensor(output_tensor)
            
            return output_image

# Process and display image
if uploaded_file is not None:
    # User uploaded an image
    input_image = Image.open(uploaded_file).convert('RGB')
    selected_image = "upload"
elif example_option != "None":
    # User selected an example image
    input_image = Image.open(example_option).convert('RGB')
    selected_image = "example"
else:
    # No image selected
    input_image = None
    selected_image = None

# Display input image and generate stylized version
if input_image is not None:
    with col1:
        st.markdown("<h2 class='sub-header'>Original Cat Photo</h2>", unsafe_allow_html=True)
        st.image(input_image, use_column_width=True)
        
        # Add source information
        if selected_image == "upload":
            st.markdown("<p class='image-caption'>Uploaded image</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='image-caption'>Example: {os.path.basename(example_option)}</p>", unsafe_allow_html=True)
    
    # Generate button
    if st.sidebar.button("Generate Artistic Drawing", key="generate"):
        # Process the image with the selected model type
        stylized_image = run_style_transfer(input_image, model_type=model_type)
        
        # Display the stylized image
        with col2:
            st.markdown("<h2 class='sub-header'>Artistic Drawing</h2>", unsafe_allow_html=True)
            st.image(stylized_image, use_column_width=True)
            
            # Add model info
            model_info = "CycleGAN (Unpaired Data)" if model_type == "CycleGAN (Unpaired)" else "Pix2Pix (Paired Data)"
            st.markdown(f"<p class='image-caption'>Generated using {model_info}</p>", unsafe_allow_html=True)
            
            # Add download button
            buf = io.BytesIO()
            stylized_image.save(buf, format="PNG")
            btn = st.download_button(
                label="Download Drawing",
                data=buf.getvalue(),
                file_name="cat_drawing.png",
                mime="image/png"
            )
else:
    # Instructions when no image is selected
    st.markdown("""
    <div class="result-container">
        <h2 class='sub-header'>Instructions</h2>
        <p>Please upload a cat photo or select an example image to get started.</p>
        <p>The app offers two different GAN models for transforming your cat photos:</p>
        <ul>
            <li><strong>CycleGAN</strong>: Trained on unpaired data (separate collections of real cats and artistic drawings)</li>
            <li><strong>Pix2Pix</strong>: Trained on paired data (specific real cats with their corresponding artistic versions)</li>
        </ul>
        <p>For best results:</p>
        <ul>
            <li>Use photos with a clear view of the cat's face</li>
            <li>Photos with simple backgrounds work best</li>
            <li>Make sure the cat is well-lit and centered</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Add information about the project
with st.expander("About This Project"):
    st.markdown("""
    # Learning to Draw My Cat: Style Transfer with GANs
    
    This application uses two different Generative Adversarial Network (GAN) architectures to transform real cat photos into artistic drawings:
    
    ## 1. CycleGAN (Unpaired Training)
    
    - Trained on separate collections of real cat photos and artistic drawings
    - Does not require paired examples (more flexible)
    - Uses cycle consistency to maintain cat features
    - Good for creative, varied artistic styles
    
    ## 2. Pix2Pix (Paired Training)
    
    - Trained on paired examples (each real cat photo with its corresponding artistic drawing)
    - More precise control over the transformation style
    - Often produces more accurate results that match the specific artistic style
    - Requires carefully paired training data
    
    ## How They Work
    
    Both models use convolutional neural networks to learn the mapping between photograph domain and artistic domain. The models were trained on custom datasets of cat photos and artistic drawings.
    
    ## Technical Details
    
    - **Architectures**: CycleGAN with residual blocks, Pix2Pix with U-Net generator
    - **Image Size**: 256×256 pixels (default)
    - **Training**: PyTorch implementation
    - **Interface**: Streamlit web app
    
    ## Credits
    
    Created as part of the final project for deep learning class.
    """)

# Add custom footer
st.markdown("""
<div style="text-align:center; margin-top:2rem;">
    <p style="color:#9E9E9E; font-size:0.8rem;">Cat Style Transfer App • Made with PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)