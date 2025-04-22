import torch
import torch.nn as nn
import torch.nn.functional as F
from .generator import Generator
from .discriminator import Discriminator

class CycleGAN(nn.Module):
    """CycleGAN model for unpaired image-to-image translation"""
    
    def __init__(self, device='cuda', ngf=64, ndf=64, input_channels=3, output_channels=3):
        """
        Initialize CycleGAN model
        
        Parameters:
            device (str): Device to run the model on ('cuda' or 'cpu')
            ngf (int): Number of generator filters in first conv layer
            ndf (int): Number of discriminator filters in first conv layer
            input_channels (int): Number of input image channels
            output_channels (int): Number of output image channels
        """
        super(CycleGAN, self).__init__()
        
        self.device = device
        
        # Generators
        self.G_real_to_artistic = Generator(
            input_channels=input_channels, 
            output_channels=output_channels,
            ngf=ngf
        )
        
        self.G_artistic_to_real = Generator(
            input_channels=input_channels, 
            output_channels=output_channels,
            ngf=ngf
        )
        
        # Discriminators
        self.D_real = Discriminator(
            input_channels=input_channels,
            ndf=ndf
        )
        
        self.D_artistic = Discriminator(
            input_channels=input_channels,
            ndf=ndf
        )
        
        # Move to device
        self.G_real_to_artistic = self.G_real_to_artistic.to(device)
        self.G_artistic_to_real = self.G_artistic_to_real.to(device)
        self.D_real = self.D_real.to(device)
        self.D_artistic = self.D_artistic.to(device)
    
    def forward(self, real_img, artistic_img):
        """
        Forward pass through the CycleGAN model
        
        Parameters:
            real_img (Tensor): Batch of real cat images
            artistic_img (Tensor): Batch of artistic cat images
        
        Returns:
            dict: Dictionary containing all generated images and original inputs
        """
        # Forward G: Real -> Artistic
        fake_artistic = self.G_real_to_artistic(real_img)
        
        # Forward G: Artistic -> Real
        fake_real = self.G_artistic_to_real(artistic_img)
        
        # Cycle Real -> Artistic -> Real
        recovered_real = self.G_artistic_to_real(fake_artistic)
        
        # Cycle Artistic -> Real -> Artistic
        recovered_artistic = self.G_real_to_artistic(fake_real)
        
        return {
            'real': real_img,
            'artistic': artistic_img,
            'fake_artistic': fake_artistic,
            'fake_real': fake_real,
            'recovered_real': recovered_real,
            'recovered_artistic': recovered_artistic
        }
    
    def generate_artistic(self, real_img):
        """Generate artistic image from real image"""
        with torch.no_grad():
            return self.G_real_to_artistic(real_img)
    
    def generate_real(self, artistic_img):
        """Generate real image from artistic image"""
        with torch.no_grad():
            return self.G_artistic_to_real(artistic_img)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad for all networks"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_visuals(self, real_img, artistic_img):
        """Return visualization images"""
        with torch.no_grad():
            fake_artistic = self.G_real_to_artistic(real_img)
            fake_real = self.G_artistic_to_real(artistic_img)
            recovered_real = self.G_artistic_to_real(fake_artistic)
            recovered_artistic = self.G_real_to_artistic(fake_real)
            
            return {
                'real': real_img,
                'fake_artistic': fake_artistic,
                'recovered_real': recovered_real,
                'artistic': artistic_img,
                'fake_real': fake_real,
                'recovered_artistic': recovered_artistic
            }

    def get_discriminator_features(self, real_img, artistic_img):
        """Get intermediate features from discriminators"""
        with torch.no_grad():
            fake_artistic = self.G_real_to_artistic(real_img)
            fake_real = self.G_artistic_to_real(artistic_img)
            
            real_features = self.D_real.get_intermediate_features(real_img)
            artistic_features = self.D_artistic.get_intermediate_features(artistic_img)
            fake_real_features = self.D_real.get_intermediate_features(fake_real)
            fake_artistic_features = self.D_artistic.get_intermediate_features(fake_artistic)
            
            return {
                'real_features': real_features,
                'artistic_features': artistic_features,
                'fake_real_features': fake_real_features,
                'fake_artistic_features': fake_artistic_features
            }
            
    def save(self, save_path):
        """Save model checkpoint"""
        torch.save({
            'G_real_to_artistic': self.G_real_to_artistic.state_dict(),
            'G_artistic_to_real': self.G_artistic_to_real.state_dict(),
            'D_real': self.D_real.state_dict(),
            'D_artistic': self.D_artistic.state_dict()
        }, save_path)
        
    def load(self, load_path):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.G_real_to_artistic.load_state_dict(checkpoint['G_real_to_artistic'])
        self.G_artistic_to_real.load_state_dict(checkpoint['G_artistic_to_real'])
        self.D_real.load_state_dict(checkpoint['D_real'])
        self.D_artistic.load_state_dict(checkpoint['D_artistic'])


# Loss functions commonly used with CycleGAN
class GANLoss(nn.Module):
    """Define GAN loss functions with proper dimension handling"""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, device='cuda'):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.device = device
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, input, target_is_real):
        """Create target tensors of the same shape as input"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
            
        # Create a tensor of same size as input filled with the target value
        # This avoids the broadcasting warning
        target_tensor = target_tensor.expand_as(input)
        return target_tensor.to(self.device)
        
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class CycleGANLoss:
    """Collection of all losses used in CycleGAN"""
    def __init__(self, lambda_identity=5.0, lambda_cycle=10.0, lambda_gan=1.0, device='cuda'):
        self.device = device
        self.lambda_identity = lambda_identity
        self.lambda_cycle = lambda_cycle
        self.lambda_gan = lambda_gan
        
        # Initialize GAN loss
        self.criterionGAN = GANLoss(device=device)
        
        # Initialize L1 losses for cycle consistency and identity
        self.criterionCycle = nn.L1Loss()
        self.criterionIdentity = nn.L1Loss()
        
    def get_loss_G(self, model, real_A, real_B, fake_A, fake_B, recovered_A, recovered_B):
        """Calculate generator loss"""
        # Identity loss (optional)
        if self.lambda_identity > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            identity_A = model.G_artistic_to_real(real_A)
            loss_identity_A = self.criterionIdentity(identity_A, real_A) * self.lambda_identity
            
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            identity_B = model.G_real_to_artistic(real_B)
            loss_identity_B = self.criterionIdentity(identity_B, real_B) * self.lambda_identity
        else:
            loss_identity_A = 0
            loss_identity_B = 0
        
        # GAN loss for generators
        # D_A(G_A(B)) should be close to 1 (real)
        pred_fake_A = model.D_real(fake_A)
        loss_G_A = self.criterionGAN(pred_fake_A, True) * self.lambda_gan
        
        # D_B(G_B(A)) should be close to 1 (real) 
        pred_fake_B = model.D_artistic(fake_B)
        loss_G_B = self.criterionGAN(pred_fake_B, True) * self.lambda_gan
        
        # Cycle consistency loss
        # Forward cycle loss: ||G_A(G_B(A)) - A||
        loss_cycle_A = self.criterionCycle(recovered_A, real_A) * self.lambda_cycle
        
        # Backward cycle loss: ||G_B(G_A(B)) - B||
        loss_cycle_B = self.criterionCycle(recovered_B, real_B) * self.lambda_cycle
        
        # Total generator loss
        loss_G = loss_identity_A + loss_identity_B + loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        
        return loss_G, {
            'loss_G': loss_G.item(),
            'loss_G_A': loss_G_A.item(),
            'loss_G_B': loss_G_B.item(),
            'loss_cycle_A': loss_cycle_A.item(),
            'loss_cycle_B': loss_cycle_B.item(),
            'loss_identity_A': loss_identity_A if isinstance(loss_identity_A, float) else loss_identity_A.item(),
            'loss_identity_B': loss_identity_B if isinstance(loss_identity_B, float) else loss_identity_B.item()
        }
        
    def get_loss_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real loss
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake loss
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        return loss_D, loss_D_real, loss_D_fake
    
    def get_loss_D_A(self, model, real_A, fake_A):
        """Calculate GAN loss for discriminator D_A"""
        loss_D_A, loss_D_real_A, loss_D_fake_A = self.get_loss_D_basic(model.D_real, real_A, fake_A)
        return loss_D_A, {
            'loss_D_A': loss_D_A.item(),
            'loss_D_real_A': loss_D_real_A.item(),
            'loss_D_fake_A': loss_D_fake_A.item()
        }
    
    def get_loss_D_B(self, model, real_B, fake_B):
        """Calculate GAN loss for discriminator D_B"""
        loss_D_B, loss_D_real_B, loss_D_fake_B = self.get_loss_D_basic(model.D_artistic, real_B, fake_B)
        return loss_D_B, {
            'loss_D_B': loss_D_B.item(),
            'loss_D_real_B': loss_D_real_B.item(),
            'loss_D_fake_B': loss_D_fake_B.item()
        }