import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """PatchGAN discriminator for style transfer"""
    
    def __init__(self, input_channels=3, ndf=64, n_layers=3):
        """
        Parameters:
            input_channels (int): Number of input channels
            ndf (int): Number of filters in the first conv layer
            n_layers (int): Number of conv layers in the discriminator
        """
        super(Discriminator, self).__init__()
        
        # Initial layer
        model = [
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        
        # Intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        
        # Last layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        
        # Output layer
        model += [nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """Pixel-level discriminator - simpler and faster than PatchGAN"""
    
    def __init__(self, input_channels=3, ndf=64):
        super(PixelDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, ndf, 1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, 1, stride=1, padding=0),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 2, 1, 1, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """Multiscale Discriminator for more stable training"""
    
    def __init__(self, input_channels=3, ndf=64, n_layers=3, num_D=3):
        """
        Parameters:
            input_channels (int): Number of input channels
            ndf (int): Number of filters in the first conv layer
            n_layers (int): Number of conv layers in each discriminator
            num_D (int): Number of discriminators at different scales
        """
        super(MultiscaleDiscriminator, self).__init__()
        
        self.num_D = num_D
        self.n_layers = n_layers
        
        # Create discriminators at different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(self._make_net(input_channels, ndf, n_layers))
        
        # Downsample module for creating different scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def _make_net(self, input_channels, ndf, n_layers):
        """Create a single discriminator network"""
        model = [
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        
        # Add intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        
        # Add output layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)
        ]
        
        return nn.Sequential(*model)
    
    def forward(self, x):
        """Forward pass through all discriminators at different scales"""
        results = []
        input_downsampled = x
        
        for i in range(self.num_D):
            results.append(self.discriminators[i](input_downsampled))
            if i != self.num_D - 1:  # Don't downsample the last input
                input_downsampled = self.downsample(input_downsampled)
        
        return results


class NLayerDiscriminator(nn.Module):
    """N-layer Discriminator with spectral normalization for better training stability"""
    
    def __init__(self, input_channels=3, ndf=64, n_layers=3, use_spectral_norm=True):
        """
        Parameters:
            input_channels (int): Number of input channels
            ndf (int): Number of filters in the first conv layer
            n_layers (int): Number of conv layers
            use_spectral_norm (bool): Whether to use spectral normalization
        """
        super(NLayerDiscriminator, self).__init__()
        
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Initial layer
        model = [
            norm_layer(nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        
        # Last layers
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            norm_layer(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False),
            norm_layer(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1))
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)