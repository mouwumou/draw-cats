import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """Generator network for style transfer"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9, ngf=64):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False)
        ]
        
        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=False)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=False)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class UnetGenerator(nn.Module):
    """U-Net Generator for style transfer (alternative architecture)"""
    def __init__(self, input_channels=3, output_channels=3, ngf=64, dropout=0.5):
        super(UnetGenerator, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, ngf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.down8 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        
        # Decoder (upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=False)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=False)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=False)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=False)
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False)
        )
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))
        
        return u8


class MobileResGenerator(nn.Module):
    """Lighter Generator using depthwise separable convolutions"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=6, ngf=64):
        super(MobileResGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=False)
        ]
        
        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                # Depthwise separable convolution
                nn.Conv2d(in_features, in_features, 3, stride=2, padding=1, groups=in_features),
                nn.Conv2d(in_features, out_features, 1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=False)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Mobile residual blocks
        for _ in range(n_residual_blocks):
            model += [MobileResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=False)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class MobileResidualBlock(nn.Module):
    """Lightweight residual block with depthwise separable convolutions"""
    def __init__(self, in_features):
        super(MobileResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            # Depthwise
            nn.Conv2d(in_features, in_features, 3, groups=in_features),
            # Pointwise
            nn.Conv2d(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=False),
            
            nn.ReflectionPad2d(1),
            # Depthwise
            nn.Conv2d(in_features, in_features, 3, groups=in_features),
            # Pointwise
            nn.Conv2d(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)