import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# class UNetGenerator(nn.Module):
#     def __init__(self, input_channels=3, output_channels=3, ngf=64, use_dropout=True):
#         super(UNetGenerator, self).__init__()
#         self.enc1 = nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1)
#         self.enc2 = self._encoder_block(ngf, ngf * 2)
#         self.enc3 = self._encoder_block(ngf * 2, ngf * 4)
#         self.enc4 = self._encoder_block(ngf * 4, ngf * 8)
#         self.enc5 = self._encoder_block(ngf * 8, ngf * 8)
#         self.enc6 = self._encoder_block(ngf * 8, ngf * 8)
#         self.enc7 = self._encoder_block(ngf * 8, ngf * 8)
#         self.enc8 = nn.Sequential(
#             nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=False)
#         )

#         self.dec1 = self._decoder_block(ngf * 8, ngf * 8, use_dropout=True)
#         self.dec2 = self._decoder_block(ngf * 8 * 2, ngf * 8, use_dropout=True)
#         self.dec3 = self._decoder_block(ngf * 8 * 2, ngf * 8, use_dropout=True)
#         self.dec4 = self._decoder_block(ngf * 8 * 2, ngf * 8)
#         self.dec5 = self._decoder_block(ngf * 8 * 2, ngf * 4)
#         self.dec6 = self._decoder_block(ngf * 4 * 2, ngf * 2)
#         self.dec7 = self._decoder_block(ngf * 2 * 2, ngf)
#         self.dec8 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 2, output_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def _encoder_block(self, in_c, out_c):
#         return nn.Sequential(
#             nn.LeakyReLU(0.2, inplace=False),
#             nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_c)
#         )

#     def _decoder_block(self, in_c, out_c, use_dropout=False):
#         layers = [
#             nn.ReLU(inplace=False),
#             nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_c)
#         ]
#         if use_dropout:
#             layers.append(nn.Dropout(0.5))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1.clone())
#         e3 = self.enc3(e2.clone())
#         e4 = self.enc4(e3.clone())
#         e5 = self.enc5(e4.clone())
#         e6 = self.enc6(e5.clone())
#         e7 = self.enc7(e6.clone())
#         e8 = self.enc8(e7.clone())

#         d1 = self.dec1(e8)
#         d1 = torch.cat([d1, e7.clone()], 1)
#         d2 = self.dec2(d1)
#         d2 = torch.cat([d2, e6.clone()], 1)
#         d3 = self.dec3(d2)
#         d3 = torch.cat([d3, e5.clone()], 1)
#         d4 = self.dec4(d3)
#         d4 = torch.cat([d4, e4.clone()], 1)
#         d5 = self.dec5(d4)
#         d5 = torch.cat([d5, e3.clone()], 1)
#         d6 = self.dec6(d5)
#         d6 = torch.cat([d6, e2.clone()], 1)
#         d7 = self.dec7(d6)
#         d7 = torch.cat([d7, e1.clone()], 1)
#         return self.dec8(d7)


# class UNetGenerator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_filters=64):
#         super().__init__()
#         # 下采样：Conv + InstanceNorm + ReLU
#         def down_block(ch_in, ch_out):
#             return nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
#                 nn.InstanceNorm2d(ch_out),
#                 nn.LeakyReLU(0.2, inplace=True)
#             )

#         # 上采样：Upsample + Conv + InstanceNorm + ReLU
#         def up_block(ch_in, ch_out, use_dropout=False):
#             layers = [
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.InstanceNorm2d(ch_out),
#                 nn.ReLU(True)
#             ]
#             if use_dropout:
#                 layers.insert(-1, nn.Dropout(0.5))
#             return nn.Sequential(*layers)

#         # 编码器
#         self.enc1 = down_block(in_channels,  base_filters)       # 128
#         self.enc2 = down_block(base_filters, base_filters*2)     # 64
#         self.enc3 = down_block(base_filters*2, base_filters*4)   # 32
#         self.enc4 = down_block(base_filters*4, base_filters*8)   # 16
#         self.enc5 = down_block(base_filters*8, base_filters*8)   # 8
#         self.enc6 = down_block(base_filters*8, base_filters*8)   # 4
#         self.enc7 = down_block(base_filters*8, base_filters*8)   # 2
#         # self.enc8 = down_block(base_filters*8, base_filters*8)   # 1
#         self.enc8 = nn.Conv2d(
#             base_filters*8, base_filters*8,
#             kernel_size=4, stride=2, padding=1, bias=False
#         )

#         # 解码器
#         self.dec1 = up_block(base_filters*8, base_filters*8, use_dropout=True)
#         self.dec2 = up_block(base_filters*16, base_filters*8, use_dropout=True)
#         self.dec3 = up_block(base_filters*16, base_filters*8, use_dropout=True)
#         self.dec4 = up_block(base_filters*16, base_filters*8)
#         self.dec5 = up_block(base_filters*16, base_filters*4)
#         self.dec6 = up_block(base_filters*8,  base_filters*2)
#         self.dec7 = up_block(base_filters*4,  base_filters)
#         # 最后一层上采样后直接输出
#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(base_filters*2, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         # 编码
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)
#         e6 = self.enc6(e5)
#         e7 = self.enc7(e6)
#         e8 = self.enc8(e7)

#         r8 = e8

#         # 解码 + 跳跃连接
#         d1 = self.dec1(r8)
#         d1 = torch.cat([d1, e7], dim=1)
#         d2 = self.dec2(d1)
#         d2 = torch.cat([d2, e6], dim=1)
#         d3 = self.dec3(d2)
#         d3 = torch.cat([d3, e5], dim=1)
#         d4 = self.dec4(d3)
#         d4 = torch.cat([d4, e4], dim=1)
#         d5 = self.dec5(d4)
#         d5 = torch.cat([d5, e3], dim=1)
#         d6 = self.dec6(d5)
#         d6 = torch.cat([d6, e2], dim=1)
#         d7 = self.dec7(d6)
#         d7 = torch.cat([d7, e1], dim=1)

#         return self.final(d7)

class UNetGenerator(nn.Module):
    """
    U-Net generator with 4 downsampling and 4 upsampling layers.
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        # Downsample blocks: Conv -> Norm -> LeakyReLU
        def down(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True)
            )
        # Upsample blocks: Upsample -> Conv -> Norm -> ReLU
        def up(ch_in, ch_out, use_dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(True)
            ]
            if use_dropout:
                layers.insert(-1, nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = down(in_channels, base_filters)         # 128x128
        self.enc2 = down(base_filters, base_filters*2)     # 64x64
        self.enc3 = down(base_filters*2, base_filters*4)   # 32x32
        self.enc4 = down(base_filters*4, base_filters*8)   # 16x16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*8, base_filters*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.ReLU(True)
        )

        # Decoder
        self.dec1 = up(base_filters*8, base_filters*8, use_dropout=True)         # 16x16
        self.dec2 = up(base_filters*16, base_filters*4, use_dropout=True)        # 32x32
        self.dec3 = up(base_filters*8, base_filters*2, use_dropout=True)         # 64x64
        self.dec4 = up(base_filters*4, base_filters)                             # 128x128
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),     # 256x256
            nn.Conv2d(base_filters*2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        # Bottleneck
        b = self.bottleneck(e4)
        # Decode with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        return self.final(d4)
    

# class PatchGANDiscriminator(nn.Module):
#     def __init__(self, input_channels=6, ndf=64, n_layers=3):
#         super(PatchGANDiscriminator, self).__init__()
#         model = [
#             nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             model += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
#                 nn.BatchNorm2d(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, inplace=False)
#             ]
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         model += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1),
#             nn.BatchNorm2d(ndf * nf_mult),
#             nn.LeakyReLU(0.2, inplace=False)
#         ]
#         model += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
#         self.model = nn.Sequential(*model)

#     def forward(self, input_image, target_image):
#         x = torch.cat([input_image, target_image], 1)
#         return self.model(x)

class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator with spectral normalization.
    """
    def __init__(self, in_channels=6, base_filters=64, n_layers=4):
        super().__init__()
        layers = []
        # First layer (no normalization)
        layers += [
            spectral_norm(nn.Conv2d(in_channels, base_filters, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        # Subsequent downsampling layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                spectral_norm(nn.Conv2d(base_filters*nf_mult_prev, base_filters*nf_mult, 4, 2, 1, bias=False)),
                nn.InstanceNorm2d(base_filters*nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # Final output layer
        layers += [
            spectral_norm(nn.Conv2d(base_filters*nf_mult, 1, 4, 1, 1))
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input_image, target_image):
        # Concatenate condition and image, then classify
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.net(x)

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=8):
        super().__init__()
        # 输入卷积
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_ch, ngf, 7, bias=False),
                  nn.InstanceNorm2d(ngf),
                  nn.ReLU(True)]
        # 下采样 2×
        for mult in [1,2]:
            layers += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.ReLU(True)
            ]
        # 残差块
        dim = ngf*4
        for _ in range(n_blocks):
            layers.append(ResBlock(dim))
        # 上采样 2×
        for mult in [2,1]:
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(dim, dim//2, 3, padding=1, bias=False),
                nn.InstanceNorm2d(dim//2),
                nn.ReLU(True)
            ]
            dim //= 2
        # 输出
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, out_ch, 7),
                   nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def weights_init_normal(m):
    """Initialize network weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

