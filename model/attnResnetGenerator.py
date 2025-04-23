import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,      kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # B, N, C'
        proj_key   = self.key_conv(x).view(B, -1, H*W)                    # B, C', N
        energy     = torch.bmm(proj_query, proj_key)                      # B, N, N
        attention  = self.softmax(energy)                                 # B, N, N
        proj_value = self.value_conv(x).view(B, -1, H*W)                  # B, C, N
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))    # B, C, N
        out        = out.view(B, C, H, W)
        return self.gamma * out + x


class ConvAttnBlock(nn.Module):
    """去掉残差跳跃的卷积 + 注意力块"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            SelfAttention(dim)
        )
    def forward(self, x):
        return self.net(x)
    
    
class AttnResNetGenerator(nn.Module):
    """
    Attention-augmented ResNet Generator without U-Net skips.
    Encoder -> Residual Attention Blocks -> Decoder
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        # 初始卷积
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        # 下采样两次
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.ReLU(True),
                SelfAttention(ngf*mult*2)
            ]
            mult *= 2

        # 更激进的“残差‐注意力”块，数量从 6 → 2
        dim = ngf * mult
        for _ in range(2):
            layers.append(ConvAttnBlock(dim))

        # 上采样两次
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(dim, dim//2, 3, padding=1, bias=False),
                nn.InstanceNorm2d(dim//2),
                nn.ReLU(True),
                SelfAttention(dim//2)
            ]
            dim //= 2

        # 输出层
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_ch, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)