import torch
import torch.nn as nn
import torch.nn.functional as F
from .STN import STN

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed.
    Uses Conv2d for patch extraction.
    """
    def __init__(self, in_ch=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        # self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_ch, embed_dim,
                            kernel_size=patch_size + 1,   # 17
                            stride=patch_size,            # 16
                            padding=patch_size // 2)      # 8

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, E], N=H*W
        return x, (H, W)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(ch)
        )
    def forward(self, x):
        return x + self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
            ResidualBlock(out_ch)          # 1×ResBlock 增细节
        )
    def forward(self, x): return self.up(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [N, B, E]
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class ViTGenerator(nn.Module):
    """
    Vision Transformer based Generator with enhanced decoder.
    - Patch embed -> Transformer encoder -> deeper convolutional decoder
    """
    def __init__(self,
                 in_ch=3,
                 out_ch=3,
                 img_size=256,
                 patch_size=16,
                 embed_dim=512,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.stn = STN(in_ch)
        self.patch_embed = PatchEmbedding(in_ch, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.up1 = UpBlock(embed_dim,      embed_dim // 2)   # 16→32
        self.up2 = UpBlock(embed_dim // 2, embed_dim // 4)   # 32→64
        self.up3 = UpBlock(embed_dim // 4, embed_dim // 8)   # 64→128
        self.up4 = UpBlock(embed_dim // 8, embed_dim // 16)  # 128→256

        self.to_rgb = nn.Sequential(
            nn.Conv2d(embed_dim // 16, out_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Patch embedding
        x = self.stn(x)                     # [B, C, H, W]
        x, (H, W) = self.patch_embed(x)      # [B, N, E]
        x = x + self.pos_embed
        x = self.dropout(x)
        # Transformer encoder
        x = x.transpose(0, 1)                # [N, B, E]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(0, 1)                # [B, N, E]
        # Unflatten and decode
        B, N, E = x.shape
        x = x.transpose(1, 2).view(B, E, H, W)  # (B, E, 16, 16)

        x = self.up1(x)   # 32×32
        x = self.up2(x)   # 64×64
        x = self.up3(x)   # 128×128
        x = self.up4(x)   # 256×256
        x = self.to_rgb(x)
        return x

