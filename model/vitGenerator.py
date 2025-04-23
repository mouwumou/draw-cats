import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed.
    Uses Conv2d for patch extraction.
    """
    def __init__(self, in_ch=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, E], N=H*W
        return x, (H, W)

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
    Vision Transformer based Generator:
    - Patch embed -> Transformer encoder -> patch unembed -> conv decoder
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
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_embed = PatchEmbedding(in_ch, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Decoder: project back to patch pixels via convtranspose
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim,
                               embed_dim//2,
                               kernel_size=patch_size,
                               stride=patch_size),
            nn.GroupNorm(1, embed_dim//2),
            nn.ReLU(True),
            nn.Conv2d(embed_dim//2, out_ch, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Patch embedding
        x, (H, W) = self.patch_embed(x)  # [B, N, E]
        x = x + self.pos_embed  # [B, N, E]
        x = self.dropout(x)
        # Transformer expects [N, B, E]
        x = x.transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [N, B, E]
        x = x.transpose(0, 1)  # [B, N, E]
        # Reshape to [B, E, H, W]
        B, N, E = x.shape
        x = x.transpose(1, 2).view(B, E, H, W)
        # Decode to image
        x = self.decoder(x)
        return x
