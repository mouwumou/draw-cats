import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    """
    Spatial Transformer Network 前端：
      - localization net 预测仿射参数 θ
      - grid_sample 对输入做仿射变换对齐
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # localization 网络：简单卷积 + 池化
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 全连接回归仿射参数 6 个数
        # 假设输入 256x256 -> localization 输出 (10, 64, 64) -> flatten 10*64*64
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 64 * 64, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        # 初始化成单位仿射：θ = [1,0,0;0,1,0]
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # x: (B, C, H, W)
        xs = self.localization(x)
        theta = self.fc_loc(xs)         # (B, 6)
        theta = theta.view(-1, 2, 3)    # (B, 2, 3)
        # 生成采样网格
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        # 对输入做仿射变换
        x_trans = F.grid_sample(x, grid, align_corners=True, padding_mode='border')
        return x_trans



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
        # Enhanced decoder
        self.decoder = nn.Sequential(
            # Unflatten to spatial
            nn.ConvTranspose2d(embed_dim, embed_dim // 2,
                               kernel_size=patch_size,
                               stride=patch_size),  # [B, E/2, H, W]
            nn.GroupNorm(1, embed_dim // 2),
            nn.ReLU(True),
            # Additional conv layers for richer decoding
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, embed_dim // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, embed_dim // 4),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 4, out_ch, kernel_size=3, padding=1),
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
        x = x.transpose(1, 2).view(B, E, H, W)
        x = self.decoder(x)
        return x

