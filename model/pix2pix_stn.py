import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

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


class UNetWithSTN(nn.Module):
    """
    在输入前端加 STN 的 U-Net 生成器。
    """
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        # 1) 前端 STN
        self.stn = STN(in_channels)

        # 2) 下采样模块
        def down(ch_in, ch_out):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)),
                nn.InstanceNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 3) 上采样模块
        def up(ch_in, ch_out, dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False)),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(True)
            ]
            if dropout:
                layers.insert(-1, nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # 编码器：4 层下采样
        self.enc1 = down(in_channels, base_filters)           # 128
        self.enc2 = down(base_filters, base_filters*2)        # 64
        self.enc3 = down(base_filters*2, base_filters*4)      # 32
        self.enc4 = down(base_filters*4, base_filters*8)      # 16

        # 瓶颈
        self.bottleneck = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters*8, base_filters*8, 4, 2, 1, bias=False)),  # 8
            nn.ReLU(True)
        )

        # 解码器：4 层上采样
        self.dec1 = up(base_filters*8, base_filters*8, dropout=True)   # 16
        self.dec2 = up(base_filters*16, base_filters*4, dropout=True)  # 32
        self.dec3 = up(base_filters*8, base_filters*2, dropout=True)   # 64
        self.dec4 = up(base_filters*4, base_filters)                   # 128

        # 最后一层，恢复到原始分辨率
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 256
            spectral_norm(nn.Conv2d(base_filters*2, out_channels, 3, 1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. STN 对齐
        x = self.stn(x)

        # 2. 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 3. 瓶颈
        b = self.bottleneck(e4)

        # 4. 解码 + 跳跃
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e1], dim=1)

        # 5. 输出
        return self.final(d4)