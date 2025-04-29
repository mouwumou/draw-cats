import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """
    Spatial Transformer Network 前端：
      - localization net 预测仿射参数 θ
      - grid_sample 对输入做仿射变换对齐
    """
    def __init__(self, in_channels=3, pool_out: int = 8):
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
        # # 全连接回归仿射参数 6 个数
        # # 假设输入 256x256 -> localization 输出 (10, 64, 64) -> flatten 10*64*64
        # self.fc_loc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(10 * 64 * 64, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 6)
        # )

        # (2) 自适应池化到固定大小，确保任何输入分辨率都得到相同维度
        self.pool = nn.AdaptiveAvgPool2d((pool_out, pool_out))  # 默认 8×8

        feat_dim = 10 * pool_out * pool_out  # 10×8×8 = 640

        # (3) 两层全连接预测 θ
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        # 初始化成单位仿射：θ = [1,0,0;0,1,0]
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # x: (B, C, H, W)
        xs = self.localization(x)
        xs = self.pool(xs)                # (B,10,8,8)
        theta = self.fc_loc(xs)         # (B, 6)
        theta = theta.view(-1, 2, 3)    # (B, 2, 3)
        # 生成采样网格
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        # 对输入做仿射变换
        x_trans = F.grid_sample(x, grid, align_corners=True, padding_mode='border')
        return x_trans