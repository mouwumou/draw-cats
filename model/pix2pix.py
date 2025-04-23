import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features.to(device)
        # take the first 21 layers for feature extraction
        self.slice = nn.Sequential(*[vgg19[i] for i in range(21)])
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, x):
        # VGG takes 3 channels [0,1] input, we assume the model output is also normalized to [-1,1] and then mapped back to [0,1]
        return self.slice(x)


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


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        # 下采样：Conv + InstanceNorm + ReLU
        def down_block(ch_in, ch_out):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # 上采样：Upsample + Conv + InstanceNorm + ReLU
        def up_block(ch_in, ch_out, use_dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(ch_out),
                nn.ReLU(True)
            ]
            if use_dropout:
                layers.insert(-1, nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # 编码器
        self.enc1 = down_block(in_channels,  base_filters)       # 128
        self.enc2 = down_block(base_filters, base_filters*2)     # 64
        self.enc3 = down_block(base_filters*2, base_filters*4)   # 32
        self.enc4 = down_block(base_filters*4, base_filters*8)   # 16
        self.enc5 = down_block(base_filters*8, base_filters*8)   # 8
        self.enc6 = down_block(base_filters*8, base_filters*8)   # 4
        self.enc7 = down_block(base_filters*8, base_filters*8)   # 2
        # self.enc8 = down_block(base_filters*8, base_filters*8)   # 1
        self.enc8 = nn.Conv2d(
            base_filters*8, base_filters*8,
            kernel_size=4, stride=2, padding=1, bias=False
        )

        # 解码器
        self.dec1 = up_block(base_filters*8, base_filters*8, use_dropout=True)
        self.dec2 = up_block(base_filters*16, base_filters*8, use_dropout=True)
        self.dec3 = up_block(base_filters*16, base_filters*8, use_dropout=True)
        self.dec4 = up_block(base_filters*16, base_filters*8)
        self.dec5 = up_block(base_filters*16, base_filters*4)
        self.dec6 = up_block(base_filters*8,  base_filters*2)
        self.dec7 = up_block(base_filters*4,  base_filters)
        # 最后一层上采样后直接输出
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_filters*2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        r8 = e8

        # 解码 + 跳跃连接
        d1 = self.dec1(r8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        return self.final(d7)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=6, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        model = [
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False)
        ]
        model += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], 1)
        return self.model(x)


class Pix2PixGAN(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, ngf=64, ndf=64, device='cuda'):
        super(Pix2PixGAN, self).__init__()
        self.device = device
        self.last_input = None
        self.G = UNetGenerator(input_channels, output_channels, ngf).to(device)
        self.D = PatchGANDiscriminator(input_channels + output_channels, ndf).to(device)

    def forward(self, input_image):
        return self.G(input_image)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
class GANLoss(nn.Module):
    """Define GAN loss with proper dimension handling"""

    def __init__(self, target_real_label=1.0, target_fake_label=0.0, device='cuda'):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.device = device
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction).to(self.device)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class Pix2PixLoss:
    """Collection of all losses used in Pix2Pix"""
    
    def __init__(self, lambda_l1=100.0, device='cuda'):
        self.lambda_l1 = lambda_l1
        self.device = device
        self.criterionGAN = GANLoss(device=device)
        self.criterionL1 = nn.L1Loss()
        # --- add perceptual loss ---
        self.lambda_perc = 1.0            
        self.perc_extractor = VGGFeatureExtractor(device=device).to(device)
        self.criterionPerc = nn.L1Loss()

    def get_generator_loss(self, fake_image, real_image, fake_pred, input_image=None, model=None):
        if input_image is None and model is not None:
            input_image = model.last_input  # fallback if not passed directly

        if input_image is not None and input_image.shape[2:] != fake_image.shape[2:]:
            input_image = F.interpolate(input_image, size=fake_image.shape[2:], mode='bilinear', align_corners=False)

        loss_G_GAN = self.criterionGAN(fake_pred, True)
        loss_G_L1 = self.criterionL1(fake_image, real_image) * self.lambda_l1
        # loss_G = loss_G_GAN + loss_G_L1
        
        # --- 新增感知损失 ---
        # 1) 将假图与真图从 [-1,1] 还原到 [0,1]
        real_norm = (real_image + 1) / 2
        fake_norm = (fake_image + 1) / 2
        # 2) 提取 VGG 特征
        feat_real = self.perc_extractor(real_norm)
        feat_fake = self.perc_extractor(fake_norm)
        # 3) 计算感知损失
        loss_perc = self.criterionPerc(feat_fake, feat_real) * self.lambda_perc
        # 总生成器损失
        loss_G = loss_G_GAN + loss_G_L1 + loss_perc

        # return loss_G, {
        #     'loss_G': loss_G.item(),
        #     'loss_G_GAN': loss_G_GAN.item(),
        #     'loss_G_L1': loss_G_L1.item()
        # }
        return loss_G, {
            'loss_G':        loss_G.item(),
            'loss_G_GAN':    loss_G_GAN.item(),
            'loss_G_L1':     loss_G_L1.item(),
            'loss_perc':     loss_perc.item()
        }

    def get_discriminator_loss(self, real_image, fake_image, input_image, model):
        """Calculate discriminator loss"""
        real_pred = model.D(input_image, real_image)
        loss_D_real = self.criterionGAN(real_pred, True)

        # Detach fake image to avoid gradient conflict
        fake_pred = model.D(input_image, fake_image.detach())
        loss_D_fake = self.criterionGAN(fake_pred, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D, fake_pred, {
            'loss_D': loss_D.item(),
            'loss_D_real': loss_D_real.item(),
            'loss_D_fake': loss_D_fake.item()
        }



def weights_init_normal(m):
    """Initialize network weights with normal distribution"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

