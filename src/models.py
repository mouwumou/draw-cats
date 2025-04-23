import torch
from torch import nn
from model.pix2pix import UNetGenerator, PatchGANDiscriminator, ResNetGenerator
from model.pix2pix_stn import UNetWithSTN
from model.generator import Generator as ResnetGenerator
from model.discriminator import Discriminator as PatchDiscriminator

from torchvision.models import vgg19
import torch.nn.functional as F

# Perceptual extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(pretrained=True).features.to(device)
        self.slice = nn.Sequential(*[vgg[i] for i in range(21)])  # up to relu4_1
        for p in self.slice.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.slice(x)
    
# Gram-style Loss
def gram_matrix(x):
    B,C,H,W = x.size()
    f = x.view(B, C, H*W)
    G = torch.bmm(f, f.transpose(1,2)) / (C*H*W)
    return G

class StyleLoss(nn.Module):
    def __init__(self, vgg_extractor, weight):
        super().__init__()
        self.vgg = vgg_extractor.eval()
        self.weight = weight
        self.criterion = nn.MSELoss()
    def forward(self, input, style):
        F_in    = self.vgg(input)
        F_style = self.vgg(style)
        return self.criterion(gram_matrix(F_in), gram_matrix(F_style)) * self.weight


def make_models_and_losses(cfg, device):
    if cfg.mode == 'pix2pix':
        # G = UNetGenerator().to(device)
        G = UNetWithSTN().to(device)
        D = PatchGANDiscriminator().to(device)
        criterion_GAN = nn.MSELoss()
        criterion_recon = nn.L1Loss()
        # add perceptual loss
        perc_extractor = VGGFeatureExtractor(device).to(device)
        criterion_perc = nn.L1Loss()
        models = (G, D, perc_extractor)
        criteria = {'gan': criterion_GAN, 'recon': criterion_recon, 'perc': criterion_perc}
    else:
        G_AB = ResnetGenerator().to(device)
        G_BA = ResnetGenerator().to(device)
        D_A = PatchDiscriminator().to(device)
        D_B = PatchDiscriminator().to(device)

        vgg_extractor = VGGFeatureExtractor(device).to(device)
        style_criterion = StyleLoss(vgg_extractor, cfg.lambda_style)
        perc_criterion  = nn.L1Loss()

        gan_criterion   = nn.MSELoss()
        cycle_criterion = nn.L1Loss()
        id_criterion    = nn.L1Loss()
        criteria = {
            'gan':      gan_criterion,
            'cyc':      cycle_criterion,
            'id':       id_criterion,
            'perc':     perc_criterion,
            'style':    style_criterion,
        }
        models = (G_AB, G_BA, D_A, D_B, vgg_extractor)
    # forward logic
    cfg.mode_forward = (pix2pix_forward if cfg.mode=='pix2pix' else cycle_forward)
    return models, criteria


def pix2pix_forward(batch, models, criteria, cfg, device):
    G, D, vgg = models
    real = batch['real'].to(device)
    fake = batch['fake'].to(device)
    # GAN
    fake_B = G(real)
    pred_fake = D(real, fake_B)
    loss_GAN = criteria['gan'](pred_fake, torch.ones_like(pred_fake)) * cfg.lambda_gan
    loss_L1 = criteria['recon'](fake_B, fake) * cfg.lambda_l1
    # perceptual loss
    real_n = (real + 1) / 2
    fake_n = (fake_B + 1) / 2
    # make sure the input to VGG is 3 channels
    if real_n.shape[1] == 1:
        real_n = real_n.repeat(1, 3, 1, 1)
        fake_n = fake_n.repeat(1, 3, 1, 1)
    # extract VGG features and calculate perceptual loss
    feat_real = vgg(real_n)
    feat_fake = vgg(fake_n)
    loss_perc = criteria['perc'](feat_fake, feat_real) * cfg.lambda_perc

    # Disc
    pred_real = D(real, fake)
    loss_D_real = criteria['gan'](pred_real, torch.ones_like(pred_real))
    pred_fake_det = D(real, fake_B.detach())
    loss_D_fake = criteria['gan'](pred_fake_det, torch.zeros_like(pred_fake_det))

    losses = {
        'G_GAN':  loss_GAN,
        'G_L1':   loss_L1,
        'G_perc': loss_perc,
        'D_real': loss_D_real,
        'D_fake': loss_D_fake
    }
    return losses


def cycle_forward(batch, models, criteria, cfg, device):
    real_A, real_B = batch[0].to(device), batch[1].to(device)
    G_AB, G_BA, D_A, D_B, vgg = models

    # A → B
    fake_B = G_AB(real_A)
    loss_GAN_AB = criteria['gan'](D_B(fake_B), torch.ones_like(D_B(fake_B))) * cfg.lambda_gan

    # B → A
    fake_A = G_BA(real_B)
    loss_GAN_BA = criteria['gan'](D_A(fake_A), torch.ones_like(D_A(fake_A))) * cfg.lambda_gan

    # 循环一致
    rec_A = G_BA(fake_B)
    rec_B = G_AB(fake_A)
    loss_cyc = (criteria['cyc'](rec_A, real_A) + criteria['cyc'](rec_B, real_B)) * cfg.lambda_cyc

    # 身份保持
    id_A = G_BA(real_A); id_B = G_AB(real_B)
    loss_id = (criteria['id'](id_A, real_A) + criteria['id'](id_B, real_B)) * cfg.lambda_id

    # —— 新增 —— #
    # 1) 感知损失：用 VGG 特征对齐 fake_B ↔ real_B
    real_B_n = (real_B + 1) / 2
    fake_B_n = (fake_B + 1) / 2
    feat_real = vgg(real_B_n)
    feat_fake = vgg(fake_B_n)
    loss_perc = criteria['perc'](feat_fake, feat_real) * cfg.lambda_perc

    # 2) 风格损失：计算 fake_B ↔ real_B 的 Gram 矩阵差
    loss_style = criteria['style'](fake_B_n, real_B_n)

    # 判别器损失同之前 …
    pred_real_A = D_A(real_A)
    loss_D_A_real = criteria['gan'](pred_real_A, torch.ones_like(pred_real_A))
    loss_D_A_fake = criteria['gan'](D_A(fake_A.detach()), torch.zeros_like(pred_real_A))
    pred_real_B  = D_B(real_B)
    loss_D_B_real = criteria['gan'](pred_real_B, torch.ones_like(pred_real_B))
    loss_D_B_fake = criteria['gan'](D_B(fake_B.detach()), torch.zeros_like(pred_real_B))

    losses = {
        'G_GAN_AB':  loss_GAN_AB,
        'G_GAN_BA':  loss_GAN_BA,
        'G_cyc':     loss_cyc,
        'G_id':      loss_id,
        'G_perc':    loss_perc,
        'G_style':   loss_style,
        'D_A_real':  loss_D_A_real,
        'D_A_fake':  loss_D_A_fake,
        'D_B_real':  loss_D_B_real,
        'D_B_fake':  loss_D_B_fake,
    }
    return losses
