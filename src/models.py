import torch
from torch import nn
from model.pix2pix import UNetGenerator, PatchGANDiscriminator, VGGFeatureExtractor
from model.generator import Generator as ResnetGenerator
from model.discriminator import Discriminator as PatchDiscriminator


def make_models_and_losses(cfg, device):
    if cfg.mode == 'pix2pix':
        G = UNetGenerator().to(device)
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
        criterion_GAN = nn.MSELoss()
        criterion_cyc = nn.L1Loss()
        criterion_id = nn.L1Loss()
        models = (G_AB, G_BA, D_A, D_B)
        criteria = {'gan': criterion_GAN, 'cycle': criterion_cyc, 'identity': criterion_id}
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
    G_AB, G_BA, D_A, D_B = models
    real_A, real_B = batch
    real_A, real_B = real_A.to(device), real_B.to(device)
    # AB->A cycle
    fake_B = G_AB(real_A)
    rec_A = G_BA(fake_B)
    # BA->B cycle
    fake_A = G_BA(real_B)
    rec_B = G_AB(fake_A)
    # GAN losses
    loss_GAN_AB = criteria['gan'](D_B(fake_B), torch.ones_like(D_B(fake_B)))
    loss_GAN_BA = criteria['gan'](D_A(fake_A), torch.ones_like(D_A(fake_A)))
    # cycle loss
    loss_cycle = (criteria['cycle'](rec_A, real_A) + criteria['cycle'](rec_B, real_B)) * cfg.lambda_cyc
    # identity
    id_A = G_BA(real_A)
    id_B = G_AB(real_B)
    loss_id = (criteria['identity'](id_A, real_A) + criteria['identity'](id_B, real_B)) * cfg.lambda_id
    # Discriminator losses
    loss_D_A = 0.5 * (criteria['gan'](D_A(real_A), torch.ones_like(D_A(real_A))) +
                      criteria['gan'](D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A))))
    loss_D_B = 0.5 * (criteria['gan'](D_B(real_B), torch.ones_like(D_B(real_B))) +
                      criteria['gan'](D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B))))
    losses = {'G_AB': loss_GAN_AB, 'G_BA': loss_GAN_BA,
              'cycle': loss_cycle, 'identity': loss_id,
              'D_A': loss_D_A, 'D_B': loss_D_B}
    return losses
