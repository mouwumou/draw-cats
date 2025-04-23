import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.data import make_dataloader
from src.models import make_models_and_losses
from src.utils import save_checkpoint, load_checkpoint, init_logging


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 保存初始和目标权重
    start_gan   = cfg.lambda_gan      # 默认 1.0
    end_gan     = 0.5                 # 训练末期希望降到 0.5
    start_l1    = cfg.lambda_l1       # 默认 100.0
    end_l1      = 5.0                # 末期降到 20
    start_perc  = cfg.lambda_perc     # 默认 1.0
    end_perc    = 5.0                 # 末期升到 5

    # DataLoader
    train_loader = make_dataloader(cfg, split='train', pin_memory=True)
    # val_loader = make_dataloader(cfg, split='val')
    if cfg.mode == 'pix2pix':
        val_loader = make_dataloader(cfg, split='val')
        val_samples = next(iter(val_loader))
        val_real = val_samples['real'].to(device)
        val_fake = val_samples['fake'].to(device)
    else:
        val_loader = make_dataloader(cfg, split='val', pin_memory=True)
        val_samples = next(iter(val_loader))
        # val_samples is a tuple (real_A_batch, real_B_batch)
        val_real_A = val_samples[0].to(device)
        val_real_B = val_samples[1].to(device)

    # Models & Losses
    models, criteria = make_models_and_losses(cfg, device)
    # G = models[0] if cfg.mode == 'pix2pix' else models[0]

    # Optimizers & Scheduler
    optimizers = []
    for m in (models if isinstance(models, (list, tuple)) else [models]):
        optimizers.append(optim.Adam(m.parameters(), lr=cfg.lr, betas=(0.5, 0.999)))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizers[0], lr_lambda=lambda epoch: 1 - max(0, epoch - cfg.decay_epoch) / (cfg.n_epochs - cfg.decay_epoch)
    )

    # Logger
    writer = SummaryWriter(log_dir=cfg.log_dir)
    logger = init_logging(writer)

    start_epoch = 1
    # Optionally resume
    if cfg.resume and os.path.isfile(cfg.resume):
        models, optimizers, start_epoch = load_checkpoint(cfg.resume, models, optimizers, device)

        
    global_step = 0
    # Training Loop
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        # ——— 动态权重更新 ———
        frac = (epoch - 1) / (cfg.n_epochs - 1)  # 0→1
        cfg.lambda_gan  = start_gan   * (1 - frac) + end_gan   * frac
        cfg.lambda_l1   = start_l1    * (1 - frac) + end_l1    * frac
        cfg.lambda_perc = start_perc  * (1 - frac) + end_perc  * frac
        print(f"[Epoch {epoch:03d}] λ_gan={cfg.lambda_gan:.3f}, λ_l1={cfg.lambda_l1:.1f}, λ_perc={cfg.lambda_perc:.1f}")

        for i, batch in enumerate(train_loader, 1):
            global_step += 1
            losses = {}  # forward_step returns dict of losses
            # Forward & backward
            for opt in optimizers:
                opt.zero_grad()

            losses = cfg.mode_forward(batch, models, criteria, cfg, device)
            total_loss = sum(losses.values())
            total_loss.backward()

            for opt in optimizers:
                opt.step()

            # Logging
            if i % cfg.log_interval == 0:
                logger.log(losses, global_step)

        # Visualization: write sample images at end of each epoch
        if val_samples is not None:
            if cfg.mode == 'pix2pix':
                with torch.no_grad():
                    G_AB, _, _ = models
                    fake_out = G_AB(val_real)
                    # Denormalize images from [-1,1] to [0,1]
                    imgs = torch.cat([val_real, fake_out, val_fake], dim=0)
                    imgs = (imgs + 1) / 2
                    writer.add_images('samples/real_fake_gt', imgs, epoch, dataformats='NCHW')
            else:
                with torch.no_grad():
                    G_AB, G_BA, _, _, _ = models
                    fake_B = G_AB(val_real_A)
                    rec_A = G_BA(fake_B)
                    fake_A = G_BA(val_real_B)
                    rec_B = G_AB(fake_A)
                    imgs = torch.cat([val_real_A, fake_B, rec_A,
                                    val_real_B, fake_A, rec_B], dim=0)
                    imgs = (imgs + 1) / 2
                    writer.add_images('samples/cyclegan/A2B2A__B2A2B', imgs, epoch, dataformats='NCHW')

        # Step scheduler
        scheduler.step()

        # Checkpoint
        if epoch % cfg.checkpoint_interval == 0:
            save_checkpoint(models, optimizers, epoch, cfg.checkpoint_dir)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Pix2Pix and CycleGAN Training')
    parser.add_argument('--mode', choices=['pix2pix', 'cycle'], required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_cyc', type=float, default=10.0)
    parser.add_argument('--lambda_id', type=float, default=5.0)
    parser.add_argument('--lambda_perc', type=float, default=1.0)
    parser.add_argument('--lambda_gan',  type=float, default=1.0,
                                        help='weight for GAN loss')
    parser.add_argument('--lambda_style', type=float, default=1.0,
                                        help='weight for style loss')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    cfg = Config.from_args(args)
    train(cfg)

