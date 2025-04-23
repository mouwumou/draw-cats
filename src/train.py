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

    # DataLoader
    train_loader = make_dataloader(cfg, split='train')
    val_loader = make_dataloader(cfg, split='val') if cfg.mode == 'pix2pix' else None

    # Models & Losses
    models, criteria = make_models_and_losses(cfg, device)
    G = models[0] if cfg.mode == 'pix2pix' else models[0]

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

    # Fixed validation samples for visualization
    val_samples = None
    if val_loader:
        val_samples = next(iter(val_loader))
        val_real = val_samples['real'].to(device)
        val_fake = val_samples['fake'].to(device)
        
    global_step = 0
    # Training Loop
    for epoch in range(start_epoch, cfg.n_epochs + 1):
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
            with torch.no_grad():
                fake_out = G(val_real)
                # Denormalize images from [-1,1] to [0,1]
                imgs = torch.cat([val_real, fake_out, val_fake], dim=0)
                imgs = (imgs + 1) / 2
                writer.add_images('samples/real_fake_gt', imgs, epoch, dataformats='NCHW')

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
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    cfg = Config.from_args(args)
    train(cfg)

