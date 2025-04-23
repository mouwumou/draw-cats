import torch
import os

def save_checkpoint(models, optimizers, epoch, ckpt_dir):
    state = {'epoch': epoch}
    for i, m in enumerate(models if isinstance(models, (list, tuple)) else [models]):
        state[f'model_{i}'] = m.state_dict()
    for i, opt in enumerate(optimizers):
        state[f'opt_{i}'] = opt.state_dict()
    path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, path)


def load_checkpoint(path, models, optimizers, device):
    checkpoint = torch.load(path, map_location=device)
    for i, m in enumerate(models if isinstance(models, (list, tuple)) else [models]):
        m.load_state_dict(checkpoint[f'model_{i}'])
    for i, opt in enumerate(optimizers):
        opt.load_state_dict(checkpoint[f'opt_{i}'])
    return models, optimizers, checkpoint['epoch'] + 1


def init_logging(writer):
    class Logger:
        def __init__(self, w): self.writer = w
        def log(self, losses, step):
            for k, v in losses.items():
                self.writer.add_scalar(k, v.item(), step)
    return Logger(writer)