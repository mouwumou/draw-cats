import os
from dataclasses import dataclass

@dataclass
class Config:
    mode: str
    data_root: str
    n_epochs: int
    batch_size: int
    lr: float
    decay_epoch: int
    lambda_l1: float
    lambda_cyc: float
    lambda_id: float
    img_size: int
    log_interval: int
    checkpoint_interval: int
    log_dir: str
    checkpoint_dir: str
    resume: str

    @staticmethod
    def from_args(args):
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        return Config(
            mode=args.mode,
            data_root=args.data_root,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            decay_epoch=args.decay_epoch,
            lambda_l1=args.lambda_l1,
            lambda_cyc=args.lambda_cyc,
            lambda_id=args.lambda_id,
            img_size=args.img_size,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            log_dir=args.log_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
        )