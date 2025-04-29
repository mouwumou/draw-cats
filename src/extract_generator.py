import argparse
from pathlib import Path
import torch

# -------------------------------------------------------------
# Script: extract_generator.py
# Goal: Extract the generator state_dict from a full checkpoint
#       containing multiple components (generator, discriminator, optimizer).
#       Save the clean generator state_dict separately.
# Example usage:
#   python extract_generator.py \
#       --ckpt weights/full_checkpoint.pth \
#       --out  models/pix2pix_G.pth
# -------------------------------------------------------------

from model.pix2pix_stn import UNetGenerator as UNetGeneratorSTN


def find_state_dict(ckpt: dict) -> dict:
    """在常见字段中定位 generator 的 state_dict"""
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict")
    # 优先级顺序
    for key in ("model_0", "netG", "G", "generator", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    # DataParallel 前缀清理
    if all(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def main(args):
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print(f"Loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = find_state_dict(raw)

    net = UNetGeneratorSTN()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"Loaded -> missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Only net.state_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f"✅ Clean generator state_dict saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract generator weights from full checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to full checkpoint (.pth)")
    parser.add_argument("--out", required=True, help="Output path for clean generator weights")
    parser.add_argument("--img_size", type=int, default=256, help="Image size used during training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size used during training")
    args = parser.parse_args()
    main(args)
