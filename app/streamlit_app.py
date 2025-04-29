import io
import pathlib
from typing import Tuple

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

import sys, pathlib

# ---------------------------- CONFIG ---------------------------------- #
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
MODEL_PATH = pathlib.Path("weights/pix2pix_G.pth")  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256  
DISPLAY_W = 256

# ---------------------------- HELPERS --------------------------------- #
@st.cache_resource(show_spinner=False)
def load_generator(path: pathlib.Path) -> torch.nn.Module:
    """懒加载生成器模型，并切到 eval 模式。"""
    if not path.exists():
        st.error(f"Can't find model file: {path}")
        st.stop()
    from model.pix2pix_stn import UNetGenerator as UNetGeneratorSTN

    netG = UNetGeneratorSTN()
    netG.load_state_dict(torch.load(path, map_location="cpu"))
    netG.eval().to(DEVICE)
    return netG

@torch.no_grad()
def run_pix2pix(netG: torch.nn.Module, img: Image.Image) -> Image.Image:
    """前向推理: PIL -> PIL"""
    # 1. 预处理
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),               # [0,1]
        T.Normalize(0.5, 0.5),      # [-1,1]
    ])
    x = tf(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 2. 生成
    fake = netG(x)[0].cpu()               # (3,H,W)

    # 3. 反归一化 & 转成 PIL
    fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)
    fake_img = T.ToPILImage()(fake)
    return fake_img

# ----------------------------- UI ------------------------------------- #
st.set_page_config(page_title="Pix2Pix Cat Sketch", page_icon="🐱", layout="wide")
st.title("🖌️ Cat Real → Cartoon (Pix2Pix)")

uploaded = st.file_uploader("Upload Cat Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    input_img = Image.open(uploaded).convert("RGB")
    with st.spinner("Stylising…"):
        netG = load_generator(MODEL_PATH)
        output_img = run_pix2pix(netG, input_img)
        output_img = output_img.resize(input_img.size, Image.BICUBIC)

    # Side‑by‑side display
    st.image([input_img, output_img], caption=["Real", "stylized"], width=DISPLAY_W)

    # Download button
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    st.download_button("Download Result", buf.getvalue(), file_name="cartoon_cat.png", mime="image/png")
else:
    st.info("👈 Please upload an image to begin")

# --------------------------- FOOTER ----------------------------------- #
st.markdown("---")
st.caption("Pix2Pix demo · Powered by Streamlit & PyTorch")
