import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import lpips
import torchvision.utils as vutils

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from dataset_precomputed import SRDatasetPrecomputed


# --------------------------------------------------
# Config
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


CKPT_PATH = "checkpoints/eval_latest.pt" # alwas make a copy of the latest checkpoint
OUT_DIR = "results/DIV2K_valid_HR_1024"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUT_DIR}/grids", exist_ok=True)

# --------------------------------------------------
# Load model
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)

pipe.to(device)
pipe.vae.eval()
pipe.unet.eval()
pipe.controlnet.eval()
pipe.text_encoder.eval()

ckpt = torch.load(CKPT_PATH, map_location=device)
pipe.controlnet.load_state_dict(ckpt["controlnet"])

# --------------------------------------------------
# LPIPS
# --------------------------------------------------
lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()

# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = SRDatasetPrecomputed(
    hr_dir="~/datasets/test_data/DIV2K_valid_HR_1024",
    lr_dir="~/datasets/test_data/DIV2K_valid_LR_x4"
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def tensor_to_img_01(t):
    t = (t.clamp(-1, 1) + 1) / 2
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

# OPTION B: Y channel for inputs in [0,1]
def rgb2y_01(img01):
    # BT.601 luma (common), input [0,1] -> output [0,1]
    return (
        0.299 * img01[..., 0]
        + 0.587 * img01[..., 1]
        + 0.114 * img01[..., 2]
    )

def bicubic_upsample(lr_tensor, size):
    lr_01 = (lr_tensor + 1) / 2
    lr_01 = lr_01.squeeze(0).cpu()

    lr_img = Image.fromarray(
        (lr_01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )

    bicubic = lr_img.resize(size, Image.BICUBIC)
    return np.array(bicubic).astype(np.float32)

def img_to_tensor_01(img):
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t.to(device).float() * 2 - 1  # → [-1,1]

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
psnr_list, ssim_list = [], []
bic_psnr_list, bic_ssim_list = [], []
lpips_sr_list, lpips_bic_list = [], []

for i, (lr, hr) in enumerate(loader):
    lr = lr.to(device)
    hr = hr.to(device)

    # ControlNet conditioning (same as training)
    cond = torch.nn.functional.interpolate(
        lr,
        size=hr.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(-1, 1)

    with torch.no_grad():
        out = pipe(
            prompt="",
            image=cond,
            num_inference_steps=40,
            generator=torch.manual_seed(0),
        )

    sr_img = out.images[0]

    # Prepare images
    sr = np.array(sr_img).astype(np.float32)               # 0..255
    hr_np = tensor_to_img_01(hr).astype(np.float32)        # 0..1
    bicubic = bicubic_upsample(lr, size=(hr.shape[-1], hr.shape[-2]))  # 0..255

    # ---- SANITY CHECK (only once) ----
    if i == 0:
        sr01 = sr / 255.0
        bic01 = bicubic / 255.0
        print("SANITY (expected ~0..1):")
        print("  HR  min/max:", float(hr_np.min()), float(hr_np.max()))
        print("  SR  min/max:", float(sr01.min()), float(sr01.max()))
        print("  BIC min/max:", float(bic01.min()), float(bic01.max()), flush=True)

    # -----------------------------
    # PSNR / SSIM (Y channel)  [FIXED RANGE]
    # -----------------------------
    crop = 4

    sr01 = sr / 255.0
    bic01 = bicubic / 255.0

    sr_y = rgb2y_01(sr01)[crop:-crop, crop:-crop]
    hr_y = rgb2y_01(hr_np)[crop:-crop, crop:-crop]
    bic_y = rgb2y_01(bic01)[crop:-crop, crop:-crop]

    psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0)
    ssim = structural_similarity(hr_y, sr_y, data_range=1.0)

    bic_psnr = peak_signal_noise_ratio(hr_y, bic_y, data_range=1.0)
    bic_ssim = structural_similarity(hr_y, bic_y, data_range=1.0)

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    bic_psnr_list.append(bic_psnr)
    bic_ssim_list.append(bic_ssim)

    # -----------------------------
    # LPIPS
    # -----------------------------
    with torch.no_grad():
        sr_t = img_to_tensor_01(sr)
        hr_t = img_to_tensor_01(hr_np)
        bic_t = img_to_tensor_01(bicubic)

        lpips_sr = lpips_fn(sr_t, hr_t).item()
        lpips_bic = lpips_fn(bic_t, hr_t).item()

    lpips_sr_list.append(lpips_sr)
    lpips_bic_list.append(lpips_bic)

    # -----------------------------
    # Save grid: LR | Bicubic | SR | HR
    # -----------------------------
    lr_up = bicubic_upsample(lr, size=(hr.shape[-1], hr.shape[-2]))

    grid = torch.cat([
        img_to_tensor_01(lr_up),
        img_to_tensor_01(bicubic),
        img_to_tensor_01(sr),
        img_to_tensor_01(hr_np),
    ], dim=0)

    vutils.save_image(
        (grid + 1) / 2,
        f"{OUT_DIR}/grids/{i:04d}_grid.png",
        nrow=4,
    )

    print(
        f"[{i}] "
        f"SR PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_sr:.4f} | "
        f"Bicubic PSNR: {bic_psnr:.2f}, SSIM: {bic_ssim:.4f}, LPIPS: {lpips_bic:.4f}"
    )

# --------------------------------------------------
# Report
# --------------------------------------------------
with open(f"{OUT_DIR}/avg_metrics.txt", "w") as f:
    f.write("=== ControlNet SR ===\n")
    f.write(f"PSNR:  {np.mean(psnr_list):.2f}\n")
    f.write(f"SSIM:  {np.mean(ssim_list):.4f}\n")
    f.write(f"LPIPS: {np.mean(lpips_sr_list):.4f}\n\n")

    f.write("=== Bicubic ===\n")
    f.write(f"PSNR:  {np.mean(bic_psnr_list):.2f}\n")
    f.write(f"SSIM:  {np.mean(bic_ssim_list):.4f}\n")
    f.write(f"LPIPS: {np.mean(lpips_bic_list):.4f}\n")

print("Evaluation complete")
