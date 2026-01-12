import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from dataset import SRDataset

# --------------------------------------------------
# Config
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

HR_DIR = os.path.expanduser("~/datasets/div2k_sample/DIV2K_valid_HR_10")
CKPT_PATH = "checkpoints/latest.pt"
OUT_DIR = "results/div2k_val_epoch20"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/images", exist_ok=True)

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
# Dataset (unchanged)
# --------------------------------------------------
dataset = SRDataset(
    hr_dir=HR_DIR,
    scale=4,
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def tensor_to_img_01(t):
    t = (t.clamp(-1, 1) + 1) / 2
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()

def rgb2y(img):
    return (
        0.257 * img[..., 0]
        + 0.504 * img[..., 1]
        + 0.098 * img[..., 2]
        + 16.0
    ) / 255.0

def bicubic_upsample(lr_tensor, size):
    lr_01 = (lr_tensor + 1) / 2
    lr_01 = lr_01.squeeze(0).cpu()

    lr_img = Image.fromarray(
        (lr_01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )

    bicubic = lr_img.resize(size, Image.BICUBIC)
    return np.array(bicubic).astype(np.float32)

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
psnr_list, ssim_list = [], []
bic_psnr_list, bic_ssim_list = [], []

for i, (lr, hr) in enumerate(loader):
    lr = lr.to(device)
    hr = hr.to(device)

    # -----------------------------
    # ControlNet conditioning (same as training)
    # -----------------------------
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
            num_inference_steps=30,
            generator=torch.manual_seed(0),
        )

    sr_img = out.images[0]

    # -----------------------------
    # Prepare images for metrics
    # -----------------------------
    sr = np.array(sr_img).astype(np.float32)
    hr_np = tensor_to_img_01(hr)

    # Bicubic baseline
    bicubic = bicubic_upsample(
        lr,
        size=(hr.shape[-1], hr.shape[-2])
    )

    # -----------------------------
    # Y channel + border crop
    # -----------------------------
    crop = 4

    sr_y = rgb2y(sr)[crop:-crop, crop:-crop]
    hr_y = rgb2y(hr_np)[crop:-crop, crop:-crop]
    bic_y = rgb2y(bicubic)[crop:-crop, crop:-crop]

    # -----------------------------
    # Metrics
    # -----------------------------
    psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0)
    ssim = structural_similarity(hr_y, sr_y, data_range=1.0)

    bic_psnr = peak_signal_noise_ratio(hr_y, bic_y, data_range=1.0)
    bic_ssim = structural_similarity(hr_y, bic_y, data_range=1.0)

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    bic_psnr_list.append(bic_psnr)
    bic_ssim_list.append(bic_ssim)

    # Save a few images
    if i < 10:
        sr_img.save(f"{OUT_DIR}/images/{i:04d}_SR.png")

    print(
        f"[{i}] "
        f"SR PSNR: {psnr:.2f}, SSIM: {ssim:.4f} | "
        f"Bicubic PSNR: {bic_psnr:.2f}, SSIM: {bic_ssim:.4f}"
    )

# --------------------------------------------------
# Report
# --------------------------------------------------
with open(f"{OUT_DIR}/avg_metrics.txt", "w") as f:
    f.write(f"ControlNet PSNR: {np.mean(psnr_list):.2f}\n")
    f.write(f"ControlNet SSIM: {np.mean(ssim_list):.4f}\n\n")
    f.write(f"Bicubic PSNR: {np.mean(bic_psnr_list):.2f}\n")
    f.write(f"Bicubic SSIM: {np.mean(bic_ssim_list):.4f}\n")

print("✅ Evaluation complete")
