import os
import json
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

HR_DIR = os.path.expanduser("~/datasets/test_data/DIV2K_valid_HR_1024")
LR_DIR = os.path.expanduser("~/datasets/test_data/DIV2K_valid_LR_x4")

CKPT_PATH = "checkpoints/eval_latest_without_prompt.pt"

OUT_DIR = "results/without_prompt/DIV2K_valid_HR_1024"
os.makedirs(OUT_DIR, exist_ok=True)

START_IDX = 0
NUM_IMAGES = None   # None = evaluate all

# -------------------------
# Prompt control
# -------------------------
PROMPT_MODE = "none"   # ["none", "lr_semantic", "lr_texture", ...]
CAPTIONS_JSONL = os.path.expanduser(
    "~/datasets/DF2K/df2k_LR_x4_semantic_captions.jsonl"
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_caption_map(jsonl_path, prefer_key="caption_clean"):
    cap_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cap_map[rec["file"]] = rec.get(prefer_key, "")
    return cap_map


# --------------------------------------------------
# Load captions
# --------------------------------------------------
caption_map = {}
if PROMPT_MODE != "none":
    caption_map = load_caption_map(CAPTIONS_JSONL)


# --------------------------------------------------
# Load ControlNet + SD
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)
pipe.to(device)
pipe.eval()

# Load trained ControlNet weights
ckpt = torch.load(CKPT_PATH, map_location=device)
pipe.controlnet.load_state_dict(ckpt["controlnet"])


# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = SRDatasetPrecomputed(
    hr_dir=HR_DIR,
    lr_dir=LR_DIR,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------
lpips_fn = lpips.LPIPS(net="alex").to(device)

psnr_list = []
ssim_list = []
lpips_list = []


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
for i, (lr, hr, fname) in enumerate(loader):

    if i < START_IDX:
        continue
    if NUM_IMAGES is not None and i >= START_IDX + NUM_IMAGES:
        break

    fname = fname[0] if isinstance(fname, (list, tuple)) else fname

    lr = lr.to(device)
    hr = hr.to(device)

    # -----------------------------
    # Prompt selection + sanity
    # -----------------------------
    if PROMPT_MODE == "none":
        prompt = ""
    else:
        prompt = caption_map.get(fname, "")
        if prompt == "":
            raise RuntimeError(f"Missing caption for {fname}")

    if i == START_IDX:
        print(f"[SANITY] first file={fname}", flush=True)
        print(f"[SANITY] first prompt={prompt}", flush=True)

    # -----------------------------
    # Run pipeline
    # -----------------------------
    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=lr,
            num_inference_steps=40,
            generator=torch.manual_seed(0),
        )

    sr = out.images[0]
    sr = torch.from_numpy(np.array(sr)).permute(2, 0, 1).float() / 255.0
    sr = sr.unsqueeze(0).to(device)

    # -----------------------------
    # Metrics
    # -----------------------------
    hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()

    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    ssim = structural_similarity(hr_np, sr_np, channel_axis=-1, data_range=1.0)
    lp = lpips_fn(sr, hr).item()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    lpips_list.append(lp)

    # -----------------------------
    # Save image grid
    # -----------------------------
    grid = vutils.make_grid(
        torch.cat([lr, sr, hr], dim=0),
        nrow=3,
        normalize=True,
        value_range=(-1, 1),
    )
    vutils.save_image(grid, f"{OUT_DIR}/{i:04d}_{fname}")

    print(
        f"[{i}] PSNR={psnr:.2f} | SSIM={ssim:.4f} | LPIPS={lp:.4f} | file={fname}",
        flush=True,
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------
print("==================================================")
print(f"Prompt mode : {PROMPT_MODE}")
print(f"Mean PSNR   : {np.mean(psnr_list):.2f}")
print(f"Mean SSIM   : {np.mean(ssim_list):.4f}")
print(f"Mean LPIPS : {np.mean(lpips_list):.4f}")
print("==================================================")
