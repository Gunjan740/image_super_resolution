import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
)
from dotenv import load_dotenv

from dataset import SRDataset  # your dataset

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
device = "cpu"
torch.manual_seed(0)

# --------------------------------------------------
# Load ControlNet + Stable Diffusion
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile",
    torch_dtype=torch.float32,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
)

pipe.to(device)

# Freeze Stable Diffusion U-Net
for p in pipe.unet.parameters():
    p.requires_grad = False

pipe.controlnet.train()

# --------------------------------------------------
# Dataset & DataLoader (small sample)
# --------------------------------------------------
dataset = SRDataset(
    hr_dir="data/sample/hr",
    scale=4,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

# --------------------------------------------------
# Noise scheduler (diffusion)
# --------------------------------------------------
noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler",
)

# --------------------------------------------------
# Optimizer (ONLY ControlNet)
# --------------------------------------------------
optimizer = torch.optim.AdamW(
    pipe.controlnet.parameters(),
    lr=1e-5,
)

# --------------------------------------------------
# Training loop (SKELETON)
# --------------------------------------------------
for step, (lr_img, hr_img) in enumerate(dataloader):

    if step >= 2:  # 🔴 only 1–2 steps for validation
        break

    # -----------------------------
    # Move data to device
    # -----------------------------
    hr_img = hr_img.to(device)
    lr_img = lr_img.to(device)

    # -----------------------------
    # Encode HR image → latent z0
    # -----------------------------
    with torch.no_grad():
        latents = pipe.vae.encode(hr_img).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # -----------------------------
    # Sample noise & timestep
    # -----------------------------
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
    ).long()

    # -----------------------------
    # Add noise (forward diffusion)
    # -----------------------------
    noisy_latents = noise_scheduler.add_noise(
        latents, noise, timesteps
    )

    # -----------------------------
    # Forward pass
    # -----------------------------
    model_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=None,
        controlnet_cond=lr_img,
    ).sample

    # -----------------------------
    # Loss (noise prediction)
    # -----------------------------
    loss = torch.nn.functional.mse_loss(model_pred, noise)

    # -----------------------------
    # Backprop
    # -----------------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.6f}")

print("Training loop skeleton completed")
