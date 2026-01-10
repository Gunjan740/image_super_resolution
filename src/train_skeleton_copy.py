import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
)
from dotenv import load_dotenv
from torch.amp import GradScaler, autocast

from dataset import SRDataset


# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# --------------------------------------------------
# Load ControlNet + Stable Diffusion (FP32 weights!)
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)

pipe.to(device)

# Freeze everything except ControlNet
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)

pipe.vae.eval()
pipe.text_encoder.eval()
pipe.unet.eval()
pipe.controlnet.train()


# --------------------------------------------------
# Dataset & DataLoader
# --------------------------------------------------
dataset = SRDataset(
    hr_dir=os.path.expanduser("~/datasets/div2k_sample/hr"),
    scale=4,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=(device == "cuda"),
)


# --------------------------------------------------
# Noise scheduler
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

scaler = GradScaler("cuda") if device == "cuda" else None


# --------------------------------------------------
# Precompute text embeddings (empty prompt)
# --------------------------------------------------
with torch.no_grad():
    tokens = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    encoder_hidden_states = pipe.text_encoder(input_ids)[0]


# --------------------------------------------------
# Training loop (SKELETON)
# --------------------------------------------------
for step, (lr_img, hr_img) in enumerate(dataloader):

    if step >= 2:  # short validation run
        break

    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)

    # -----------------------------
    # Encode HR → latent
    # -----------------------------
    with torch.no_grad():
        latents = pipe.vae.encode(hr_img).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    # -----------------------------
    # Noise + timestep
    # -----------------------------
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
    ).long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # -----------------------------
    # ControlNet conditioning (image space)
    # -----------------------------
    h_lat, w_lat = noisy_latents.shape[-2:]
    cond = F.interpolate(
        lr_img,
        size=(h_lat * 8, w_lat * 8),
        mode="bilinear",
        align_corners=False,
    ).clamp(-1.0, 1.0)

    # -----------------------------
    # Forward + loss (AMP)
    # -----------------------------
    if device == "cuda":
        ctx = autocast("cuda")
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        controlnet_out = pipe.controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=cond,
            return_dict=True,
        )

        model_pred = pipe.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_out.down_block_res_samples,
            mid_block_additional_residual=controlnet_out.mid_block_res_sample,
            return_dict=True,
        ).sample

        loss = F.mse_loss(model_pred, noise)

    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ NaN/Inf loss detected, skipping step", flush=True)
        continue

    # -----------------------------
    # Backprop (AMP-safe)
    # -----------------------------
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    print(
        f"Step {step} | Loss: {loss.item():.6f} | "
        f"latents: {tuple(noisy_latents.shape)} | cond: {tuple(cond.shape)}",
        flush=True,
    )

print("✅ Training loop skeleton completed")

