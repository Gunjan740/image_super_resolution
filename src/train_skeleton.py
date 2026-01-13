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
# Checkpoint & logging setup
# --------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

save_every = 200
max_steps = 5_000_000
num_epochs = 200

loss_log_path = "logs/train_loss.csv"


# --------------------------------------------------
# Load ControlNet + Stable Diffusion (FP32 weights)
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
    hr_dir=os.path.expanduser("~/datasets/DIV2K/DIV2K_train_HR"),
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
# Optimizer & AMP
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
# 🔁 Resume from checkpoint if available
# --------------------------------------------------
latest_ckpt = "checkpoints/latest.pt"
global_step = 0
start_epoch = 0

if os.path.exists(latest_ckpt):
    print(f"🔄 Resuming from {latest_ckpt}", flush=True)
    ckpt = torch.load(latest_ckpt, map_location=device)

    pipe.controlnet.load_state_dict(ckpt["controlnet"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    global_step = ckpt["global_step"]
    start_epoch = ckpt["epoch"]

    print(
        f"✅ Resumed at epoch {start_epoch}, global step {global_step}",
        flush=True,
    )

# --------------------------------------------------
# Loss logging (append-safe)
# --------------------------------------------------
if not os.path.exists(loss_log_path):
    loss_log = open(loss_log_path, "w")
    loss_log.write("global_step,loss\n")
else:
    loss_log = open(loss_log_path, "a")


# --------------------------------------------------
# Training loop (epoch + global_step)
# --------------------------------------------------
for epoch in range(start_epoch, num_epochs):
    print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====", flush=True)

    for lr_img, hr_img in dataloader:

        if global_step >= max_steps:
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
        # ControlNet conditioning
        # -----------------------------
        h_lat, w_lat = noisy_latents.shape[-2:]
        cond = F.interpolate(
            lr_img,
            size=(h_lat * 8, w_lat * 8),
            mode="bilinear",
            align_corners=False,
        ).clamp(-1.0, 1.0)

        # -----------------------------
        # Forward + loss
        # -----------------------------
        ctx = autocast("cuda") if device == "cuda" else torch.no_grad()

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
        # Backprop
        # -----------------------------
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # -----------------------------
        # Save checkpoints
        # -----------------------------
        if global_step % save_every == 0 and global_step > 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "controlnet": pipe.controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            }

            torch.save(ckpt, "checkpoints/latest.pt")
            torch.save(
                ckpt,
                f"checkpoints/controlnet_step_{global_step}.pt",
            )

        # -----------------------------
        # Log loss
        # -----------------------------
        loss_log.write(f"{global_step},{loss.item()}\n")
        loss_log.flush()

        print(
            f"Epoch {epoch + 1} | Step {global_step} | "
            f"Loss: {loss.item():.6f} | "
            f"latents: {tuple(noisy_latents.shape)} | "
            f"cond: {tuple(cond.shape)}",
            flush=True,
        )

        global_step += 1

    if global_step >= max_steps:
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------
loss_log.close()
print("✅ Training loop completed")
