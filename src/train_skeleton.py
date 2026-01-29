import os
import json
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

from dataset_precomputed import SRDatasetPrecomputed


# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# --------------------------------------------------
# Config: checkpoints + prompts
# --------------------------------------------------
CKPT_DIR = "checkpoints_LR_texture"
os.makedirs(CKPT_DIR, exist_ok=True)

os.makedirs("logs", exist_ok=True)
loss_log_path = "logs/train_loss_LR_texture.csv"

save_every = 5000
max_steps = 50_000_00

latest_ckpt = os.path.join(CKPT_DIR, "latest.pt")

PROMPT_MODE = "lr_texture"
CAPTIONS_JSONL = os.path.expanduser(
    "~/datasets/DF2K/df2k_LR_x4_texture_captions.jsonl"
)


# --------------------------------------------------
# Helpers: load captions + embedding cache
# --------------------------------------------------
def load_caption_map(jsonl_path: str, prefer_key: str = "caption_clean"):
    cap_map = {}
    if jsonl_path is None:
        return cap_map

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fname = rec.get("file")
            if not fname:
                continue
            cap = rec.get(prefer_key) or rec.get("caption_raw") or ""
            cap_map[fname] = cap
    return cap_map


@torch.no_grad()
def get_text_embedding(prompt: str, pipe, cache: dict, device: str):
    if prompt in cache:
        return cache[prompt]

    tokens = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    emb = pipe.text_encoder(input_ids)[0]
    cache[prompt] = emb
    return emb


# --------------------------------------------------
# Load ControlNet + Stable Diffusion
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)
pipe.to(device)

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
dataset = SRDatasetPrecomputed(
    lr_dir=os.path.expanduser("~/datasets/DF2K/DF2K_LR_x4"),
    hr_dir=os.path.expanduser("~/datasets/DF2K/DF2K_HR_1024"),
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=2,
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
# Captions + embedding cache
# --------------------------------------------------
caption_map = {}
if PROMPT_MODE != "none":
    caption_map = load_caption_map(CAPTIONS_JSONL, prefer_key="caption_clean")

text_emb_cache = {}


# --------------------------------------------------
# Resume from checkpoint
# --------------------------------------------------
global_step = 0

if os.path.exists(latest_ckpt):
    print(f"Resuming from {latest_ckpt}", flush=True)
    ckpt = torch.load(latest_ckpt, map_location=device)

    pipe.controlnet.load_state_dict(ckpt["controlnet"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    global_step = ckpt["global_step"]
    print(f"Resumed at global step {global_step}", flush=True)


# --------------------------------------------------
# Loss logging
# --------------------------------------------------
if not os.path.exists(loss_log_path):
    loss_log = open(loss_log_path, "w")
    loss_log.write("global_step,loss\n")
else:
    loss_log = open(loss_log_path, "a")


# --------------------------------------------------
# Training loop
# --------------------------------------------------
while global_step < max_steps:

    for batch in dataloader:
        if global_step >= max_steps:
            break

        lr_img, hr_img, fname = batch
        fname = fname[0] if isinstance(fname, (list, tuple)) else fname

        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        # -----------------------------
        # Text prompt + sanity checks
        # -----------------------------
        if PROMPT_MODE == "none":
            prompt = ""
        else:
            prompt = caption_map.get(fname, "")

            if prompt == "":
                raise RuntimeError(f"Missing caption for {fname}")

        if global_step == 0:
            print(f"[SANITY] first file={fname}", flush=True)
            print(f"[SANITY] first prompt={prompt}", flush=True)

        with torch.no_grad():
            encoder_hidden_states = get_text_embedding(
                prompt, pipe, text_emb_cache, device
            )

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
            print("NaN/Inf loss detected, skipping step", flush=True)
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
                "global_step": global_step,
                "controlnet": pipe.controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "prompt_mode": PROMPT_MODE,
                "captions_jsonl": CAPTIONS_JSONL,
            }
            torch.save(ckpt, os.path.join(CKPT_DIR, "latest.pt"))
            torch.save(
                ckpt,
                os.path.join(CKPT_DIR, f"controlnet_step_{global_step}.pt"),
            )

        # -----------------------------
        # Log loss
        # -----------------------------
        loss_log.write(f"{global_step},{loss.item()}\n")
        loss_log.flush()

        print(
            f"Step {global_step} | Loss: {loss.item():.6f} | "
            f"prompt_mode: {PROMPT_MODE} | prompt_len: {len(prompt)} | "
            f"latents: {tuple(noisy_latents.shape)} | cond: {tuple(cond.shape)} | "
            f"file: {fname}",
            flush=True,
        )

        global_step += 1


# --------------------------------------------------
# Cleanup
# --------------------------------------------------
loss_log.close()
print("Training loop completed")
