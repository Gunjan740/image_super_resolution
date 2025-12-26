import os
import torch
from dotenv import load_dotenv
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)

# Load HF token

load_dotenv()
assert "HF_TOKEN" in os.environ, "HF_TOKEN not found in environment"

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cpu"

# --------------------------------------------------
# Load ControlNet (Tile)
# --------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile",
    torch_dtype=torch.float32,
)

# --------------------------------------------------
# Load Stable Diffusion + attach ControlNet
# --------------------------------------------------
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
)

pipe.to(device)

print("✅ Stable Diffusion + ControlNet (Tile) loaded")

# --------------------------------------------------
# Freeze Stable Diffusion U-Net
# --------------------------------------------------
for p in pipe.unet.parameters():
    p.requires_grad = False

# --------------------------------------------------
# Validate trainable parameters
# --------------------------------------------------
sd_trainable = sum(p.requires_grad for p in pipe.unet.parameters())
cn_trainable = sum(p.requires_grad for p in pipe.controlnet.parameters())

print(f"Trainable SD U-Net params: {sd_trainable}")
print(f"Trainable ControlNet params: {cn_trainable}")

assert sd_trainable == 0, "❌ SD U-Net is not frozen"
assert cn_trainable > 0, "❌ ControlNet has no trainable parameters"

print("✅ MODEL PIPELINE VALIDATION PASSED")
