import os
import torch
from PIL import Image
from dotenv import load_dotenv
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ---------------------------
# Load env + device
# ---------------------------
load_dotenv()
device = "cpu"

# ---------------------------
# Load models (same as validation)
# ---------------------------
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

# Freeze SD U-Net (as in training)
for p in pipe.unet.parameters():
    p.requires_grad = False

# ---------------------------
# Load ONE sample image
# ---------------------------
hr_path = "data/sample/hr/0001.png"
hr = Image.open(hr_path).convert("RGB")

# Resize HR to a manageable size for CPU (important)
hr = hr.resize((512, 512))

# Create LR (×4)
lr = hr.resize((128, 128), resample=Image.BICUBIC)

# ControlNet expects conditioning image at generation size
control_image = lr.resize((512, 512), resample=Image.BICUBIC)

# ---------------------------
# Run inference (no prompt)
# ---------------------------
with torch.no_grad():
    result = pipe(
        prompt="",
        image=control_image,
        num_inference_steps=20,
        guidance_scale=1.0
    ).images[0]

# ---------------------------
# Save output
# ---------------------------
os.makedirs("results", exist_ok=True)
result.save("results/sanity_sr_output.png")

print("✅ Inference sanity check completed")
print("Saved: results/sanity_sr_output.png")
