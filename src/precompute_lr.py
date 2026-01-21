import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from utils import make_lr   # same function used in training


# -------------------------
# Config
# -------------------------
# use DF2k to train data
HR_SRC = Path("~/datasets/DF2K/DF2K_HR").expanduser()
HR_OUT = Path("~/datasets/DF2K/DF2K_HR_1024").expanduser()
LR_OUT = Path("~/datasets/DF2K/DF2K_LR_x8").expanduser()


# test data 
#HR_SRC = Path("~/datasets/Set14_HR").expanduser()
#HR_OUT = Path("~/datasets/test_data/Set14_HR_1024").expanduser()
#LR_OUT = Path("~/datasets/test_data/Set14_LR_x4").expanduser()

HR_SIZE = 1024
SCALE =8

HR_OUT.mkdir(parents=True, exist_ok=True)
LR_OUT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Processing
# -------------------------
files = sorted(
    f for f in HR_SRC.iterdir()
    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
)

print(f"Found {len(files)} HR images")

for i, hr_path in enumerate(files):
    img = Image.open(hr_path).convert("RGB")

    # EXACT SAME resize as dataset.py
    img = TF.resize(
        img,
        (HR_SIZE, HR_SIZE),
        interpolation=Image.BICUBIC,
    )

    # EXACT SAME LR generation
    lr, hr = make_lr(img, scale=SCALE)

    # Save
    if not (HR_OUT / hr_path.name).exists():
        hr.save(HR_OUT / hr_path.name)

    #hr.save(HR_OUT / hr_path.name)
    lr.save(LR_OUT / hr_path.name)

    if (i + 1) % 100 == 0:
        print(f"[{i+1}/{len(files)}] processed")

print(" generation complete")

