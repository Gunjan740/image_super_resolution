import os
from PIL import Image
from utils import make_lr

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

img_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "sample",
    "hr",
    "0001.png"
)

hr = Image.open(img_path).convert("RGB")

lr, hr_cropped = make_lr(hr, scale=4)

print("HR cropped size:", hr_cropped.size)
print("LR size:", lr.size)

print("HR % 8:", hr_cropped.size[0] % 8, hr_cropped.size[1] % 8)
print("LR % 8:", lr.size[0] % 8, lr.size[1] % 8)

lr.save(os.path.join(PROJECT_ROOT, "data", "sample", "lr_test.png"))
