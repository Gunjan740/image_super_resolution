import os
from PIL import Image

img_dir = "data/sample/hr"

for fname in os.listdir(img_dir):
    path = os.path.join(img_dir, fname)
    img = Image.open(path).convert("RGB")
    print(fname, img.size)
