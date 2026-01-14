import shutil
from pathlib import Path

# -------- paths --------
div2k_dir = Path("~/datasets/DIV2K/DIV2K_train_HR").expanduser()
flickr_dir = Path("~/datasets/Flickr2K/Flickr2K_HR").expanduser()
out_dir = Path("~/datasets/DF2K/DF2K_HR").expanduser()

out_dir.mkdir(parents=True, exist_ok=True)

count = 0

# -------- copy DIV2K --------
for img in sorted(div2k_dir.glob("*.png")):
    dst = out_dir / f"DIV2K_{img.name}"
    shutil.copy2(img, dst)
    count += 1

print(f"Copied {count} DIV2K images")

# -------- copy Flickr2K --------
flickr_count = 0
for img in sorted(flickr_dir.glob("*.png")):
    dst = out_dir / f"FLICKR_{img.name}"
    shutil.copy2(img, dst)
    flickr_count += 1

print(f"Copied {flickr_count} Flickr2K images")
print(f"Total DF2K images: {count + flickr_count}")
