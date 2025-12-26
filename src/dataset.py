import os
from PIL import Image
from torch.utils.data import Dataset
from utils import make_lr

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=4):
        self.hr_dir = hr_dir
        self.scale = scale
        self.files = sorted(
            f for f in os.listdir(hr_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.files[idx])
        hr = Image.open(hr_path).convert("RGB")

        lr, hr = make_lr(hr, scale=self.scale)

        return lr, hr
