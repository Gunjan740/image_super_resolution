import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import make_lr


class SRDataset(Dataset):
    def __init__(
        self,
        hr_dir,
        scale=4,
        hr_size=1024,   # 🔑 FIXED HR SIZE
    ):
        self.hr_dir = hr_dir
        self.scale = scale
        self.hr_size = hr_size

        self.files = sorted(
            f for f in os.listdir(hr_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        # 🔑 SD expects [-1, 1]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),                 # [0, 1]
            transforms.Normalize([0.5]*3, [0.5]*3) # → [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.files[idx])
        hr = Image.open(hr_path).convert("RGB")

        # --------------------------------------------------
        # 🔑 FIX 1: deterministic resize (VERY IMPORTANT)
        # --------------------------------------------------
        hr = TF.resize(
            hr,
            (self.hr_size, self.hr_size),
            interpolation=Image.BICUBIC,
        )

        # --------------------------------------------------
        # Generate LR–HR pair
        # --------------------------------------------------
        lr, hr = make_lr(hr, scale=self.scale)

        # --------------------------------------------------
        # Convert to tensors in [-1, 1]
        # --------------------------------------------------
        lr = self.to_tensor(lr)
        hr = self.to_tensor(hr)

        return lr, hr

