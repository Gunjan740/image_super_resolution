import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRDatasetPrecomputed(Dataset):
    """
    Dataset  with precomputed HR and LR images.

    Assumptions:
    - HR images are already resized to 1024x1024
    - LR images are already generated (e.g., x4 downscale → 256x256)
    - Filenames in HR and LR directories match exactly
    - Output tensors are normalized to [-1, 1], same as SRDataset
    """

    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = os.path.expanduser(hr_dir)
        self.lr_dir = os.path.expanduser(lr_dir)

        self.files = sorted(
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.hr_dir}")

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        hr_path = os.path.join(self.hr_dir, fname)
        lr_path = os.path.join(self.lr_dir, fname)

        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"LR image not found: {lr_path}")

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        return lr, hr, fname
