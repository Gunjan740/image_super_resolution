import os
from dataset import SRDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
HR_DIR = os.path.join(PROJECT_ROOT, "data", "sample", "hr")

dataset = SRDataset(HR_DIR, scale=4)

print("Dataset length:", len(dataset))

lr, hr = dataset[0]

print("HR size:", hr.size)
print("LR size:", lr.size)

print("HR % 8:", hr.size[0] % 8, hr.size[1] % 8)
print("LR % 8:", lr.size[0] % 8, lr.size[1] % 8)
