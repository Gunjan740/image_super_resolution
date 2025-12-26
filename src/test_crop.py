from PIL import Image
from utils import crop_to_multiple

img = Image.open("data/sample/hr/0001.png").convert("RGB")
cropped = crop_to_multiple(img, multiple=8)

print("Original:", img.size)
print("Cropped:", cropped.size)
print("Divisible by 8:",
      cropped.size[0] % 8 == 0,
      cropped.size[1] % 8 == 0)
