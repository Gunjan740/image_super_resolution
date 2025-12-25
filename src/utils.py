import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")

def calculate_psnr(original, generated):
    original_np = np.array(original)
    generated_np = np.array(generated)
    
    if original_np.shape != generated_np.shape:
        generated_np = np.array(generated.resize(original.size))
    
    return psnr(original_np, generated_np, data_range=255)

def calculate_ssim(original, generated):
    original_np = np.array(original)
    generated_np = np.array(generated)
    
    if original_np.shape != generated_np.shape:
        generated_np = np.array(generated.resize(original.size))
    
    return ssim(original_np, generated_np, channel_axis=2, data_range=255)

def resize_image(image, scale_factor=2):
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, Image.BICUBIC)
