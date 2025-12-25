import argparse
import os
from model import SuperResolutionModel
from utils import load_image, save_image, calculate_psnr, calculate_ssim
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Image Super-Resolution Inference')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='./data/output/', help='Output directory')
    parser.add_argument('--prompt', type=str, default='high quality, detailed, sharp', help='Prompt for generation')
    parser.add_argument('--reference', type=str, default=None, help='Reference high-res image for metrics')
    
    args = parser.parse_args()
    
    config = Config()
    
    print("Initializing Super-Resolution Model...")
    model = SuperResolutionModel()
    model.load_models()
    
    print(f"Loading input image: {args.input}")
    input_image = load_image(args.input)
    
    print("Performing super-resolution...")
    output_image = model.upscale_image(input_image, prompt=args.prompt)
    
    output_filename = os.path.basename(args.input).replace('.', '_upscaled.')
    output_path = os.path.join(args.output, output_filename)
    save_image(output_image, output_path)
    
    if args.reference:
        print("Calculating metrics...")
        reference_image = load_image(args.reference)
        psnr_value = calculate_psnr(reference_image, output_image)
        ssim_value = calculate_ssim(reference_image, output_image)
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
    
    print("Inference complete!")

if __name__ == '__main__':
    main()
