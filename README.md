# Image Super-Resolution using Stable Diffusion and ControlNet

A university project implementing image super-resolution using Stable Diffusion models with ControlNet for enhanced control and quality.

## Project Overview

This project uses pre-trained Stable Diffusion models combined with ControlNet to perform high-quality image super-resolution. The approach leverages diffusion models' generative capabilities to upscale low-resolution images while preserving details and adding realistic textures.

## Setup Instructions

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your HuggingFace token:
   ```bash
   cp .env.example .env
   # Edit .env and add your HUGGINGFACE_TOKEN
   ```

### Cluster Deployment

1. Transfer code to cluster:
   ```bash
   rsync -avz --exclude='data/' --exclude='models/' --exclude='results/' . username@cluster:/path/to/project/
   ```

2. On the cluster, run setup:
   ```bash
   bash scripts/setup_env.sh
   ```

3. Download pretrained models:
   ```bash
   bash scripts/download_models.sh
   ```

4. Submit job:
   ```bash
   sbatch scripts/run_job.slurm
   ```

## Usage

### Inference

```bash
python src/inference.py --input data/input/image.png --output data/output/
```

### Training (Fine-tuning)

```bash
python src/train.py --config src/config.py
```

### Demo Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

## Project Structure

- `data/` - Input images, output results, and samples
- `src/` - Source code for models, training, and inference
- `models/` - Pretrained model weights
- `results/` - Generated images and evaluation metrics
- `scripts/` - Cluster job scripts and utilities
- `logs/` - Training and job logs
- `notebooks/` - Jupyter notebooks for demonstration

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+
- Diffusers library
- ControlNet models

## Results

[Add your results, metrics (PSNR, SSIM), and sample outputs here]

## References

- Stable Diffusion: https://github.com/Stability-AI/stablediffusion
- ControlNet: https://github.com/lllyasviel/ControlNet
- Diffusers: https://github.com/huggingface/diffusers

## License

[Add your license here]
