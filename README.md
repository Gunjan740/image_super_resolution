# Image Super-Resolution with ControlNet

A ControlNet-based image super-resolution pipeline using Stable Diffusion v1.5 with the tile ControlNet (`lllyasviel/control_v11f1e_sd15_tile`). The model is conditioned on a low-resolution (LR) image and optionally guided by a text prompt describing either the scene semantics or the low-level texture of the LR input.

---

## Setup

```bash
conda create -n sr python=3.10 -y
conda activate sr
pip install -r requirements.txt
```

> Requires CUDA-capable GPU for training and evaluation. Caption generation (LLaVA) also requires a GPU.

---

## Project Structure

```
src/
├── dataset.py                     # On-the-fly LR/HR dataset
├── dataset_precomputed.py         # Dataset for precomputed LR/HR pairs
├── precompute_lr.py               # Precompute LR images from HR source
├── generate_semantic_captions.py  # Generate scene-level text captions (LLaVA)
├── generate_texture_captions.py   # Generate texture-level text captions (LLaVA)
├── merge_caption_chunks.py        # Merge chunked caption files into one JSONL
├── train_skeleton.py              # Training script (supports all prompt modes)
├── eval.py                        # Evaluation script (PSNR, SSIM, LPIPS)
└── utils.py                       # Shared utilities (make_lr, crop_to_multiple)
```

---

## Data

### Datasets

Training uses the **DF2K** dataset (DIV2K + Flickr2K combined). Evaluation uses standard SR benchmarks such as **Set5**, **Set14**, and **DIV2K validation**.

All the HR images were taken from these places:
**DIV2K Train and Validation**: https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
**Flickr2K**: https://huggingface.co/datasets/yangtao9009/Flickr2K/tree/main
**Set5**: https://huggingface.co/datasets/eugenesiow/Set5/tree/main
**Set14**: https://huggingface.co/datasets/eugenesiow/Set14/tree/main

Expected directory layout:

```
~/datasets/
├── DF2K/
│   ├── DF2K_HR_1024/           # HR images resized to 1024×1024
│   ├── DF2K_LR_x4/             # LR images at 256×256 (4× downscale)
│   └── df2k_LR_x4_texture_captions.jsonl
├── 
│
└── test_data/
    ├── Set5_HR_512/
    ├── Set5_512_LR_x4/
    └── DIV2K_valid_HR_512/ and the semantic captions for these images are stored in `DIV2K_valid_HR_512_semantic_captions.jsonl`

```

---

## Dataset Classes

### `dataset.py` — `SRDataset`

On-the-fly dataset that loads raw HR images and generates LR/HR pairs at runtime.

- Takes a directory of HR images.
- Resizes each HR image to `hr_size × hr_size` using bicubic interpolation.
- Generates the LR image using `make_lr` (bicubic downscale by `scale`, default 4×).
- Returns `(lr_tensor, hr_tensor)`, both normalized to `[-1, 1].

**Use this** when you want to avoid storing precomputed pairs on disk. In this work we first resized the HR images to 1024x1024 (512x512 for evaluation) and then generated the LR images and stored them in a separate directory using precompute_lr.py.

### `dataset_precomputed.py` — `SRDatasetPrecomputed`

Dataset for precomputed LR/HR image pairs stored on disk.

- Takes separate `hr_dir` and `lr_dir` directories. Filenames must match exactly across both.
- Assumes HR images are already 1024×1024 (or 512×512 for evaluation) and LR images are already downscaled.
- Returns `(lr_tensor, hr_tensor, filename)`, normalized to `[-1, 1]`.

**Use this** for training and evaluation to avoid repeated on-the-fly processing overhead.

---

## Precomputing LR Images

### `precompute_lr.py`

Precomputes and saves LR/HR image pairs from a raw HR source directory. This ensures training and evaluation use identical preprocessing to `SRDataset`.

**What it does:**
1. Reads HR images from `HR_SRC`.
2. Resizes each to `HR_SIZE × HR_SIZE` using bicubic interpolation — identical to `SRDataset`.
3. Applies `make_lr` to generate the LR image at `1/SCALE` resolution.
4. Saves the resized HR to `HR_OUT` and the LR to `LR_OUT`.

**Configure** the paths and size at the top of the script before running:

```python
HR_SRC  = Path("~/datasets/DIV2K/DIV2K_valid_HR").expanduser()
HR_OUT  = Path("~/datasets/test_data/DIV2K_valid_HR_512").expanduser()
LR_OUT  = Path("~/datasets/test_data/DIV2K_valid_512_LR_x4").expanduser()
HR_SIZE = 512
SCALE   = 4
```

```bash
python src/precompute_lr.py
```

---

## Generating Text Prompts

Prompts are generated from the **LR images** using LLaVA 1.5 (7B). Two prompt types are supported.

### `generate_semantic_captions.py` — Scene-Level Prompts

Generates a short factual sentence describing the main objects and scene of the LR image.

- Instruction: *"Describe the image in one short, factual sentence. Focus on the main objects and the overall scene."*
- Output: JSONL files (one per chunk) with fields `file`, `caption_raw`, `caption_clean` (identical for semantic captions).

```bash
python src/generate_semantic_captions.py
```

Configure `IMAGE_DIR`, `OUT_DIR`, and `NUM_CHUNKS` at the top of the script.

**SLURM job** (1 GPU, LLaVA inference):
sbatch generate_semantic_captions.slurm

Then call `merge_caption_chunks.py` once all chunks finish.

### `generate_texture_captions.py` — Texture-Level Prompts

Generates a description focused purely on low-level visual properties (texture, edges, patterns, noise, blur) — **without** mentioning objects or scene content.

- Instruction guides the model to describe texture, edge sharpness, patterns, and contrast only.
- Includes a **semantic leakage filter**: if the generated caption contains object/scene words (e.g., "road", "person", "tree"), it is rewritten using extracted texture attributes.
- Output: JSONL with `file`, `caption_raw` (raw model output), `caption_clean` (filtered/rewritten caption).

```bash
python src/generate_texture_captions.py
```

Configure `IMAGE_DIR`, `OUT_DIR`, and `NUM_CHUNKS` at the top of the script.

**SLURM job** (1 GPU, LLaVA inference):
sbatch generate_texture_captions.slurm

Same chunked array job pattern applies as for semantic captions.

### `merge_caption_chunks.py`

After generating captions in chunks (e.g., across multiple GPUs or runs), merge them into a single JSONL file:

```bash
python src/merge_caption_chunks.py
```

---

## Training

### Architecture

- **Backbone**: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- **Control signal**: `lllyasviel/control_v11f1e_sd15_tile` ControlNet
- **Trainable**: ControlNet only — VAE, UNet, and text encoder are frozen
- **Conditioning**: The LR image is bilinearly upsampled to the latent spatial size and passed as the ControlNet condition
- **Objective**: MSE loss on predicted vs. actual noise (standard diffusion denoising objective)

### `train_skeleton.py`

Single unified training script. The `PROMPT_MODE` variable at the top selects which type of text conditioning to use.

#### Mode A — No Prompt

```python
PROMPT_MODE = "none"
```

The text prompt is always an empty string. The model learns to super-resolve using only the LR image as the ControlNet condition, with no text guidance.

#### Mode B — Semantic Prompt

```python
PROMPT_MODE = "lr_semantic"
CAPTIONS_JSONL = "~/datasets/DF2K/df2k_LR_x4_semantic_captions.jsonl"
```

Each image is paired with its scene-level caption generated by `generate_semantic_captions.py`. The caption is encoded by the frozen CLIP text encoder and injected into the UNet via cross-attention.

#### Mode C — Texture Prompt

```python
PROMPT_MODE = "lr_texture"
CAPTIONS_JSONL = "~/datasets/DF2K/df2k_LR_x4_texture_captions.jsonl"
```

Each image is paired with its texture-level caption generated by `generate_texture_captions.py`. This guides the model to hallucinate perceptually plausible high-frequency detail consistent with the LR image's texture properties.

### Running Training

**SLURM job** (1 GPU, long-running):
sbatch train_skeleton_large_gpu.slurm or for testing sbatch train_skeleton.slurm

Training resumes automatically from `latest.pt` if the job is requeued — just resubmit the same script.

Checkpoints are saved to `CKPT_DIR` (default `checkpoints_LR_texture/`). Training automatically resumes from `latest.pt` if it exists. Loss is logged to `logs/train_loss_LR_texture.csv`.

---

## Evaluation

### `eval.py`

Evaluates a trained checkpoint against a test dataset. Computes metrics for both the ControlNet SR output and bicubic upsampling (baseline):

- **PSNR** and **SSIM** on the Y (luminance) channel with a 4-pixel border crop
- **LPIPS** (AlexNet) for perceptual quality

**Configure** at the top of the script:

```python
HR_DIR         = "~/datasets/test_data/Set5_HR_512"
LR_DIR         = "~/datasets/test_data/Set5_512_LR_x4"
CKPT_PATH      = "checkpoints/eval_latest_semantic.pt"
PROMPT_MODE    = "lr_semantic"   # or "lr_texture" or "none"
CAPTIONS_JSONL = "~/datasets/test_data/Set5_512_LR_x4_semantic_captions.jsonl"
OUT_DIR        = "results/semantic/Set5_HR_512"
```


**SLURM job** (1 GPU):
sbatch eval.slurm

Evaluation is resumable — if the job is interrupted, resubmit and it will continue from the last completed image via `progress.txt`.

**Outputs** (written to `OUT_DIR`):

| File | Contents |
|---|---|
| `results.txt` | Per-image PSNR, SSIM, LPIPS for SR and bicubic |
| `avg_metrics.txt` | Dataset-level averages |
| `grids/` | Side-by-side images: `[bicubic \| SR \| HR]` |
| `progress.txt` | Last completed index (allows resumable evaluation) |
