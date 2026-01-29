#!/bin/bash
#SBATCH --job-name=controlnet_sr_eval
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started on $(hostname)"
echo "Time: $(date)"

# -------------------------
# Load modules
# -------------------------
module purge
module load devel/python/3.11.7-gnu-14.2
module load devel/cuda/11.8

# -------------------------
# Activate virtualenv
# -------------------------
source ~/image_super_resolution/venv/bin/activate

# -------------------------
# Hugging Face token (safe, local-only)
# -------------------------
export HF_TOKEN=$(cat ~/.hf_token)

# -------------------------
# Hugging Face cache (persistent)
# -------------------------
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export DIFFUSERS_CACHE=$SCRATCH/hf_cache/diffusers

# -------------------------
# Go to project root
# -------------------------
cd ~/image_super_resolution

# -------------------------
# Sanity: check GPU
# -------------------------
nvidia-smi

# -------------------------
# Run evaluation
# -------------------------
python src/eval.py

echo "Job finished at $(date)"
