#!/bin/bash
#SBATCH --job-name=super_res
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load python/3.10
module load cuda/11.8

source venv/bin/activate

nvidia-smi

echo "Running inference..."
python src/inference.py \
    --input data/input/test_image.png \
    --output data/output/ \
    --prompt "high quality, detailed, sharp"

echo "Job completed at: $(date)"
