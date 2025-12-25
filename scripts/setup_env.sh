#!/bin/bash

echo "Setting up environment for Image Super-Resolution project..."

module load python/3.10
module load cuda/11.8

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete!"
echo "Don't forget to:"
echo "1. Copy .env.example to .env"
echo "2. Add your HUGGINGFACE_TOKEN to .env"
echo "3. Run download_models.sh to download pretrained weights"
