#!/bin/bash

echo "Downloading pretrained models..."

source venv/bin/activate

python -c "
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

print('Downloading Stable Diffusion model...')
snapshot_download(
    repo_id='stabilityai/stable-diffusion-2-1',
    cache_dir='./models',
    token=token
)

print('Downloading ControlNet model...')
snapshot_download(
    repo_id='lllyasviel/control_v11f1e_sd15_tile',
    cache_dir='./models',
    token=token
)

print('Models downloaded successfully!')
"

echo "Model download complete!"
