import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')
    
    MODEL_NAME = os.getenv('MODEL_NAME', 'stabilityai/stable-diffusion-2-1')
    CONTROLNET_MODEL = os.getenv('CONTROLNET_MODEL', 'lllyasviel/control_v11f1e_sd15_tile')
    
    DATA_DIR = os.getenv('DATA_DIR', './data')
    MODEL_DIR = os.getenv('MODEL_DIR', './models')
    RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
    LOG_DIR = os.getenv('LOG_DIR', './logs')
    
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-5'))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '10'))
    
    GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))
    NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '50'))
    
    DEVICE = 'cuda'
    SEED = 42
