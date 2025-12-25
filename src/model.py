import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from config import Config

class SuperResolutionModel:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        self.controlnet = None
        self.pipeline = None
        
    def load_models(self):
        print("Loading ControlNet model...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.config.CONTROLNET_MODEL,
            torch_dtype=torch.float16,
            use_auth_token=self.config.HUGGINGFACE_TOKEN
        )
        
        print("Loading Stable Diffusion pipeline...")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.MODEL_NAME,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_auth_token=self.config.HUGGINGFACE_TOKEN
        )
        
        self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()
        
        print("Models loaded successfully!")
        
    def upscale_image(self, image, prompt="high quality, detailed", negative_prompt="blurry, low quality"):
        if self.pipeline is None:
            raise ValueError("Models not loaded. Call load_models() first.")
            
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=self.config.NUM_INFERENCE_STEPS,
            guidance_scale=self.config.GUIDANCE_SCALE,
        ).images[0]
        
        return output
