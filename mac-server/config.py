import os
from dotenv import load_dotenv
load_dotenv()

# Server
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# Models — SD 1.5 + LCM LoRA + OpenPose ControlNet (float32 on MPS)
SD_MODEL_ID         = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_openpose"
LCM_LORA_ID         = "latent-consistency/lcm-lora-sdv1-5"

# Generation
SD_STEPS   = int(os.getenv("SD_STEPS", "4"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
# LCM requires guidance_scale=1.0 (not 0.0 — that's SDXL-Turbo)
GUIDANCE_SCALE = 1.0

FIXED_PROMPT = (
    "photorealistic human face portrait, neutral expression, "
    "soft studio lighting, symmetrical, sharp focus, "
    "detailed skin texture, front facing, close up, cinematic"
)
NEGATIVE_PROMPT = (
    "cartoon, anime, painting, blurry, distorted, "
    "deformed, bad anatomy, watermark, text, body, clothes, "
    "multiple faces, extra eyes, artifacts"
)

# Morph
MORPH_ADVANCE_INTERVAL_SECONDS = float(os.getenv("MORPH_INTERVAL", "15"))
MORPH_INCREMENT                = float(os.getenv("MORPH_INCREMENT", "0.018"))

# JPEG quality for WebSocket transmission
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
