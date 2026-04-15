import os
from dotenv import load_dotenv
load_dotenv()

# Server
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# Models
SD_MODEL_ID         = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_openpose"
LCM_LORA_ID         = "latent-consistency/lcm-lora-sdv1-5"

# Generation
SD_STEPS       = int(os.getenv("SD_STEPS", "4"))
IMAGE_SIZE     = int(os.getenv("IMAGE_SIZE", "512"))
GUIDANCE_SCALE = 1.0   # LCM: must be 1.0, not 0.0

FIXED_PROMPT = (
    "photorealistic human face portrait, studio lighting, "
    "sharp focus, detailed skin texture, neutral expression, "
    "front facing, close up, high resolution, cinematic"
)
NEGATIVE_PROMPT = (
    "cartoon, anime, painting, blurry, distorted, "
    "deformed, bad anatomy, watermark, text, body, clothes, "
    "multiple faces, extra eyes, artifacts, sunglasses, hat"
)

# IP-Adapter FaceID scales
IP_MIN = 0.25   # scale at zero confidence
IP_MAX = 0.85   # scale at full confidence

# Morph (ControlNet conditioning scale)
MORPH_ADVANCE_INTERVAL_SECONDS = float(os.getenv("MORPH_INTERVAL", "15"))
MORPH_INCREMENT                = float(os.getenv("MORPH_INCREMENT", "0.018"))

# JPEG quality for WebSocket transmission
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
