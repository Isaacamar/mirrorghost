"""
generation.py — SD 1.5 + LCM LoRA + ControlNet OpenPose

Key constraints for MPS (Apple Silicon):
  - torch_dtype=torch.float32 everywhere (float16 → black images on MPS)
  - NO enable_attention_slicing — it replaces attention processors and breaks the pipeline
  - guidance_scale=1.0 for LCM (not 0.0, which is SDXL-Turbo's requirement)
"""

import os
import random
import time
import torch
from PIL import Image

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LCMScheduler,
)
from config import (
    SD_MODEL_ID, CONTROLNET_MODEL_ID, LCM_LORA_ID,
    SD_STEPS, IMAGE_SIZE, GUIDANCE_SCALE,
    FIXED_PROMPT, NEGATIVE_PROMPT,
)


class GenerationPipeline:
    def __init__(self):
        self.device       = torch.device("mps")
        self.pipe         = None
        self.session_seed: int | None = None

    def setup(self):
        """Load models. Blocks for 30-90s on first run (downloading weights)."""
        print("Loading ControlNet OpenPose...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float32,   # float32 — MPS requirement
        )

        print("Loading SD 1.5...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.float32,   # float32 — MPS requirement
            safety_checker=None,
        )

        # LCM LoRA: fast inference in 4 steps
        print("Loading LCM LoRA...")
        self.pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
        self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        # Move to MPS AFTER fusing LoRA
        self.pipe = self.pipe.to(self.device)
        # DO NOT call enable_attention_slicing() — destroys ControlNet attention processors

        print(f"Pipeline ready on {self.device}.")

    def new_session(self):
        """Call on each new client connection to reset the base identity."""
        self.session_seed = random.randint(0, 2 ** 32)
        print(f"[session]  seed={self.session_seed}")

    def generate(self,
                 conditioning_image: Image.Image,
                 morph_weight: float) -> tuple[Image.Image, float]:
        """
        Run one SD generation.
        conditioning_image: OpenPose skeleton PIL image (512×512)
        morph_weight: ControlNet conditioning scale [0.0, 1.0]
        Returns (PIL Image, elapsed_seconds)
        """
        if self.session_seed is None:
            self.new_session()

        generator = torch.Generator(device=self.device).manual_seed(self.session_seed)

        t0 = time.time()
        result = self.pipe(
            prompt=FIXED_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=conditioning_image,
            num_inference_steps=SD_STEPS,
            guidance_scale=GUIDANCE_SCALE,            # 1.0 for LCM
            controlnet_conditioning_scale=morph_weight,
            generator=generator,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        elapsed = time.time() - t0
        return result.images[0], elapsed
