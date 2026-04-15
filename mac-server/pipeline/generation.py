"""
generation.py — SD 1.5 + LCM LoRA + ControlNet OpenPose + IP-Adapter FaceID

Identity stack:
  ControlNet:      face geometry (from ARKit blend shapes → OpenPose skeleton)
  IP-Adapter FaceID: face identity (from InsightFace embedding of camera frames)

MPS constraints:
  - torch_dtype=torch.float32 (float16 → black images on Apple Silicon)
  - NO enable_attention_slicing (destroys IPAdapterAttnProcessor)
  - guidance_scale=1.0 for LCM (0.0 is SDXL-Turbo's requirement)
"""

import os
import random
import time
import numpy as np
import torch
from PIL import Image

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LCMScheduler,
)
from pipeline.embeddings import make_static, noise_blend, NOISE_MIN
from config import (
    SD_MODEL_ID, CONTROLNET_MODEL_ID, LCM_LORA_ID,
    SD_STEPS, IMAGE_SIZE, GUIDANCE_SCALE,
    FIXED_PROMPT, NEGATIVE_PROMPT,
    IP_MIN, IP_MAX,
)


class GenerationPipeline:
    def __init__(self):
        self.device       = torch.device("mps")
        self.pipe         = None
        self.session_seed: int | None = None

    def setup(self):
        print("Loading ControlNet OpenPose...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float32,
        )

        print("Loading SD 1.5...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
        )

        # IP-Adapter FaceID: identity conditioning from 512-dim ArcFace embeddings
        print("Loading IP-Adapter FaceID...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sd15.bin",
            image_encoder_folder=None,
        )

        # LCM LoRA: 4-step fast inference
        print("Loading LCM LoRA...")
        self.pipe.load_lora_weights(LCM_LORA_ID, adapter_name="lcm")
        self.pipe.set_adapters(["lcm", "faceid_0"], adapter_weights=[1.0, 1.0])
        self.pipe.fuse_lora()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe = self.pipe.to(self.device)
        # DO NOT call enable_attention_slicing — destroys IPAdapterAttnProcessor

        print(f"Pipeline ready on {self.device}.")

    def new_session(self):
        self.session_seed = random.randint(0, 2 ** 32)
        print(f"[session]  seed={self.session_seed}")

    def generate(self,
                 conditioning_image: Image.Image,
                 morph_weight: float,
                 embedding: np.ndarray | None,
                 confidence: float) -> tuple[Image.Image, float]:
        """
        Run one SD generation.

        conditioning_image: OpenPose skeleton from blend shapes (512×512 PIL)
        morph_weight:       ControlNet scale [0,1] — face geometry influence
        embedding:          512-dim ArcFace embedding, or None if no face yet
        confidence:         [0,1] — how stable the embedding is

        Returns (output_image_with_noise_overlay, elapsed_seconds)
        """
        if self.session_seed is None:
            self.new_session()

        # IP-Adapter scale rises with confidence; zero embedding has no effect
        ip_scale = IP_MIN + confidence * (IP_MAX - IP_MIN) if embedding is not None else 0.0
        self.pipe.set_ip_adapter_scale(ip_scale)

        # Prepare face embedding tensor — always required when IP-Adapter is loaded.
        # Zero embedding + scale=0.0 when no face detected → pure ControlNet+text.
        if embedding is not None:
            face_t = torch.from_numpy(embedding).float().unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            face_t = torch.zeros(1, 1, 512, dtype=torch.float32, device=self.device)
        ip_embeds = [face_t]

        generator = torch.Generator(device=self.device).manual_seed(self.session_seed)

        t0 = time.time()
        result = self.pipe(
            prompt=FIXED_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=conditioning_image,
            num_inference_steps=SD_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            controlnet_conditioning_scale=morph_weight,
            ip_adapter_image_embeds=ip_embeds,
            generator=generator,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        elapsed = time.time() - t0

        # Apply noise overlay: fades as confidence builds, never fully disappears
        noise_alpha = max(NOISE_MIN, 1.0 - confidence)
        output = noise_blend(result.images[0], noise_alpha)

        return output, elapsed
