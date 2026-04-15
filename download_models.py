"""
Pre-download all model weights to Hugging Face cache.
Run once before first launch: .venv/bin/python download_models.py

Downloads ~5.6 GB total:
  - runwayml/stable-diffusion-v1-5          (~4.0 GB)
  - lllyasviel/control_v11p_sd15_openpose   (~1.4 GB)
  - latent-consistency/lcm-lora-sdv1-5      (~0.2 GB)
"""

import sys

def main():
    print("Downloading models (this runs once, ~5.6 GB total)...")
    print("Hugging Face cache: ~/.cache/huggingface/hub/\n")

    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler
    except ImportError:
        print("ERROR: dependencies not installed. Run: .venv/bin/pip install -r requirements.txt")
        sys.exit(1)

    print("[1/3] Downloading ControlNet openpose (~1.4 GB)...")
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float32,
    )
    print("      Done.\n")

    print("[2/3] Downloading Stable Diffusion v1.5 (~4.0 GB)...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float32,
        ),
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    print("      Done.\n")

    print("[3/3] Downloading LCM-LoRA (~0.2 GB)...")
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    print("      Done.\n")

    print("=" * 50)
    print("All models downloaded. You're ready to run:")
    print("  .venv/bin/python main.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
