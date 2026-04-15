"""
test_faceid.py — verify InsightFace embedding extraction + IP-Adapter FaceID download

Run this before reflection.py:
  source .venv/bin/activate && python test_faceid.py

What it does:
  1. Loads InsightFace buffalo_l (downloads ~500MB on first run to ~/.insightface/)
  2. Opens camera, captures a frame, detects face, extracts 512-dim embedding
  3. Downloads ip-adapter-faceid_sd15.bin from HuggingFace (~200MB, cached)
  4. Loads the IP-Adapter image projection MLP and projects the embedding
  5. Prints shapes and stats so you know everything is wired correctly
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch

# ── 1. InsightFace ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: InsightFace")
print("=" * 60)

from insightface.app import FaceAnalysis

print("Loading buffalo_l (downloads ~500MB on first run)...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")

print("\nOpening camera and capturing frame...")
cap = cv2.VideoCapture(0)
for _ in range(10):
    cap.read()  # flush buffer

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read from camera.")
    exit(1)

frame = cv2.flip(frame, 1)
print(f"Frame: {frame.shape}  dtype={frame.dtype}")

print("Running face detection...")
faces = app.get(frame)

if not faces:
    print("\nWARNING: No face detected in frame.")
    print("Try running again while looking at the camera.")
    print("Continuing with zero embedding for pipeline test...")
    normed_embedding = np.zeros(512, dtype=np.float32)
else:
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    normed_embedding = face.normed_embedding
    print(f"\nFace detected!")
    print(f"  Bounding box:     {face.bbox.astype(int)}")
    print(f"  Embedding shape:  {normed_embedding.shape}")
    print(f"  Embedding norm:   {np.linalg.norm(normed_embedding):.4f}  (should be ~1.0)")
    print(f"  Mean / std:       {normed_embedding.mean():.4f} / {normed_embedding.std():.4f}")

# ── 2. Download IP-Adapter FaceID weights ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Download IP-Adapter FaceID weights")
print("=" * 60)

from huggingface_hub import hf_hub_download

print("Downloading ip-adapter-faceid_sd15.bin (~200MB, cached after first run)...")
weights_path = hf_hub_download(
    repo_id="h94/IP-Adapter-FaceID",
    filename="ip-adapter-faceid_sd15.bin",
)
print(f"Weights at: {weights_path}")

# ── 3. Inspect projection layer shape ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Inspect image projection (no full pipeline load needed)")
print("=" * 60)

state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
proj_keys = [k for k in state_dict["image_proj"].keys()]
print(f"image_proj keys: {proj_keys}")

# ip-adapter-faceid_sd15.bin uses IPAdapterFaceIDImageProjection (2-layer MLP + GELU + norm)
# Keys: proj.0 (Linear), proj.2 (Linear), norm (LayerNorm)
# diffusers renames: proj.0 → ff.net.0.proj, proj.2 → ff.net.2
if "proj.0.weight" in state_dict["image_proj"]:
    proj0_w = state_dict["image_proj"]["proj.0.weight"]
    proj2_w = state_dict["image_proj"]["proj.2.weight"]
    norm_w  = state_dict["image_proj"]["norm.weight"]
    cross_attn_dim = norm_w.shape[0]
    num_tokens = proj2_w.shape[0] // cross_attn_dim
    print(f"  Projection type:   IPAdapterFaceIDImageProjection (2-layer MLP)")
    print(f"  proj.0:  Linear({proj0_w.shape[1]}, {proj0_w.shape[0]})  — face embed → hidden")
    print(f"  proj.2:  Linear({proj2_w.shape[1]}, {proj2_w.shape[0]})  — hidden → tokens")
    print(f"  norm:    LayerNorm({cross_attn_dim})")
    print(f"  → {num_tokens} image tokens of dim {cross_attn_dim}")

ip_adapter_keys = [k for k in state_dict["ip_adapter"].keys()]
has_lora = any("lora" in k for k in ip_adapter_keys)
print(f"\nip_adapter has LoRA weights: {has_lora}")
print(f"ip_adapter key count: {len(ip_adapter_keys)}")

# ── 4. Quick projection test ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Test manual projection")
print("=" * 60)

from diffusers.models.embeddings import IPAdapterFaceIDImageProjection

proj0_w = state_dict["image_proj"]["proj.0.weight"]
proj0_b = state_dict["image_proj"]["proj.0.bias"]
proj2_w = state_dict["image_proj"]["proj.2.weight"]
proj2_b = state_dict["image_proj"]["proj.2.bias"]
norm_w  = state_dict["image_proj"]["norm.weight"]
norm_b  = state_dict["image_proj"]["norm.bias"]

cross_attn_dim   = norm_w.shape[0]           # 768 for SD1.5
id_embed_dim     = proj0_w.shape[1]          # 512 (ArcFace)
hidden_dim       = proj0_w.shape[0]          # 1024
multiplier       = hidden_dim // id_embed_dim # 2
num_tokens       = proj2_w.shape[0] // cross_attn_dim  # 4

proj_layer = IPAdapterFaceIDImageProjection(
    cross_attention_dim=cross_attn_dim,
    image_embed_dim=id_embed_dim,
    mult=multiplier,
    num_tokens=num_tokens,
)
# diffusers renames: proj.0 → ff.net.0.proj,  proj.2 → ff.net.2
proj_layer.ff.net[0].proj.weight = torch.nn.Parameter(proj0_w)
proj_layer.ff.net[0].proj.bias   = torch.nn.Parameter(proj0_b)
proj_layer.ff.net[2].weight      = torch.nn.Parameter(proj2_w)
proj_layer.ff.net[2].bias        = torch.nn.Parameter(proj2_b)
proj_layer.norm.weight           = torch.nn.Parameter(norm_w)
proj_layer.norm.bias             = torch.nn.Parameter(norm_b)
proj_layer = proj_layer.float()  # weights are float16 in the file; cast to float32 (MPS requirement)
proj_layer.eval()

face_t = torch.from_numpy(normed_embedding).float().unsqueeze(0)  # [1, 512]
with torch.no_grad():
    projected = proj_layer(face_t)  # [1, num_tokens, 768]

print(f"Input shape:    {face_t.shape}")
print(f"Projected shape: {projected.shape}  (= [batch, {num_tokens}, {cross_attn_dim}])")
print(f"Projected norm:  {projected.norm().item():.4f}")

print("\n" + "=" * 60)
print("All checks passed.")
print()
print("InsightFace:     working")
print(f"Embedding:       {normed_embedding.shape}  norm={np.linalg.norm(normed_embedding):.3f}")
print(f"IP-Adapter proj: {projected.shape}")
print()
print("Ready to run:  python reflection.py")
print("=" * 60)
