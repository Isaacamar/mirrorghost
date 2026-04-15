# Reflection

A real-time generative face installation. A person stands in front of a camera. On screen, a face slowly builds — an AI impression that eerily resembles them but isn't them. The process is visible: noise resolves into features, identity accumulates over time.

Intended for projection onto a mannequin head.

---

## Current State

Working: `reflection.py`
- Camera → MediaPipe face crop → SD1.5 img2img → tkinter display
- Iterative: each generation feeds its own output back in
- Strength schedule: high early (rough), low later (refining)
- Camera blended in each pass to stay anchored to subject

**Core problem not yet solved:**
img2img preserves *structure* (rough face shape, lighting) but doesn't preserve *identity* (your specific nose, eye shape, jawline, skin tone). The output looks like "a face" not "your face."

---

## The Identity Problem

### How forensic artists actually reconstruct faces

Police composite systems (FACES, E-FIT, IDENTIKIT) work by decomposing the face into discrete features — eye shape, nose bridge width, lip thickness, jaw angle — and iterating with a witness until the composite converges on the target. The witness is the feedback loop.

In our system: **the camera is the witness. The face embedding is the composite data.**

### What we need: face identity embeddings

A face recognition model (ArcFace, FaceNet, InsightFace) compresses a face into a ~512-dimensional vector that encodes identity — not pixels, not structure, but the mathematical signature of WHO someone is. Two photos of the same person taken years apart will have nearly identical embeddings. A photo of their sibling will be close but distinct.

This embedding is what we need to feed into generation.

### The right technical stack

| Component | Purpose | Model |
|-----------|---------|-------|
| Face detection + crop | Isolate face from background | MediaPipe FaceLandmarker ✅ |
| Face identity embedding | Extract WHO the person is | InsightFace / ArcFace |
| Structure conditioning | Preserve face geometry | ControlNet OpenPose/Face |
| Identity-conditioned generation | Generate face that looks like them | IP-Adapter FaceID |

**IP-Adapter FaceID** is the key missing piece. It was specifically designed for this:
- Takes a face embedding as input (not a prompt, not an image)
- Injects identity into SD at the attention layer level
- Output looks like the person even with style changes

**InstantID** is even stronger — combines face embedding + ControlNet structure in one pass.

### The "noise resolving into your face" visual

Currently the noise-to-face visual comes from watching SD denoising steps.
The right version: start with pure noise, use IP-Adapter FaceID to guide denoising
toward the subject's identity. As more camera frames are captured and averaged,
the identity embedding becomes more confident → the face that emerges becomes
more specifically them.

Early frames: noisy embedding → face could be anyone
Later frames: stable embedding → unmistakably that person

---

## File Structure

```
reflection.py          — current working version (img2img iterative)
test_camera.py         — tests MediaPipe face landmark rendering, no ML
download_models.py     — pre-downloads SD1.5 + ControlNet weights
models/
  face_landmarker.task — MediaPipe face detection model
past_version/          — earlier architecture (ControlNet + full pipeline)
src/                   — earlier modular components (camera, features, display, generator)
.venv/                 — Python 3.12 virtualenv
requirements.txt       — dependencies
```

---

## Environment

- Hardware: M4 MacBook Pro 24GB RAM
- Python: 3.12 (venv at `.venv/`)
- Device: MPS (Apple Silicon GPU)
- **Important:** must use `float32` not `float16` — float16 produces black images on MPS
- **Important:** load ML models BEFORE initializing pygame/tkinter to avoid Metal GPU conflict
- Activate venv: `source .venv/bin/activate`

## Installed

```
torch (MPS, float32)
diffusers + transformers + accelerate + peft
mediapipe 0.10+ (Tasks API — NOT solutions API which was removed)
opencv-python-headless
pillow
```

## Models cached (~5.6GB)

```
~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5
~/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_openpose
~/.cache/huggingface/hub/models--latent-consistency--lcm-lora-sdv1-5
```

---

## Next Steps

1. **Add InsightFace** for face embedding extraction
2. **Add IP-Adapter FaceID** for identity-conditioned generation
3. **Accumulate embeddings** over time — average across frames for stability
4. **Confidence-gated strength** — noisier output when embedding is weak, sharper when stable
5. **Projection mapping** — map output to UV space for mannequin head projection

---

## Run

```bash
source .venv/bin/activate

# Test camera + face landmarks (no ML)
python test_camera.py

# Main experience
python reflection.py
```
