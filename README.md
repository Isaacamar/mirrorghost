# mirrorghost

A real-time generative face installation. A person stands in front of a camera. On screen, their face is continuously reconstructed — recognizable but wrong, always mid-process, never resolving. Intended for projection onto a mannequin head.

---

## What it does

The system runs two feedback loops simultaneously:

**Identity loop (slow):** InsightFace extracts a 512-dim face embedding every 0.5s. A rolling average accumulates over 20+ readings. Confidence builds over minutes. IP-Adapter FaceID injects this identity into every generation — the face becomes more specifically *you* as the session runs.

**Generation loop (continuous):** SD 1.5 + LCM LoRA runs img2img with no interval. Each output feeds directly into the next generation's init image, blended with the live camera frame. Every denoising step is decoded and pushed to a 60fps display that smoothly interpolates between steps. You see the model working — not discrete outputs, continuous transformation.

TV static overlays the output at a level inverse to confidence. At zero confidence: pure static. At full confidence: a minimum residual grain that never fully disappears.

---

## Versions

| File | What it is |
|------|-----------|
| `reflection.py` | **Current.** Standalone webcam version. |
| `mac-server/` | WebSocket server for iPad TrueDepth integration. |
| `ipad-app/` | Swift/ARKit iPad client (import into Xcode). |

---

## Hardware

- MacBook Pro M4 (24GB) — all inference runs on MPS (Apple Silicon GPU)
- Webcam or built-in camera — for standalone `reflection.py`
- iPad Pro M4 — for TrueDepth integration (optional, separate setup)

---

## Environment

```bash
# Python 3.12 venv at .venv/
source .venv/bin/activate
```

All models load from `~/.cache/huggingface/hub/` and `~/.insightface/`. First run downloads ~6GB. Subsequent runs are instant.

**Critical constraints:**
- `torch_dtype=torch.float32` — float16 produces black images on MPS
- No `enable_attention_slicing()` — destroys IPAdapterAttnProcessor at runtime
- `guidance_scale=1.0` — LCM requires this (0.0 is SDXL-Turbo's requirement)
- Load all ML models **before** initializing tkinter — avoids Metal GPU conflict

---

## Run

### Standalone (webcam)

```bash
source .venv/bin/activate
python reflection.py
```

| Key | Action |
|-----|--------|
| `R` | Noise flood — resets embedding accumulator and canvas |
| `Q` / `Escape` | Quit |

### Mac server (for iPad)

```bash
source .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-dotenv   # first time only
cd mac-server
python main.py
```

Server prints the Mac's local IP on startup. Enter that IP in the iPad app.

---

## Tuning

Key parameters in `reflection.py`:

```python
STRENGTH    = 0.55   # how hard SD transforms each frame [0.3–0.8]
                     # 0.3 = subtle drift, nearly stable, quietly wrong
                     # 0.55 = clear morphing, recognizably you but uncanny
                     # 0.8 = hallucinatory, barely remembers the last frame

CAM_WEIGHT  = 0.25   # camera bleed into init image [0–1]
                     # 0 = pure previous output (can drift from reality)
                     # 0.4 = strongly anchored to your actual face shape
                     # 1 = pure vid2vid, re-anchors every generation to camera

NOISE_MIN   = 0.08   # minimum static at full confidence
                     # raise to keep more grain even at full confidence

CONFIDENCE_FULL = 20 # embeddings needed for 100% confidence (~10s at default rate)
```

---

## Architecture

```
Camera (60fps)
  │
  ├─ embedding_loop (every 0.5s)
  │    InsightFace buffalo_l → 512-dim normed embedding
  │    EmbeddingAccumulator → rolling average, confidence score
  │
  └─ generation_loop (continuous, no interval)
       init_img = blend(last_output, camera_frame, CAM_WEIGHT)
       SD 1.5 img2img + LCM LoRA (4–6 steps)
       IP-Adapter FaceID (scale rises with confidence)
         │
         └─ callback_on_step_end
              decode latent → PIL image
              noise_blend(step_img, step_noise)
              push_step() → interpolation buffer
                │
                └─ tick() at 60fps
                     Image.blend(step_a, step_b, alpha)
                     → tkinter display
```

---

## Models (cached, ~6.5GB total)

| Model | Size | Path |
|-------|------|------|
| `runwayml/stable-diffusion-v1-5` | ~4GB | `~/.cache/huggingface/hub/` |
| `h94/IP-Adapter-FaceID` | ~200MB | `~/.cache/huggingface/hub/` |
| `latent-consistency/lcm-lora-sdv1-5` | ~67MB | `~/.cache/huggingface/hub/` |
| `insightface buffalo_l` | ~500MB | `~/.insightface/models/` |

---

## iPad Integration

TrueDepth camera on iPad Pro sends 52 ARKit blend shapes + head euler angles over WebSocket to the Mac. The Mac renders using ControlNet OpenPose conditioned on a skeleton built from those blend shapes, with morph weight rising over ~15 minutes (0 = random face, 1 = your geometry).

### Xcode setup

1. New Xcode project — App, SwiftUI, iPad target
2. Add files from `ipad-app/Mirror/` (ARKit, App, Network, Views folders)
3. `Info.plist` → add `NSCameraUsageDescription`
4. Build & run on iPad Pro (TrueDepth required — simulator won't work)

### Protocol

```
iPad → Mac  (10fps)
  { type: "face_frame", blend_shapes: {...52 keys...}, head_euler: {pitch, yaw, roll} }

Mac → iPad
  { type: "face_frame", jpeg_b64: "...", morph_weight: 0.34, generation_ms: 4200 }
```

---

## Dependencies

```
torch (MPS, float32)
diffusers + transformers + accelerate + peft
insightface
opencv-python-headless
pillow
tkinter (stdlib)

# iPad server only:
fastapi + uvicorn[standard] + python-dotenv
```
