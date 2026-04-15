"""
Reflection v3 — continuous img2img + 60fps step interpolation

Visual concept:
  The face is permanently mid-generation. It never resolves, never resets.
  Each denoising step is decoded and pushed to a display thread that runs
  at 60fps, smoothly interpolating between steps so the model's work
  appears as continuous flowing transformation — not discrete jumps.

  Two forces in tension:
    Camera input  — bleeds into every init image, anchors output to your face
    SD generation — continuously transforms it into something slightly wrong

  IP-Adapter FaceID pulls each generation toward your specific identity.
  TV static fades as face confidence builds, but never fully disappears.

  STRENGTH controls the character of the piece:
    0.3 = subtle per-frame drift, nearly stable, quietly wrong
    0.55 = clear transformation, morphing, recognizably you but uncanny
    0.8 = heavy, each frame barely remembers the last, hallucinatory

Controls: Q / Escape = quit    R = noise flood (reset)
"""

import os, time, threading, collections, random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler

# ── Config ─────────────────────────────────────────────────────────────────────
STRENGTH        = 0.55   # how much each generation transforms the init [0.3–0.8]
CAM_WEIGHT      = 0.25   # camera bleed into init [0=pure canvas, 1=pure camera]
EMBED_INTERVAL  = 0.5    # seconds between embedding extractions
MAX_EMBEDDINGS  = 60
CONFIDENCE_FULL = 20
NOISE_MIN       = 0.08   # minimum static at full confidence

SESSION_SEED    = random.randint(0, 2**32)

CAM  = 0
W, H = 800, 800

STEPS    = 6             # LCM steps; actual denoising = int(STEPS * STRENGTH) ≈ 3
IP_MIN   = 0.25          # IP-Adapter scale at zero confidence
IP_MAX   = 0.85          # IP-Adapter scale at full confidence
GUIDANCE = 1.0           # 1.0 = no CFG (LCM requirement — not 0.0)

PROMPT   = ("photorealistic human face portrait, studio lighting, "
            "sharp focus, detailed skin texture, neutral expression, "
            "front facing, close up, high resolution, cinematic")
NEGATIVE = ("cartoon, anime, painting, deformed, blurry, text, watermark, "
            "body, clothes, background, hat, sunglasses, multiple faces, "
            "bad anatomy, artifacts")


# ── Noise / static ─────────────────────────────────────────────────────────────
def make_static(size: int = 512) -> Image.Image:
    """TV static: grayscale base with chromatic fringing."""
    luma = np.random.randint(20, 190, (size, size), dtype=np.uint8)
    r = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    g = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    b = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=2))


def noise_blend(img: Image.Image, alpha: float) -> Image.Image:
    """Overlay static on img. alpha=1.0 → pure static, 0.0 → clean image."""
    if alpha <= 0.01:
        return img
    return Image.blend(img, make_static(img.width), min(alpha, 1.0))


# ── Embedding accumulator ───────────────────────────────────────────────────────
class EmbeddingAccumulator:
    """Thread-safe rolling average of 512-dim InsightFace embeddings."""

    def __init__(self, max_size: int = MAX_EMBEDDINGS):
        self._buf  = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def update(self, emb: np.ndarray):
        with self._lock:
            self._buf.append(emb.copy())

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            n = len(self._buf)
            if n == 0:
                return None, 0.0
            return np.mean(self._buf, axis=0), min(1.0, n / CONFIDENCE_FULL)

    def reset(self):
        with self._lock:
            self._buf.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buf)


# ── Load ML models BEFORE tkinter ──────────────────────────────────────────────
print("Loading InsightFace buffalo_l...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")

print("Loading SD 1.5 (img2img)...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)

print("Loading IP-Adapter FaceID...")
pipe.load_ip_adapter(
    "h94/IP-Adapter-FaceID",
    subfolder=None,
    weight_name="ip-adapter-faceid_sd15.bin",
    image_encoder_folder=None,
)

print("Loading LCM LoRA...")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
pipe.set_adapters(["lcm", "faceid_0"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")
# DO NOT call enable_attention_slicing — destroys IPAdapterAttnProcessor

print(f"Pipeline ready.  session_seed={SESSION_SEED}\n")


# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM)
for _ in range(8):
    cap.read()  # flush startup frames

accumulator = EmbeddingAccumulator()


def extract_embedding(bgr: np.ndarray) -> np.ndarray | None:
    faces = face_app.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


# ── tkinter display: 60fps with step interpolation ─────────────────────────────
root = tk.Tk()
root.title("Reflection")
root.configure(bg="#080808")
root.geometry(f"{W}x{H+40}")
root.resizable(False, False)

canvas_widget = tk.Canvas(root, width=W, height=H, bg="#080808", highlightthickness=0)
canvas_widget.pack()

status_label = tk.Label(root, text="initializing...", fg="#555", bg="#080808",
                        font=("Courier", 11), anchor="w")
status_label.pack(fill="x", padx=12)

# Interpolation state: smoothly blend between consecutive denoising steps
_ilock    = threading.Lock()
_step_a   = [None]   # image at step N-1 (interpolation source)
_step_b   = [None]   # image at step N   (interpolation target)
_step_b_t = [0.0]    # time _step_b was set
_step_dur = [1.5]    # EMA of seconds between steps

_status_buf  = [""]
_status_lock = threading.Lock()


def push_step(img: Image.Image):
    """Push a newly decoded denoising step. The display interpolates toward it."""
    now = time.time()
    with _ilock:
        if _step_b[0] is not None:
            elapsed = now - _step_b_t[0]
            if elapsed > 0.05:
                _step_dur[0] = 0.7 * _step_dur[0] + 0.3 * elapsed  # EMA smoothing
            _step_a[0] = _step_b[0]
        else:
            _step_a[0] = img  # first step ever: no src yet, set both
        _step_b[0] = img
        _step_b_t[0] = now


def push_status(s: str):
    with _status_lock:
        _status_buf[0] = s


def get_display_frame() -> Image.Image | None:
    """Compute the interpolated frame for the current moment in time."""
    with _ilock:
        a   = _step_a[0]
        b   = _step_b[0]
        t   = _step_b_t[0]
        dur = _step_dur[0]
    if b is None:
        return None
    if a is None or a is b:
        return b
    alpha = min(1.0, (time.time() - t) / max(dur, 0.05))
    return Image.blend(a, b, alpha)


_tk_img = [None]
_img_id = [None]


def tick():
    frame = get_display_frame()
    if frame is not None:
        disp = frame.resize((W, H), Image.BILINEAR)
        _tk_img[0] = ImageTk.PhotoImage(disp)
        if _img_id[0] is None:
            _img_id[0] = canvas_widget.create_image(0, 0, anchor="nw", image=_tk_img[0])
        else:
            canvas_widget.itemconfig(_img_id[0], image=_tk_img[0])
    with _status_lock:
        s = _status_buf[0]
    if s:
        status_label.config(text=f"  {s}")
    root.after(16, tick)  # ~60fps


root.after(16, tick)
root.bind("<Escape>", lambda _: root.quit())
root.bind("<q>",      lambda _: root.quit())

reset_flag = [False]


def on_reset(_):
    reset_flag[0] = True
    accumulator.reset()
    push_step(make_static())
    push_status("noise flood")


root.bind("<r>", on_reset)


# ── Embedding extraction thread ─────────────────────────────────────────────────
def embedding_loop():
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            emb = extract_embedding(frame)
            if emb is not None:
                accumulator.update(emb)
        time.sleep(EMBED_INTERVAL)


threading.Thread(target=embedding_loop, daemon=True).start()


# ── Generation loop ─────────────────────────────────────────────────────────────
def generation_loop():
    canvas       = [None]                              # last complete output (512×512 PIL)
    actual_steps = max(1, int(STEPS * STRENGTH))       # real denoising steps after noise schedule

    while True:
        # Noise flood reset
        if reset_flag[0]:
            reset_flag[0] = False
            canvas[0] = None
            push_step(make_static())
            print("[reset]  noise flood")
            time.sleep(0.3)
            continue

        # Camera frame — always read fresh
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame   = cv2.flip(frame, 1)
        cam_img = Image.fromarray(
            cv2.cvtColor(cv2.resize(frame, (512, 512)), cv2.COLOR_BGR2RGB)
        )

        # Identity
        embedding, confidence = accumulator.get()
        n           = accumulator.count
        noise_alpha = max(NOISE_MIN, 1.0 - confidence)
        ip_scale    = IP_MIN + confidence * (IP_MAX - IP_MIN)

        # Build init image:
        #   first run → noisy camera (seeds the process with your presence + static)
        #   subsequent → blend previous output with camera (continuity + anchoring)
        if canvas[0] is None:
            init_img = noise_blend(cam_img, 0.75)
        else:
            init_img = Image.blend(canvas[0], cam_img, CAM_WEIGHT)

        # No face detected yet — hold in animated static
        if embedding is None:
            push_step(make_static(512))
            push_status("no face — look at camera")
            time.sleep(0.5)
            continue

        face_t = torch.from_numpy(embedding).float().unsqueeze(0).unsqueeze(0).to("mps")
        pipe.set_ip_adapter_scale(ip_scale)
        generator = torch.Generator(device="mps").manual_seed(SESSION_SEED)

        push_status(
            f"● n={n}  conf={confidence:.0%}  "
            f"ip={ip_scale:.2f}  static={noise_alpha:.2f}  |  R=flood"
        )

        # Step callback: decode each latent and push to the display interpolator.
        # Noise is heavier at early steps, clears toward the final step.
        def on_step(pipeline, step_idx, _, callback_kwargs,
                    _steps=actual_steps, _na=noise_alpha):
            latents = callback_kwargs.get("latents")
            if latents is not None:
                with torch.no_grad():
                    scaled  = latents / pipeline.vae.config.scaling_factor
                    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
                decoded   = (decoded / 2 + 0.5).clamp(0, 1)
                arr       = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                step_img  = Image.fromarray(arr)
                step_frac = (step_idx + 1) / max(_steps, 1)
                # Noise envelope: starts above _na, falls to _na at final step
                step_noise = _na + (1.0 - _na) * (1.0 - step_frac) * 0.45
                push_step(noise_blend(step_img, min(0.92, step_noise)))
            return callback_kwargs

        try:
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                image=init_img,
                strength=STRENGTH,
                ip_adapter_image_embeds=[face_t],
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            canvas[0] = result.images[0]
            # Final push: clean output at base noise level — immediately becomes
            # the next generation's init, so it flows without pause
            push_step(noise_blend(canvas[0], noise_alpha))

        except Exception as e:
            print(f"[gen]  error: {e}")
            import traceback
            traceback.print_exc()
            push_status(f"error: {e}")
            time.sleep(1.0)


threading.Thread(target=generation_loop, daemon=True).start()

try:
    root.mainloop()
finally:
    cap.release()
    print("Done.")
