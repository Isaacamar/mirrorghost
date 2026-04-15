"""
Reflection v2 — IP-Adapter FaceID + noise persistence

Visual concept:
  TV static fills the screen at first.
  As InsightFace accumulates your identity, the static clears.
  Your face emerges from the noise — uncanny, recognizable, wrong.

Two convergence axes:
  Session axis:    static fades as embedding count rises (minutes)
  Generation axis: each run shows noise → face in real-time (seconds)

A fixed session seed anchors the base identity for the session.
IP-Adapter scale rises with confidence, pulling that face toward the subject.
The static never fully disappears — it lingers at NOISE_MIN.

Controls: Q / Escape = quit    R = noise flood (reset embedding + canvas)
"""

import os, time, threading, collections, random
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionPipeline, LCMScheduler

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL         = 12      # seconds between generations
EMBED_INTERVAL   = 1.0     # seconds between embedding extractions
MAX_EMBEDDINGS   = 60      # rolling window size
CONFIDENCE_FULL  = 20      # embeddings for 100% confidence
DISPLAY_RATE     = 0.33    # seconds between display refreshes between generations
NOISE_MIN        = 0.12    # minimum noise at full confidence — always slightly wrong

SESSION_SEED     = random.randint(0, 2**32)  # fixes the base face for this session

CAM  = 0
W, H = 800, 800

STEPS_MIN   = 4            # denoising steps at zero confidence
STEPS_MAX   = 8            # denoising steps at full confidence
IP_MIN      = 0.25         # IP-Adapter scale at zero confidence
IP_MAX      = 0.90         # IP-Adapter scale at full confidence
GUIDANCE    = 1.0          # 1.0 = no CFG (required for ip_adapter_image_embeds shape)

PROMPT   = ("photorealistic human face portrait, studio lighting, "
            "sharp focus, detailed skin texture, neutral expression, "
            "front facing, close up, high resolution, cinematic")
NEGATIVE = ("cartoon, anime, painting, deformed, blurry, text, watermark, "
            "body, clothes, background, hat, sunglasses, multiple faces, "
            "bad anatomy, artifacts")


# ── Noise / static ────────────────────────────────────────────────────────────
def make_static(size: int = 512) -> Image.Image:
    """TV static: grayscale base with chromatic fringing."""
    luma = np.random.randint(20, 190, (size, size), dtype=np.uint8)
    r = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    g = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    b = np.clip(luma.astype(int) + np.random.randint(-20, 20, (size, size)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=2))


def noise_blend(img: Image.Image, alpha: float) -> Image.Image:
    """Overlay static on img. alpha=1.0 → pure static, alpha=0.0 → clean image."""
    if alpha <= 0.01:
        return img
    return Image.blend(img, make_static(img.width), min(alpha, 1.0))


# ── Embedding accumulator ─────────────────────────────────────────────────────
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


# ── Load all ML models BEFORE opening tkinter ─────────────────────────────────
print("Loading InsightFace buffalo_l...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")

print("Loading SD 1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
)

# IP-Adapter FaceID also loads the FaceID LoRA (K/Q attention) via PEFT
print("Loading IP-Adapter FaceID...")
pipe.load_ip_adapter(
    "h94/IP-Adapter-FaceID",
    subfolder=None,
    weight_name="ip-adapter-faceid_sd15.bin",
    image_encoder_folder=None,
)

# Load LCM LoRA for fast inference, fuse with FaceID LoRA
print("Loading LCM LoRA...")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
pipe.set_adapters(["lcm", "faceid_0"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")
# IMPORTANT: do not call enable_attention_slicing after load_ip_adapter —
# it replaces IPAdapterAttnProcessor with SlicedAttnProcessor and breaks the pipeline.

print(f"Pipeline ready.  session_seed={SESSION_SEED}\n")


# ── Camera + InsightFace ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAM)
for _ in range(8):
    cap.read()  # flush startup frames

accumulator = EmbeddingAccumulator()


def extract_embedding(bgr: np.ndarray) -> np.ndarray | None:
    """Run InsightFace; return normed 512-dim embedding of largest face, or None."""
    faces = face_app.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


# ── tkinter display ───────────────────────────────────────────────────────────
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

_lock        = threading.Lock()
_pending_img = [None]
_pending_st  = [""]


def push_img(img: Image.Image):
    with _lock:
        _pending_img[0] = img.copy()


def push_status(s: str):
    with _lock:
        _pending_st[0] = s


_tk_img = [None]
_img_id = [None]


def tick():
    with _lock:
        img = _pending_img[0]; _pending_img[0] = None
        s   = _pending_st[0];  _pending_st[0]  = ""
    if img is not None:
        disp = img.resize((W, H), Image.BILINEAR)
        _tk_img[0] = ImageTk.PhotoImage(disp)
        if _img_id[0] is None:
            _img_id[0] = canvas_widget.create_image(0, 0, anchor="nw", image=_tk_img[0])
        else:
            canvas_widget.itemconfig(_img_id[0], image=_tk_img[0])
    if s:
        status_label.config(text=f"  {s}")
    root.after(33, tick)


root.after(33, tick)
root.bind("<Escape>", lambda _: root.quit())
root.bind("<q>",      lambda _: root.quit())

reset_flag = [False]


def on_reset(_):
    reset_flag[0] = True
    accumulator.reset()
    push_status("noise flood")


root.bind("<r>", on_reset)


# ── Embedding extraction thread ───────────────────────────────────────────────
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


# ── Generation loop ───────────────────────────────────────────────────────────
def generation_loop():
    clean_canvas = [None]   # latest clean generated image (512×512 PIL)
    last_gen = -INTERVAL    # trigger first generation immediately

    while True:
        # Noise flood reset: clear canvas, restart convergence
        if reset_flag[0]:
            reset_flag[0] = False
            clean_canvas[0] = None
            last_gen = -INTERVAL
            print("[reset]  noise flood")

        n = accumulator.count
        embedding, confidence = accumulator.get()

        # noise_alpha: 1.0 = pure static, NOISE_MIN = minimal static at full confidence
        noise_alpha = max(NOISE_MIN, 1.0 - confidence)

        now = time.time()
        remaining = int(INTERVAL - (now - last_gen))

        if remaining > 0:
            # Between generations: show canvas blended with static (flickering at ~3fps)
            base = clean_canvas[0] if clean_canvas[0] is not None else make_static()
            push_img(noise_blend(base, noise_alpha))
            icon = "●" if n > 0 else "○"
            push_status(
                f"{icon}  n={n}  conf={confidence:.0%}  "
                f"static={noise_alpha:.2f}  next in {remaining}s  |  R=flood"
            )
            time.sleep(DISPLAY_RATE)
            continue

        # No face yet — hold in static
        if embedding is None:
            push_img(make_static())
            push_status("no face detected — look at camera")
            time.sleep(1.0)
            continue

        # Schedule from confidence
        steps    = int(STEPS_MIN + confidence * (STEPS_MAX - STEPS_MIN))
        ip_scale = IP_MIN + confidence * (IP_MAX - IP_MIN)

        print(f"\n[gen]  n={n}  conf={confidence:.0%}  steps={steps}  "
              f"ip={ip_scale:.2f}  seed={SESSION_SEED}")
        push_status(f"generating...  conf={confidence:.0%}  steps={steps}  ip={ip_scale:.2f}")

        last_gen = time.time()

        # Raw 512-dim embedding — pipeline projects to [1, 4, 768] internally
        face_t = torch.from_numpy(embedding).float().unsqueeze(0).unsqueeze(0).to("mps")  # [1, 1, 512]
        pipe.set_ip_adapter_scale(ip_scale)
        generator = torch.Generator(device="mps").manual_seed(SESSION_SEED)

        def on_step(pipeline, step_idx, _, callback_kwargs,
                    _steps=steps, _na=noise_alpha):
            """Show each denoising step blended with noise — face emerges through static."""
            latents = callback_kwargs.get("latents")
            if latents is not None:
                with torch.no_grad():
                    scaled  = latents / pipeline.vae.config.scaling_factor
                    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                arr = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                step_img = Image.fromarray(arr)
                # Extra noise at early steps, clears by final step
                step_frac = (step_idx + 1) / _steps
                step_noise = _na + (1.0 - _na) * (1.0 - step_frac) * 0.5
                push_img(noise_blend(step_img, min(0.95, step_noise)))
                push_status(f"step {step_idx + 1}/{_steps}  |  conf {confidence:.0%}")
            return callback_kwargs

        try:
            t0 = time.time()
            result = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                ip_adapter_image_embeds=[face_t],
                num_inference_steps=steps,
                guidance_scale=GUIDANCE,
                generator=generator,
                callback_on_step_end=on_step,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            elapsed = time.time() - t0
            clean_canvas[0] = result.images[0]
            # Final display: clean output at current noise level
            push_img(noise_blend(clean_canvas[0], noise_alpha))
            push_status(
                f"done {elapsed:.1f}s  |  conf={confidence:.0%}  "
                f"static={noise_alpha:.2f}  n={n}  |  R=flood"
            )
            print(f"[gen]  done in {elapsed:.1f}s")

        except Exception as e:
            print(f"[gen]  error: {e}")
            import traceback
            traceback.print_exc()
            push_status(f"error: {e}")


threading.Thread(target=generation_loop, daemon=True).start()

try:
    root.mainloop()
finally:
    cap.release()
    print("Done.")
