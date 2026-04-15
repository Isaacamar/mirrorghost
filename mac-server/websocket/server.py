"""
server.py — WebSocket handler with InsightFace + EmbeddingAccumulator

Message flow:
  iPad → Mac: face_frame (blend shapes + euler), face_image (JPEG for InsightFace)
  Mac → iPad: server_ready, face_frame (generated JPEG + morph_weight)
"""

import asyncio
import base64
import queue
import threading
import time

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from websocket.protocol import (
    server_ready, face_frame_out, morph_update, parse_incoming
)
from pipeline.conditioning import build_conditioning_image
from pipeline.face_utils    import encode_jpeg, blank_canvas
from pipeline.embeddings    import EmbeddingAccumulator
from config import (
    SD_MODEL_ID, CONTROLNET_MODEL_ID,
    MORPH_ADVANCE_INTERVAL_SECONDS, MORPH_INCREMENT, JPEG_QUALITY
)


class SharedState:
    """Thread-safe state updated by the WebSocket receiver."""
    def __init__(self):
        self._lock        = threading.Lock()
        self.blend_shapes: dict = {}
        self.head_euler:   dict = {}
        self.has_face: bool     = False

    def update_face_frame(self, blend_shapes: dict, head_euler: dict):
        with self._lock:
            self.blend_shapes = blend_shapes
            self.head_euler   = head_euler
            self.has_face     = True

    def get(self) -> tuple[dict, dict, bool]:
        with self._lock:
            return dict(self.blend_shapes), dict(self.head_euler), self.has_face


def extract_embedding(face_app, bgr: np.ndarray) -> np.ndarray | None:
    """Run InsightFace on a BGR frame; return normed 512-dim embedding or None."""
    faces = face_app.get(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


stop_gen = threading.Event()


async def websocket_handler(websocket: WebSocket, pipeline, morph_state, face_app):
    await websocket.accept()
    print("[ws]  client connected")

    pipeline.new_session()
    morph_state.reset()
    stop_gen.clear()

    shared      = SharedState()
    accumulator = EmbeddingAccumulator()
    send_q: queue.Queue = queue.Queue()

    # ── Async sender: drains send_q → WebSocket ───────────────────────────
    async def sender():
        while True:
            try:
                msg = send_q.get_nowait()
                await websocket.send_text(msg)
            except queue.Empty:
                await asyncio.sleep(0.02)

    sender_task = asyncio.create_task(sender())

    # ── Generation thread ─────────────────────────────────────────────────
    def generation_loop():
        frame_index  = 0
        last_advance = time.time()

        while not stop_gen.is_set():
            blend, euler, has_face = shared.get()
            embedding, confidence  = accumulator.get()
            weight = morph_state.get_weight()
            n      = accumulator.count

            conditioning = build_conditioning_image(blend, euler) if has_face else blank_canvas()

            try:
                image, elapsed = pipeline.generate(
                    conditioning_image=conditioning,
                    morph_weight=weight,
                    embedding=embedding,
                    confidence=confidence,
                )
                jpeg_b64 = encode_jpeg(image, quality=JPEG_QUALITY)
                send_q.put(face_frame_out(
                    jpeg_b64      = jpeg_b64,
                    morph_weight  = weight,
                    frame_index   = frame_index,
                    generation_ms = int(elapsed * 1000),
                ))
                frame_index += 1
                print(f"[gen]  frame={frame_index}  morph={weight:.3f}  "
                      f"conf={confidence:.0%}  n={n}  ip={pipeline.pipe.unet.config.get('ip_adapter_scale', '?')}  "
                      f"{elapsed:.1f}s")
            except Exception as e:
                import traceback
                print(f"[gen]  error: {e}")
                traceback.print_exc()

            # Auto-advance morph
            if time.time() - last_advance > MORPH_ADVANCE_INTERVAL_SECONDS:
                morph_state.advance(MORPH_INCREMENT)
                send_q.put(morph_update(morph_state.get_weight()))
                last_advance = time.time()

    gen_thread = threading.Thread(target=generation_loop, daemon=True)
    gen_thread.start()

    await websocket.send_text(server_ready(SD_MODEL_ID, CONTROLNET_MODEL_ID))

    # ── Receive loop ──────────────────────────────────────────────────────
    try:
        while True:
            raw = await websocket.receive_text()
            msg = parse_incoming(raw)
            msg_type = msg.get("type")

            if msg_type == "face_frame":
                shared.update_face_frame(
                    msg.get("blend_shapes", {}),
                    msg.get("head_euler",   {}),
                )

            elif msg_type == "face_image":
                # iPad sends a JPEG of the user's face for InsightFace identity extraction.
                # Use PIL (not cv2) to decode — PIL applies EXIF orientation automatically,
                # which matters because the ARKit front camera captures in landscape.
                b64 = msg.get("jpeg_b64", "")
                if b64:
                    try:
                        from PIL import Image as PILImage, ImageOps
                        import io
                        img_bytes = base64.b64decode(b64)
                        pil_img   = PILImage.open(io.BytesIO(img_bytes))
                        pil_img   = ImageOps.exif_transpose(pil_img)   # apply rotation
                        frame     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        emb = extract_embedding(face_app, frame)
                        if emb is not None:
                            accumulator.update(emb)
                            _, conf = accumulator.get()
                            print(f"[face]  embedding updated  "
                                  f"n={accumulator.count}  conf={conf:.0%}")
                        else:
                            print("[face]  no face detected in frame")
                    except Exception as e:
                        print(f"[face]  image error: {e}")

            elif msg_type == "advance_morph":
                morph_state.advance(MORPH_INCREMENT)
                send_q.put(morph_update(morph_state.get_weight()))

            elif msg_type == "reset":
                morph_state.reset()
                accumulator.reset()
                send_q.put(morph_update(0.0))

    except WebSocketDisconnect:
        print("[ws]  client disconnected")
    finally:
        stop_gen.set()
        sender_task.cancel()
        gen_thread.join(timeout=2)
