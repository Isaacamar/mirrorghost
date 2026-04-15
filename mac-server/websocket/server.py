"""
server.py — FastAPI WebSocket handler

Architecture:
  - One WebSocket client at a time (iPad)
  - WS receive handler runs in the async event loop (updates shared state)
  - Generation loop runs in a thread executor (blocks during SD inference)
  - Results are sent back via a queue bridged to the async sender task
"""

import asyncio
import queue
import threading
import time

from fastapi import WebSocket, WebSocketDisconnect

from websocket.protocol import (
    server_ready, face_frame_out, morph_update, parse_incoming
)
from pipeline.conditioning import build_conditioning_image
from pipeline.face_utils import encode_jpeg, blank_canvas
from config import (
    SD_MODEL_ID, CONTROLNET_MODEL_ID,
    MORPH_ADVANCE_INTERVAL_SECONDS, MORPH_INCREMENT, JPEG_QUALITY
)


class SharedState:
    """Thread-safe face tracking state updated by the WebSocket receiver."""
    def __init__(self):
        self._lock        = threading.Lock()
        self.blend_shapes: dict = {}
        self.head_euler:   dict = {}
        self.has_face: bool     = False

    def update(self, blend_shapes: dict, head_euler: dict):
        with self._lock:
            self.blend_shapes = blend_shapes
            self.head_euler   = head_euler
            self.has_face     = True

    def get(self) -> tuple[dict, dict, bool]:
        with self._lock:
            return dict(self.blend_shapes), dict(self.head_euler), self.has_face


shared_state  = SharedState()
stop_gen      = threading.Event()


async def websocket_handler(websocket: WebSocket, pipeline, morph_state):
    await websocket.accept()
    print("[ws]  client connected")

    pipeline.new_session()
    morph_state.reset()
    stop_gen.clear()

    # Queue bridges the generation thread → async sender
    send_q: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    # ── Sender task: drains send_q and writes to WebSocket ────────────────
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

        # Send first frame immediately (blank/random face before any tracking data)
        while not stop_gen.is_set():
            blend, euler, has_face = shared_state.get()

            weight = morph_state.get_weight()

            if has_face:
                conditioning = build_conditioning_image(blend, euler)
            else:
                conditioning = blank_canvas()

            try:
                image, elapsed = pipeline.generate(conditioning, weight)
                jpeg_b64 = encode_jpeg(image, quality=JPEG_QUALITY)
                msg = face_frame_out(
                    jpeg_b64      = jpeg_b64,
                    morph_weight  = weight,
                    frame_index   = frame_index,
                    generation_ms = int(elapsed * 1000),
                )
                send_q.put(msg)
                frame_index += 1
                print(f"[gen]  frame={frame_index}  morph={weight:.3f}  "
                      f"face={has_face}  {elapsed:.1f}s")
            except Exception as e:
                import traceback
                print(f"[gen]  error: {e}")
                traceback.print_exc()

            # Auto-advance morph on timer
            if time.time() - last_advance > MORPH_ADVANCE_INTERVAL_SECONDS:
                morph_state.advance(MORPH_INCREMENT)
                send_q.put(morph_update(morph_state.get_weight()))
                last_advance = time.time()

    gen_thread = threading.Thread(target=generation_loop, daemon=True)
    gen_thread.start()

    # ── Send server_ready ────────────────────────────────────────────────
    await websocket.send_text(server_ready(SD_MODEL_ID, CONTROLNET_MODEL_ID))

    # ── Receive loop ─────────────────────────────────────────────────────
    try:
        while True:
            raw = await websocket.receive_text()
            msg = parse_incoming(raw)
            msg_type = msg.get("type")

            if msg_type == "face_frame":
                shared_state.update(
                    msg.get("blend_shapes", {}),
                    msg.get("head_euler",   {}),
                )

            elif msg_type == "advance_morph":
                morph_state.advance(MORPH_INCREMENT)
                send_q.put(morph_update(morph_state.get_weight()))

            elif msg_type == "reset":
                morph_state.reset()
                send_q.put(morph_update(0.0))

    except WebSocketDisconnect:
        print("[ws]  client disconnected")
    finally:
        stop_gen.set()
        sender_task.cancel()
        gen_thread.join(timeout=2)
