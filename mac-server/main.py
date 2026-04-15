"""
main.py — Mirror Mac server entry point

Run:
  source ../.venv/bin/activate
  cd mac-server
  python main.py
"""

import os
import socket
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from fastapi import FastAPI, WebSocket
from insightface.app import FaceAnalysis

from pipeline.generation import GenerationPipeline
from pipeline.morph      import MorphState
from websocket.server    import websocket_handler
from config              import WS_HOST, WS_PORT

print("=" * 60)
print("Mirror Mac Server")
print("=" * 60)

# InsightFace must load before pipeline (both use CoreML)
print("Loading InsightFace buffalo_l...")
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")

pipeline = GenerationPipeline()
pipeline.setup()

morph_state = MorphState()
print("Models ready.\n")

def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

ip = local_ip()
print(f"  Local IP:  {ip}")
print(f"  WebSocket: ws://{ip}:{WS_PORT}/ws")
print(f"  Enter this IP in the Mirror iPad app.\n")

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket_handler(websocket, pipeline, morph_state, face_app)

@app.get("/health")
def health():
    return {"status": "ok", "seed": pipeline.session_seed}

if __name__ == "__main__":
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, log_level="info")
