"""
main.py — Mirror Mac server entry point

Startup:
  source ../.venv/bin/activate
  cd mac-server
  pip install fastapi uvicorn[standard] python-dotenv
  python main.py

The server loads SD models first (30-90s), then starts listening.
It logs the Mac's local IP so you know what to type into the iPad app.
"""

import os
import socket
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add mac-server to path so sub-modules resolve
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from fastapi import FastAPI, WebSocket

from pipeline.generation import GenerationPipeline
from pipeline.morph      import MorphState
from websocket.server    import websocket_handler
from config              import WS_HOST, WS_PORT

# ── Load models before starting the server ────────────────────────────────────
print("=" * 60)
print("Mirror Mac Server")
print("=" * 60)
pipeline    = GenerationPipeline()
pipeline.setup()          # blocks until models are loaded
morph_state = MorphState()
print("Models ready.\n")

# Print Mac's local IP so you know what to enter in the iPad app
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

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket_handler(websocket, pipeline, morph_state)


@app.get("/health")
def health():
    return {"status": "ok", "seed": pipeline.session_seed}


if __name__ == "__main__":
    uvicorn.run(app, host=WS_HOST, port=WS_PORT, log_level="info")
