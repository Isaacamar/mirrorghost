"""
protocol.py — JSON message schemas for the WebSocket protocol.

iPad → Mac:
  {"type": "face_frame", "timestamp": float, "blend_shapes": {...}, "head_euler": {...}}
  {"type": "advance_morph"}
  {"type": "reset"}

Mac → iPad:
  {"type": "server_ready", "sd_model": str, "controlnet_model": str}
  {"type": "face_frame", "jpeg_b64": str, "morph_weight": float, "frame_index": int, "generation_ms": int}
  {"type": "morph_update", "morph_weight": float}
"""

import json
import time


def server_ready(sd_model: str, controlnet_model: str) -> str:
    return json.dumps({
        "type":             "server_ready",
        "sd_model":         sd_model,
        "controlnet_model": controlnet_model,
    })


def face_frame_out(jpeg_b64: str, morph_weight: float,
                   frame_index: int, generation_ms: int) -> str:
    return json.dumps({
        "type":          "face_frame",
        "jpeg_b64":      jpeg_b64,
        "morph_weight":  round(morph_weight, 4),
        "frame_index":   frame_index,
        "generation_ms": generation_ms,
    })


def morph_update(morph_weight: float) -> str:
    return json.dumps({
        "type":         "morph_update",
        "morph_weight": round(morph_weight, 4),
    })


def parse_incoming(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
