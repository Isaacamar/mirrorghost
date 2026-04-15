"""
embeddings.py — face identity accumulation + noise utilities (shared with generation)
"""

import threading
import collections
import numpy as np
from PIL import Image


CONFIDENCE_FULL = 20    # embeddings needed for 100% confidence
NOISE_MIN       = 0.08  # minimum static at full confidence
MAX_EMBEDDINGS  = 60    # rolling window size


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
