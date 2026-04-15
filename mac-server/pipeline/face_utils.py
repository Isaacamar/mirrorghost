import base64
import io
from PIL import Image


def encode_jpeg(image: Image.Image, quality: int = 85) -> str:
    """Encode PIL image to base64 JPEG string for WebSocket transmission."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def blank_canvas(size: int = 512) -> Image.Image:
    """All-black image. ControlNet conditioning_scale=0 makes it irrelevant."""
    return Image.new("RGB", (size, size), (0, 0, 0))
