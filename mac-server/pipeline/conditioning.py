"""
conditioning.py — ARKit blend shapes → OpenPose-style face skeleton image

The ControlNet was trained on OpenPose skeletons: a black canvas with
colored dots at landmark positions. We approximate this from ARKit's
52 blend shape coefficients + head euler angles.

Landmark positions are in normalized [0,1] space (multiply by 512 for pixels).
"""

import math
from PIL import Image, ImageDraw


# Neutral/rest positions on a 512×512 canvas (normalized)
BASE_LANDMARKS = {
    # Eyes
    "left_eye_outer":   (0.32, 0.38),
    "left_eye_inner":   (0.42, 0.38),
    "left_eye_top":     (0.37, 0.35),
    "left_eye_bottom":  (0.37, 0.41),
    "right_eye_outer":  (0.68, 0.38),
    "right_eye_inner":  (0.58, 0.38),
    "right_eye_top":    (0.63, 0.35),
    "right_eye_bottom": (0.63, 0.41),
    # Eyebrows
    "left_brow_inner":  (0.40, 0.31),
    "left_brow_outer":  (0.28, 0.32),
    "right_brow_inner": (0.60, 0.31),
    "right_brow_outer": (0.72, 0.32),
    # Nose
    "nose_tip":         (0.50, 0.52),
    "nose_left":        (0.44, 0.55),
    "nose_right":       (0.56, 0.55),
    # Mouth
    "mouth_left":       (0.38, 0.65),
    "mouth_right":      (0.62, 0.65),
    "mouth_top":        (0.50, 0.61),
    "mouth_bottom":     (0.50, 0.70),
    # Jaw / chin
    "chin":             (0.50, 0.82),
    "jaw_left":         (0.28, 0.72),
    "jaw_right":        (0.72, 0.72),
}

# OpenPose face colors (roughly matched to training data)
LANDMARK_COLORS = {
    "left_eye_outer":   (255, 0, 0),
    "left_eye_inner":   (255, 85, 0),
    "left_eye_top":     (255, 170, 0),
    "left_eye_bottom":  (255, 255, 0),
    "right_eye_outer":  (170, 255, 0),
    "right_eye_inner":  (85, 255, 0),
    "right_eye_top":    (0, 255, 0),
    "right_eye_bottom": (0, 255, 85),
    "left_brow_inner":  (0, 255, 170),
    "left_brow_outer":  (0, 255, 255),
    "right_brow_inner": (0, 170, 255),
    "right_brow_outer": (0, 85, 255),
    "nose_tip":         (85, 0, 255),
    "nose_left":        (170, 0, 255),
    "nose_right":       (255, 0, 255),
    "mouth_left":       (255, 0, 170),
    "mouth_right":      (255, 0, 85),
    "mouth_top":        (255, 85, 85),
    "mouth_bottom":     (255, 170, 85),
    "chin":             (200, 200, 200),
    "jaw_left":         (180, 180, 180),
    "jaw_right":        (160, 160, 160),
}

CONNECTIONS = [
    ("left_eye_outer",  "left_eye_top"),
    ("left_eye_top",    "left_eye_inner"),
    ("left_eye_inner",  "left_eye_bottom"),
    ("left_eye_bottom", "left_eye_outer"),
    ("right_eye_outer", "right_eye_top"),
    ("right_eye_top",   "right_eye_inner"),
    ("right_eye_inner", "right_eye_bottom"),
    ("right_eye_bottom","right_eye_outer"),
    ("left_brow_outer", "left_brow_inner"),
    ("right_brow_inner","right_brow_outer"),
    ("mouth_left",      "mouth_top"),
    ("mouth_top",       "mouth_right"),
    ("mouth_right",     "mouth_bottom"),
    ("mouth_bottom",    "mouth_left"),
    ("nose_left",       "nose_tip"),
    ("nose_tip",        "nose_right"),
    ("jaw_left",        "chin"),
    ("chin",            "jaw_right"),
]


def blend_shapes_to_offsets(blend_shapes: dict) -> dict:
    """Map ARKit blend shape coefficients to pixel offsets from base positions."""
    offsets: dict = {}

    jaw = blend_shapes.get("jawOpen", 0.0)
    offsets["mouth_bottom"] = (0, jaw * 30)

    smile_l = blend_shapes.get("mouthSmileLeft", 0.0)
    smile_r = blend_shapes.get("mouthSmileRight", 0.0)
    offsets["mouth_left"]  = (-smile_l * 15, -smile_l * 10)
    offsets["mouth_right"] = (smile_r * 15,  -smile_r * 10)

    brow_up = blend_shapes.get("browInnerUp", 0.0)
    offsets["left_brow_inner"]  = (0, -brow_up * 12)
    offsets["right_brow_inner"] = (0, -brow_up * 12)

    brow_dl = blend_shapes.get("browDownLeft",  0.0)
    brow_dr = blend_shapes.get("browDownRight", 0.0)
    offsets["left_brow_outer"]  = (brow_dl * 5, brow_dl * 10)
    offsets["right_brow_outer"] = (-brow_dr * 5, brow_dr * 10)

    blink_l = blend_shapes.get("eyeBlinkLeft",  0.0)
    blink_r = blend_shapes.get("eyeBlinkRight", 0.0)
    offsets["left_eye_top"]     = (0,  blink_l * 8)
    offsets["left_eye_bottom"]  = (0, -blink_l * 8)
    offsets["right_eye_top"]    = (0,  blink_r * 8)
    offsets["right_eye_bottom"] = (0, -blink_r * 8)

    # Mouth funnel / pucker
    funnel = blend_shapes.get("mouthFunnel", 0.0)
    pucker = blend_shapes.get("mouthPucker", 0.0)
    compress = (funnel + pucker) * 10
    offsets["mouth_left"]  = (offsets.get("mouth_left",  (0, 0))[0] + compress,
                              offsets.get("mouth_left",  (0, 0))[1])
    offsets["mouth_right"] = (offsets.get("mouth_right", (0, 0))[0] - compress,
                              offsets.get("mouth_right", (0, 0))[1])

    return offsets


def apply_head_rotation(landmarks: dict, pitch: float, yaw: float, roll: float) -> dict:
    """Rotate all landmarks around center (0.5, 0.5) by head euler angles (degrees)."""
    cx, cy = 0.5, 0.5
    roll_rad = math.radians(roll)
    rotated = {}
    for name, (x, y) in landmarks.items():
        # Roll rotation
        dx, dy = x - cx, y - cy
        rx = dx * math.cos(roll_rad) - dy * math.sin(roll_rad) + cx
        ry = dx * math.sin(roll_rad) + dy * math.cos(roll_rad) + cy
        # Yaw: perspective x-compression toward center
        yaw_factor = 1.0 - abs(yaw) / 90.0 * 0.4
        rx = cx + (rx - cx) * yaw_factor + (yaw / 90.0) * 0.08
        # Pitch: vertical shift
        ry += (pitch / 45.0) * 0.05
        rotated[name] = (rx, ry)
    return rotated


def render_openpose_canvas(landmarks: dict, size: int = 512) -> Image.Image:
    """Draw OpenPose-style face skeleton on black canvas."""
    img  = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    def px(coord):
        return (int(coord[0] * size), int(coord[1] * size))

    for a, b in CONNECTIONS:
        if a in landmarks and b in landmarks:
            draw.line([px(landmarks[a]), px(landmarks[b])],
                      fill=(100, 100, 100), width=2)

    for name, coord in landmarks.items():
        x, y = px(coord)
        color = LANDMARK_COLORS.get(name, (255, 255, 255))
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

    return img


def build_conditioning_image(blend_shapes: dict, head_euler: dict,
                             size: int = 512) -> Image.Image:
    """Main entry point called by generation.py."""
    offsets   = blend_shapes_to_offsets(blend_shapes)
    landmarks = {}
    for name, (bx, by) in BASE_LANDMARKS.items():
        ox, oy = offsets.get(name, (0, 0))
        landmarks[name] = (bx + ox / size, by + oy / size)
    landmarks = apply_head_rotation(
        landmarks,
        head_euler.get("pitch", 0.0),
        head_euler.get("yaw",   0.0),
        head_euler.get("roll",  0.0),
    )
    return render_openpose_canvas(landmarks, size)
