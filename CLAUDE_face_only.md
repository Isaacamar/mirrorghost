# Mirror — Face Pipeline Build Spec
> Claude Code prompt file. Read this entire document before writing any code.

---

## Project Summary

An iPad displays a generated face. Initially random. Over time it morphs toward the face of whoever is sitting in front of it — driven by ARKit TrueDepth face tracking on the iPad, Stable Diffusion + ControlNet on the Mac, communicating over local WiFi.

That's the entire scope of this file. No audio. No voice. No questions. Just:

```
iPad TrueDepth → blend shapes → WebSocket → Mac → SD + ControlNet → JPEG → WebSocket → iPad display
```

---

## Hardware

| Device | Role |
|---|---|
| iPad Pro M4 | Front camera (TrueDepth), full-screen face display |
| MacBook Pro M4 | Stable Diffusion inference, ControlNet conditioning, WebSocket server |
| Local WiFi | LAN bridge — no internet needed for core loop |

---

## Repository Structure

```
mirror/
├── CLAUDE.md                     ← this file
├── README.md
├── .env.example
├── .gitignore
│
├── mac-server/
│   ├── main.py                   ← FastAPI + WebSocket server entry point
│   ├── requirements.txt
│   ├── config.py                 ← all tuning constants in one place
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── generation.py         ← SD + ControlNet inference
│   │   ├── conditioning.py       ← ARKit blend shapes → OpenPose canvas image
│   │   ├── morph.py              ← morph_weight state machine
│   │   └── face_utils.py         ← PIL helpers, JPEG encode to base64
│   │
│   ├── websocket/
│   │   ├── __init__.py
│   │   ├── server.py             ← WebSocket connection handler
│   │   └── protocol.py           ← Pydantic message schemas
│   │
│   └── tests/
│       ├── test_generation.py    ← run SD in isolation, save output image
│       ├── test_conditioning.py  ← feed fake blend shapes, inspect canvas
│       └── test_morph.py         ← verify weight increments correctly
│
└── ipad-app/
    ├── Mirror.xcodeproj
    └── Mirror/
        ├── App/
        │   ├── MirrorApp.swift
        │   └── AppState.swift        ← ObservableObject, single source of truth
        │
        ├── Views/
        │   ├── SessionView.swift     ← root view, wires everything together
        │   ├── FaceDisplayView.swift ← full-screen JPEG frame renderer
        │   └── DebugOverlayView.swift← morph_weight, fps, connection status (dev only)
        │
        ├── ARKit/
        │   ├── FaceTracker.swift     ← ARKit session, blend shape extraction
        │   └── BlendShapeMapper.swift← ARFaceAnchor keys → serializable dict
        │
        └── Network/
            ├── WSClient.swift        ← URLSessionWebSocketTask wrapper
            └── MessageHandler.swift  ← routes incoming server messages to AppState
```

---

## WebSocket Protocol

All messages are JSON strings. One connection, persistent for the session.

### iPad → Mac

```json
// Sent at 10fps (throttled from 60fps ARKit)
{
  "type": "face_frame",
  "timestamp": 1712345678.123,
  "blend_shapes": {
    "eyeBlinkLeft": 0.02,
    "eyeBlinkRight": 0.03,
    "eyeLookDownLeft": 0.11,
    "eyeLookDownRight": 0.09,
    "eyeLookInLeft": 0.0,
    "eyeLookInRight": 0.0,
    "eyeLookOutLeft": 0.0,
    "eyeLookOutRight": 0.0,
    "eyeLookUpLeft": 0.0,
    "eyeLookUpRight": 0.0,
    "eyeSquintLeft": 0.05,
    "eyeSquintRight": 0.04,
    "eyeWideLeft": 0.01,
    "eyeWideRight": 0.01,
    "jawForward": 0.0,
    "jawLeft": 0.0,
    "jawOpen": 0.15,
    "jawRight": 0.0,
    "mouthClose": 0.0,
    "mouthDimpleLeft": 0.02,
    "mouthDimpleRight": 0.03,
    "mouthFrownLeft": 0.05,
    "mouthFrownRight": 0.04,
    "mouthFunnel": 0.0,
    "mouthLeft": 0.0,
    "mouthLowerDownLeft": 0.08,
    "mouthLowerDownRight": 0.09,
    "mouthPressLeft": 0.0,
    "mouthPressRight": 0.0,
    "mouthPucker": 0.0,
    "mouthRight": 0.0,
    "mouthRollLower": 0.0,
    "mouthRollUpper": 0.0,
    "mouthShrugLower": 0.0,
    "mouthShrugUpper": 0.0,
    "mouthSmileLeft": 0.44,
    "mouthSmileRight": 0.42,
    "mouthStretchLeft": 0.0,
    "mouthStretchRight": 0.0,
    "mouthUpperUpLeft": 0.05,
    "mouthUpperUpRight": 0.06,
    "noseSneerLeft": 0.0,
    "noseSneerRight": 0.0,
    "browDownLeft": 0.1,
    "browDownRight": 0.09,
    "browInnerUp": 0.2,
    "browOuterUpLeft": 0.0,
    "browOuterUpRight": 0.0,
    "cheekPuff": 0.0,
    "cheekSquintLeft": 0.3,
    "cheekSquintRight": 0.28,
    "tongueOut": 0.0
  },
  "head_euler": {
    "pitch": -2.3,
    "yaw": 0.8,
    "roll": 0.1
  }
}

// Sent when user taps screen (manual morph advance, dev/testing)
{
  "type": "advance_morph"
}
```

### Mac → iPad

```json
// Sent after each SD generation cycle
{
  "type": "face_frame",
  "jpeg_b64": "<base64 encoded JPEG>",
  "morph_weight": 0.34,
  "frame_index": 47,
  "generation_ms": 4200
}

// Sent on connect so iPad can show connection state
{
  "type": "server_ready",
  "sd_model": "stabilityai/sdxl-turbo",
  "controlnet_model": "lllyasviel/control_v11p_sd15_openpose"
}

// Sent whenever morph_weight changes
{
  "type": "morph_update",
  "morph_weight": 0.34
}
```

---

## Mac Backend — Detailed Spec

### `main.py`

```python
# Startup sequence:
# 1. Load config from .env
# 2. Initialize SD pipeline (blocks until model loaded — can take 30-60s first run)
# 3. Initialize MorphState
# 4. Start FastAPI app with WebSocket endpoint at ws://0.0.0.0:8765
# 5. Log Mac's local IP so user knows what to put in iPad app
#
# On WebSocket connect:
# 1. Send server_ready message
# 2. Start generation loop in background executor
# 3. Begin receiving face_frame messages from iPad
#
# Generation loop (runs continuously while client connected):
# 1. Get latest blend_shapes from shared state (updated by WS handler)
# 2. Build conditioning image from blend_shapes + morph_weight
# 3. Run SD inference
# 4. Encode result as JPEG base64
# 5. Send face_frame to iPad
# 6. Sleep briefly, repeat
```

### `pipeline/generation.py`

**Purpose:** Wraps SD + ControlNet. Accepts a conditioning image and morph weight, returns a PIL Image.

**Model choice:**
- Primary: `stabilityai/sdxl-turbo` — 4 steps, no CFG, fast
- Fallback: `runwayml/stable-diffusion-v1-5` + LCM scheduler — more ControlNet compatibility

**ControlNet model:** `lllyasviel/control_v11p_sd15_openpose`
- Takes an OpenPose skeleton image as conditioning
- Face-specific variant: `CrucibleAI/ControlNetMediaPipeFace` (better for faces, try this first)

```python
class GenerationPipeline:
    def __init__(self):
        self.device = torch.device("mps")  # Apple Silicon GPU — never cuda, never cpu
        self.pipe = None                   # loaded in setup()
        self.session_seed = None           # fixed per session — set on first connect

    def setup(self):
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float16
        )
        # Load SD pipeline with ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(self.device)
        # Enable memory optimizations for M4
        self.pipe.enable_attention_slicing()

    def generate(self, conditioning_image: Image.Image, morph_weight: float) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(self.session_seed)
        result = self.pipe(
            prompt=FIXED_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=conditioning_image,
            num_inference_steps=SD_STEPS,
            guidance_scale=0.0,                          # SDXL-Turbo requires 0.0
            controlnet_conditioning_scale=morph_weight,  # ← THE core mechanism
            generator=generator,
            width=512,
            height=512,
        )
        return result.images[0]
```

**Fixed prompts (in `config.py`):**
```python
FIXED_PROMPT = (
    "photorealistic human face, neutral expression, "
    "soft studio lighting, symmetrical, sharp focus, "
    "high detail skin, 512x512"
)
NEGATIVE_PROMPT = (
    "cartoon, anime, painting, blurry, distorted, "
    "extra eyes, deformed, bad anatomy, watermark"
)
```

**The core mechanism — say this clearly:**
```
controlnet_conditioning_scale = morph_weight

At morph_weight = 0.0: ControlNet has zero influence.
SD generates from noise guided only by the text prompt.
Result: a random photorealistic face.

At morph_weight = 1.0: ControlNet fully controls structure.
SD must produce a face matching the OpenPose skeleton.
Result: a face with the user's geometry.

Between 0.0 and 1.0: blend.
The face drifts — slowly, imperceptibly frame to frame.
```

**Session seed logic:**
```python
# On new client connect, pick a random seed and lock it for the session
self.session_seed = random.randint(0, 2**32)
# Same seed every generation call = same "identity" drifts toward user
# Without this: face jumps to a different random person every frame
```

### `pipeline/conditioning.py`

**Purpose:** Converts ARKit blend shapes → a 512×512 OpenPose-style face skeleton image that ControlNet understands.

**What ControlNet expects:** A black image with colored dots/lines marking facial landmark positions. Standard OpenPose face format.

**Approach — build a 2D face landmark canvas:**

```python
# Base face template — normalized coordinates on 512x512
# These are neutral/rest positions, blend shapes offset from here

BASE_LANDMARKS = {
    # Eyes
    "left_eye_outer":  (0.32, 0.38),
    "left_eye_inner":  (0.42, 0.38),
    "left_eye_top":    (0.37, 0.35),
    "left_eye_bottom": (0.37, 0.41),
    "right_eye_outer": (0.68, 0.38),
    "right_eye_inner": (0.58, 0.38),
    "right_eye_top":   (0.63, 0.35),
    "right_eye_bottom":(0.63, 0.41),
    # Eyebrows
    "left_brow_inner": (0.40, 0.31),
    "left_brow_outer": (0.28, 0.32),
    "right_brow_inner":(0.60, 0.31),
    "right_brow_outer":(0.72, 0.32),
    # Nose
    "nose_tip":        (0.50, 0.52),
    "nose_left":       (0.44, 0.55),
    "nose_right":      (0.56, 0.55),
    # Mouth
    "mouth_left":      (0.38, 0.65),
    "mouth_right":     (0.62, 0.65),
    "mouth_top":       (0.50, 0.61),
    "mouth_bottom":    (0.50, 0.70),
    # Jaw / chin
    "chin":            (0.50, 0.82),
    "jaw_left":        (0.28, 0.72),
    "jaw_right":       (0.72, 0.72),
}

def blend_shapes_to_offsets(blend_shapes: dict) -> dict:
    """
    Map ARKit blend shape values to pixel offsets from BASE_LANDMARKS.
    Only implement the high-impact ones first.
    """
    offsets = {}

    # Jaw open — push mouth_bottom down, mouth_top stays
    jaw = blend_shapes.get("jawOpen", 0.0)
    offsets["mouth_bottom"] = (0, jaw * 30)  # up to 30px down

    # Mouth smile — pull corners outward and up
    smile_l = blend_shapes.get("mouthSmileLeft", 0.0)
    smile_r = blend_shapes.get("mouthSmileRight", 0.0)
    offsets["mouth_left"]  = (-smile_l * 15, -smile_l * 10)
    offsets["mouth_right"] = (smile_r * 15,  -smile_r * 10)

    # Brow raise — push brows up
    brow_up = blend_shapes.get("browInnerUp", 0.0)
    offsets["left_brow_inner"]  = (0, -brow_up * 12)
    offsets["right_brow_inner"] = (0, -brow_up * 12)

    # Brow down — push brows down and together
    brow_dl = blend_shapes.get("browDownLeft", 0.0)
    brow_dr = blend_shapes.get("browDownRight", 0.0)
    offsets["left_brow_outer"]  = (0, brow_dl * 10)
    offsets["right_brow_outer"] = (0, brow_dr * 10)

    # Eye blink — push eye top and bottom together
    blink_l = blend_shapes.get("eyeBlinkLeft", 0.0)
    blink_r = blend_shapes.get("eyeBlinkRight", 0.0)
    offsets["left_eye_top"]    = (0, blink_l * 8)
    offsets["left_eye_bottom"] = (0, -blink_l * 8)
    offsets["right_eye_top"]   = (0, blink_r * 8)
    offsets["right_eye_bottom"]= (0, -blink_r * 8)

    return offsets

def apply_head_rotation(landmarks: dict, pitch: float, yaw: float, roll: float) -> dict:
    """
    Rotate all landmark positions around center (0.5, 0.5) by head euler angles.
    Yaw: horizontal turn — compress x coords toward center on turn side
    Pitch: vertical tilt — shift all points up/down
    Roll: tilt — rotate around center
    Keep this simple: just roll rotation matrix + yaw x-compression
    """
    cx, cy = 0.5, 0.5
    roll_rad = math.radians(roll)
    rotated = {}
    for name, (x, y) in landmarks.items():
        # Roll
        dx, dy = x - cx, y - cy
        rx = dx * math.cos(roll_rad) - dy * math.sin(roll_rad) + cx
        ry = dx * math.sin(roll_rad) + dy * math.cos(roll_rad) + cy
        # Yaw compression (perspective shortcut)
        yaw_factor = 1.0 - abs(yaw) / 90.0 * 0.4
        rx = cx + (rx - cx) * yaw_factor + (yaw / 90.0) * 0.08
        # Pitch shift
        ry += (pitch / 45.0) * 0.05
        rotated[name] = (rx, ry)
    return rotated

def render_openpose_canvas(landmarks: dict) -> Image.Image:
    """
    Draw OpenPose-style face skeleton on black 512x512 canvas.
    Dots for landmarks, lines for connections.
    """
    img = Image.new("RGB", (512, 512), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    def px(coord):
        return (int(coord[0] * 512), int(coord[1] * 512))

    # Draw landmark dots
    for name, coord in landmarks.items():
        x, y = px(coord)
        draw.ellipse([x-4, y-4, x+4, y+4], fill=(255, 255, 255))

    # Draw connections
    connections = [
        ("left_eye_outer", "left_eye_inner"),
        ("right_eye_outer", "right_eye_inner"),
        ("left_brow_outer", "left_brow_inner"),
        ("right_brow_inner", "right_brow_outer"),
        ("mouth_left", "mouth_top"),
        ("mouth_top", "mouth_right"),
        ("mouth_right", "mouth_bottom"),
        ("mouth_bottom", "mouth_left"),
        ("nose_left", "nose_tip"),
        ("nose_tip", "nose_right"),
        ("jaw_left", "chin"),
        ("chin", "jaw_right"),
    ]
    for a, b in connections:
        if a in landmarks and b in landmarks:
            draw.line([px(landmarks[a]), px(landmarks[b])], fill=(180, 180, 180), width=2)

    return img

def build_conditioning_image(blend_shapes: dict, head_euler: dict) -> Image.Image:
    """Main entry point. Call this from generation.py."""
    offsets = blend_shapes_to_offsets(blend_shapes)
    landmarks = {}
    for name, (bx, by) in BASE_LANDMARKS.items():
        ox, oy = offsets.get(name, (0, 0))
        landmarks[name] = (bx + ox/512, by + oy/512)
    landmarks = apply_head_rotation(
        landmarks,
        head_euler.get("pitch", 0),
        head_euler.get("yaw", 0),
        head_euler.get("roll", 0)
    )
    return render_openpose_canvas(landmarks)
```

### `pipeline/morph.py`

```python
import time
import random

class MorphState:
    def __init__(self):
        self.weight: float = 0.0
        self.target_weight: float = 0.0
        self.session_start: float = time.time()
        self.frame_count: int = 0

    def advance(self, increment: float = 0.018):
        """Call this to nudge morph forward (e.g. on timer or manual trigger)."""
        self.target_weight = min(1.0, self.target_weight + increment)

    def get_weight(self) -> float:
        """
        Smoothly interpolate toward target_weight.
        Add tiny noise so movement feels organic, not mechanical.
        """
        self.weight += (self.target_weight - self.weight) * 0.05
        noise = random.gauss(0, 0.003)
        return max(0.0, min(1.0, self.weight + noise))

    def reset(self):
        self.weight = 0.0
        self.target_weight = 0.0
        self.frame_count = 0
        self.session_start = time.time()
```

**Morph advance strategy** — choose one for initial build, tune later:

```python
# Option A: Time-based (advances automatically every N seconds)
# Simple, no user input needed, good for unattended installation
if time.time() - last_advance > MORPH_ADVANCE_INTERVAL_SECONDS:
    morph_state.advance(MORPH_INCREMENT)
    last_advance = time.time()

# Option B: Frame-based (advances every N generated frames)
if frame_count % MORPH_ADVANCE_EVERY_N_FRAMES == 0:
    morph_state.advance(MORPH_INCREMENT)

# Recommended starting values:
MORPH_ADVANCE_INTERVAL_SECONDS = 15   # advance every 15 seconds
MORPH_INCREMENT = 0.018               # reaches 1.0 in ~15 minutes
```

### `websocket/server.py`

```python
# Single WebSocket connection expected (one client at a time)
# Shared state between WS handler and generation loop:

latest_blend_shapes: dict = {}   # updated by WS receive handler
latest_head_euler: dict = {}     # updated by WS receive handler
generation_queue: asyncio.Queue  # generation loop reads from this

# WS receive handler — runs in async loop
async def handle_message(message: str):
    data = json.loads(message)
    msg_type = data.get("type")

    if msg_type == "face_frame":
        latest_blend_shapes.update(data["blend_shapes"])
        latest_head_euler.update(data["head_euler"])

    elif msg_type == "advance_morph":
        morph_state.advance()
        await send_morph_update()

# Generation loop — runs in thread executor
def generation_loop():
    while running:
        blend_shapes = latest_blend_shapes.copy()
        head_euler = latest_head_euler.copy()
        weight = morph_state.get_weight()

        if blend_shapes:
            conditioning = build_conditioning_image(blend_shapes, head_euler)
        else:
            conditioning = blank_canvas()  # black image — SD ignores ControlNet

        image = pipeline.generate(conditioning, weight)
        jpeg_b64 = face_utils.encode_jpeg(image)

        asyncio.run_coroutine_threadsafe(
            send_face_frame(jpeg_b64, weight, frame_index),
            event_loop
        )
        frame_index += 1

        # Auto-advance morph on timer
        nonlocal last_advance
        if time.time() - last_advance > MORPH_ADVANCE_INTERVAL_SECONDS:
            morph_state.advance()
            last_advance = time.time()
```

### `pipeline/face_utils.py`

```python
import base64
import io
from PIL import Image

def encode_jpeg(image: Image.Image, quality: int = 85) -> str:
    """Encode PIL image to base64 JPEG string for WebSocket transmission."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def blank_canvas(size: int = 512) -> Image.Image:
    """All-black image. ControlNet ignores this — SD runs freely."""
    return Image.new("RGB", (size, size), (0, 0, 0))
```

### `config.py`

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Server
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))

# Models
SD_MODEL_ID         = os.getenv("SD_MODEL_ID", "stabilityai/sdxl-turbo")
CONTROLNET_MODEL_ID = os.getenv("CONTROLNET_MODEL_ID", "CrucibleAI/ControlNetMediaPipeFace")

# Generation
SD_STEPS            = int(os.getenv("SD_STEPS", "4"))
IMAGE_SIZE          = int(os.getenv("IMAGE_SIZE", "512"))
FIXED_PROMPT        = "photorealistic human face, neutral expression, soft studio lighting, symmetrical, sharp focus, high detail skin"
NEGATIVE_PROMPT     = "cartoon, anime, painting, blurry, distorted, extra eyes, deformed, bad anatomy, watermark, text"

# Morph
MORPH_ADVANCE_INTERVAL_SECONDS = float(os.getenv("MORPH_INTERVAL", "15"))
MORPH_INCREMENT                = float(os.getenv("MORPH_INCREMENT", "0.018"))
```

---

## iPad App — Detailed Spec

### Permissions Required (`Info.plist`)

```xml
<key>NSCameraUsageDescription</key>
<string>Used for face tracking to guide the generative experience.</string>
```

No microphone permission needed in this version.

### `App/AppState.swift`

```swift
import SwiftUI
import Combine

class AppState: ObservableObject {
    // Connection
    @Published var connected: Bool = false

    // Display
    @Published var currentFaceImage: UIImage? = nil
    @Published var morphWeight: Double = 0.0
    @Published var frameIndex: Int = 0
    @Published var generationMs: Int = 0

    // Face tracking
    @Published var faceDetected: Bool = false

    // Debug
    @Published var fps: Double = 0.0
    @Published var blendShapeCount: Int = 0

    // Shared instances
    let wsClient = WSClient()
    let faceTracker = FaceTracker()
}
```

### `ARKit/FaceTracker.swift`

```swift
import ARKit
import simd

class FaceTracker: NSObject, ObservableObject, ARSessionDelegate {
    private let session = ARSession()
    var onBlendShapes: (([String: Float], simd_float3) -> Void)?

    // Throttle: only send at 10fps
    private var lastSendTime: Date = .distantPast
    private let sendInterval: TimeInterval = 0.1  // 10fps

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("ARFaceTracking not supported — requires TrueDepth camera")
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false  // not needed
        session.delegate = self
        session.run(config)
    }

    func stop() {
        session.pause()
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.first(where: { $0 is ARFaceAnchor }) as? ARFaceAnchor else { return }

        let now = Date()
        guard now.timeIntervalSince(lastSendTime) >= sendInterval else { return }
        lastSendTime = now

        // Extract all 52 blend shapes as [String: Float]
        var blendShapes: [String: Float] = [:]
        for (key, value) in faceAnchor.blendShapes {
            blendShapes[key.rawValue] = value.floatValue
        }

        // Extract head euler angles from transform matrix
        let transform = faceAnchor.transform
        let eulerAngles = extractEulerAngles(from: transform)

        onBlendShapes?(blendShapes, eulerAngles)
    }

    private func extractEulerAngles(from transform: simd_float4x4) -> simd_float3 {
        // Extract pitch, yaw, roll from 4x4 transform matrix
        let pitch = asin(-transform.columns.2.y)
        let yaw   = atan2(transform.columns.2.x, transform.columns.2.z)
        let roll  = atan2(transform.columns.0.y, transform.columns.1.y)
        // Convert radians to degrees
        return simd_float3(
            pitch * (180 / .pi),
            yaw   * (180 / .pi),
            roll  * (180 / .pi)
        )
    }
}
```

### `ARKit/BlendShapeMapper.swift`

```swift
import Foundation

struct BlendShapeMapper {
    // Converts ARKit blend shape dict + euler angles to the protocol JSON payload
    static func toMessage(blendShapes: [String: Float], eulerAngles: SIMD3<Float>) -> [String: Any] {
        return [
            "type": "face_frame",
            "timestamp": Date().timeIntervalSince1970,
            "blend_shapes": blendShapes,
            "head_euler": [
                "pitch": eulerAngles.x,
                "yaw":   eulerAngles.y,
                "roll":  eulerAngles.z
            ]
        ]
    }
}
```

### `Network/WSClient.swift`

```swift
import Foundation

class WSClient: NSObject, ObservableObject, URLSessionWebSocketDelegate {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession!
    var onMessage: ((String) -> Void)?
    var onConnected: (() -> Void)?
    var onDisconnected: (() -> Void)?

    // Set this to your Mac's local IP before building
    // Run: ipconfig getifaddr en0 in Mac terminal
    private let serverURL = URL(string: "ws://192.168.x.x:8765")!

    override init() {
        super.init()
        urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func connect() {
        webSocketTask = urlSession.webSocketTask(with: serverURL)
        webSocketTask?.resume()
        listen()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
    }

    func send(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let string = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(string)) { error in
            if let error { print("WS send error: \(error)") }
        }
    }

    private func listen() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text): self?.onMessage?(text)
                case .data(_): break  // no binary in this version
                @unknown default: break
                }
                self?.listen()  // recurse to keep listening
            case .failure(let error):
                print("WS receive error: \(error)")
                self?.scheduleReconnect()
            }
        }
    }

    private func scheduleReconnect() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in
            self?.connect()
        }
    }

    // MARK: - URLSessionWebSocketDelegate
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        DispatchQueue.main.async { self.onConnected?() }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        DispatchQueue.main.async { self.onDisconnected?() }
    }
}
```

### `Network/MessageHandler.swift`

```swift
import Foundation
import UIKit

class MessageHandler {
    weak var appState: AppState?

    func handle(_ message: String) {
        guard let data = message.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        DispatchQueue.main.async { [weak self] in
            switch type {
            case "server_ready":
                self?.appState?.connected = true

            case "face_frame":
                if let b64 = json["jpeg_b64"] as? String,
                   let imageData = Data(base64Encoded: b64),
                   let image = UIImage(data: imageData) {
                    self?.appState?.currentFaceImage = image
                }
                if let weight = json["morph_weight"] as? Double {
                    self?.appState?.morphWeight = weight
                }
                if let idx = json["frame_index"] as? Int {
                    self?.appState?.frameIndex = idx
                }
                if let ms = json["generation_ms"] as? Int {
                    self?.appState?.generationMs = ms
                }

            case "morph_update":
                if let weight = json["morph_weight"] as? Double {
                    self?.appState?.morphWeight = weight
                }

            default:
                print("Unknown message type: \(type)")
            }
        }
    }
}
```

### `Views/FaceDisplayView.swift`

```swift
import SwiftUI

struct FaceDisplayView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image = appState.currentFaceImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .ignoresSafeArea()
                    .animation(.easeInOut(duration: 0.3), value: appState.frameIndex)
                    .transition(.opacity)
            } else {
                // Waiting for first frame
                VStack {
                    ProgressView()
                        .tint(.white)
                    Text(appState.connected ? "Generating..." : "Connecting...")
                        .foregroundColor(.white.opacity(0.5))
                        .font(.caption)
                        .padding(.top, 8)
                }
            }
        }
    }
}
```

### `Views/DebugOverlayView.swift`

```swift
// Show during development, hide for installation
import SwiftUI

struct DebugOverlayView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Connected: \(appState.connected ? "✓" : "✗")")
            Text("Face: \(appState.faceDetected ? "✓" : "✗")")
            Text("Morph: \(String(format: "%.3f", appState.morphWeight))")
            Text("Frame: \(appState.frameIndex)")
            Text("Gen: \(appState.generationMs)ms")
        }
        .font(.system(size: 12, design: .monospaced))
        .foregroundColor(.green)
        .padding(8)
        .background(Color.black.opacity(0.6))
        .cornerRadius(6)
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
```

### `Views/SessionView.swift`

```swift
import SwiftUI

struct SessionView: View {
    @EnvironmentObject var appState: AppState
    private let messageHandler = MessageHandler()

    var body: some View {
        ZStack {
            FaceDisplayView()
            #if DEBUG
            DebugOverlayView()
            #endif
        }
        .onAppear { setup() }
        .onTapGesture { appState.wsClient.send(["type": "advance_morph"]) }
    }

    private func setup() {
        messageHandler.appState = appState

        // Wire up WebSocket
        appState.wsClient.onMessage = { [weak messageHandler] msg in
            messageHandler?.handle(msg)
        }
        appState.wsClient.onConnected = {
            DispatchQueue.main.async { appState.connected = true }
        }
        appState.wsClient.onDisconnected = {
            DispatchQueue.main.async { appState.connected = false }
        }
        appState.wsClient.connect()

        // Wire up face tracker
        appState.faceTracker.onBlendShapes = { blendShapes, eulerAngles in
            DispatchQueue.main.async { appState.faceDetected = true }
            let message = BlendShapeMapper.toMessage(
                blendShapes: blendShapes,
                eulerAngles: eulerAngles
            )
            appState.wsClient.send(message)
        }
        appState.faceTracker.start()
    }
}
```

---

## Dependencies

### Mac (`requirements.txt`)

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
websockets==12.0
pydantic==2.7.0

torch==2.3.0
diffusers==0.29.0
transformers==4.41.0
accelerate==0.30.0
controlnet-aux==0.0.9
Pillow==10.3.0
numpy==1.26.4
opencv-python==4.9.0.80

python-dotenv==1.0.1
aiofiles==23.2.1
```

### Mac — System (Homebrew)

```bash
# Nothing extra needed for this scope
# opencv-python handles image ops
```

### iPad (Swift)

No external Swift packages. All system frameworks:
- `ARKit`
- `SwiftUI`
- `Foundation` (URLSession WebSocket)
- `AVFoundation` (not used in this version)

---

## Environment Variables (`.env`)

```bash
# mac-server/.env

WS_HOST=0.0.0.0
WS_PORT=8765

SD_MODEL_ID=stabilityai/sdxl-turbo
CONTROLNET_MODEL_ID=CrucibleAI/ControlNetMediaPipeFace

SD_STEPS=4
IMAGE_SIZE=512

MORPH_INTERVAL=15
MORPH_INCREMENT=0.018
```

---

## Network Setup

```bash
# Step 1 — find your Mac's local IP
ipconfig getifaddr en0
# Example output: 192.168.1.42

# Step 2 — update WSClient.swift
# Change this line:
private let serverURL = URL(string: "ws://192.168.x.x:8765")!
# To:
private let serverURL = URL(string: "ws://192.168.1.42:8765")!

# Step 3 — check Mac firewall
# System Settings → Network → Firewall → Options
# Add uvicorn / python3 to allowed incoming connections
# OR temporarily disable firewall during dev

# Step 4 — verify connection before running full app
# On Mac:
python3 -c "
import asyncio, websockets
async def echo(ws):
    async for msg in ws: await ws.send(msg)
asyncio.run(websockets.serve(echo, '0.0.0.0', 8765))
"
# On iPad: use 'Socket - WebSocket Client' from App Store
# Connect to ws://192.168.1.42:8765 and send test message
# Confirm echo before building real client
```

---

## Build Phases

### Phase 0 — Smoke Test SD on Mac

```bash
cd mac-server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 -c "
import torch
from diffusers import AutoPipelineForText2Image

assert torch.backends.mps.is_available(), 'MPS not available — check PyTorch version'

pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    torch_dtype=torch.float16
).to('mps')

img = pipe(
    'photorealistic human face, neutral expression, studio lighting',
    num_inference_steps=4,
    guidance_scale=0.0
).images[0]

img.save('smoke_test.jpg')
print('Success — open smoke_test.jpg')
"
# Target: image generated in under 8 seconds on M4
```

### Phase 1 — SD → WebSocket → iPad display

**Goal:** iPad shows a generated face. No face tracking yet. Prove the pipeline end to end.

```
Mac: WS server generates one face on connect, sends face_frame every 5s
iPad: WSClient connects, MessageHandler decodes JPEG, FaceDisplayView renders it
Success: face appears on iPad screen and updates every ~5 seconds
```

### Phase 2 — ARKit → WebSocket → Mac (receive only)

**Goal:** Mac receives blend shapes. iPad does not change visually yet.

```
iPad: FaceTracker starts, captures blend shapes at 10fps
iPad: BlendShapeMapper serializes → WSClient sends face_frame JSON
Mac: WS handler receives, logs blend_shapes dict to console
Success: Mac terminal shows 52 float values updating in real time
         Values change when you open your mouth, raise brows, etc.
```

### Phase 3 — Blend shapes → ControlNet conditioning

**Goal:** Blend shapes affect the conditioning image. Morph weight fixed at 0.5 for testing.

```
Mac: conditioning.py converts latest blend_shapes → OpenPose canvas
Mac: Save canvas to disk after each receive — inspect it visually
Mac: Pass canvas to SD with controlnet_conditioning_scale=0.5
Mac: Send result to iPad
Success: Different facial expressions on iPad → different conditioning canvases
         Generated faces vary subtly based on your expression
         Head turn left → generated face looks slightly left
```

### Phase 4 — Morph state machine

**Goal:** Morph weight increases over time. Face visibly drifts toward user's geometry.

```
Mac: MorphState initialized at 0.0 on connect
Mac: Auto-advances every 15 seconds
Mac: controlnet_conditioning_scale = morph_state.get_weight()
iPad: Tap screen sends advance_morph for manual testing
iPad: DebugOverlay shows morph_weight ticking up
Success: At weight=0.0 face is random
         At weight=0.5 face has user's rough proportions
         At weight=1.0 face matches user's geometry
```

### Phase 5 — Polish

```
- Tune MORPH_INTERVAL and MORPH_INCREMENT for desired pace
- Tune JPEG quality vs frame delivery speed
- Verify crossfade between frames feels smooth
- Test in varied lighting (TrueDepth is IR so lighting independent)
- Test with different faces — verify conditioning generalizes
- Remove debug overlay for installation mode
- Make IP configurable without recompiling (UserDefaults settings screen)
```

---

## Common Failure Modes

| Problem | Likely Cause | Fix |
|---|---|---|
| SD running slow | On CPU not MPS | `assert torch.backends.mps.is_available()` at startup |
| ARKit not working | Running on Simulator | Deploy to physical iPad — Simulator has no TrueDepth |
| WebSocket won't connect | Mac firewall | Allow Python/uvicorn in System Settings → Firewall |
| Face doesn't change with expressions | Conditioning image not wired to ControlNet | Log conditioning image to disk, inspect it looks right |
| Generated face flickers to different person | No fixed seed | Ensure `session_seed` is set once and reused every call |
| ControlNet ignored | Wrong conditioning scale | Verify `controlnet_conditioning_scale` is not 0.0 |
| Blend shapes all zero | ARKit session not running | Check `ARFaceTrackingConfiguration.isSupported` returns true |
| Head rotation wrong direction | Euler angle sign flip | Negate pitch or yaw in `apply_head_rotation` and retest |
| iPad can't find Mac | Different subnets | Confirm both devices on same WiFi network (not guest vs main) |

---

## Notes for Claude Code

- **Never call SD inference from the async WebSocket handler.** It blocks. Always run in `asyncio.get_event_loop().run_in_executor(None, blocking_fn)` or a dedicated thread.
- **MPS device only.** `torch.device("mps")`. Do not use `"cuda"`. Do not fall back to `"cpu"` silently — fail loudly if MPS unavailable so it's obvious.
- **ARKit requires physical device.** `ARFaceTrackingConfiguration.isSupported` returns false on Simulator. Build and run target must be the real iPad Pro, connected via USB with Developer Mode enabled.
- **Fixed seed per session is critical.** Without it, the base face identity changes every SD call and the morph looks like random flickering instead of drift. Set `session_seed = random.randint(0, 2**32)` once when client connects, reuse for every generation call in that session.
- **`controlnet_conditioning_scale = morph_weight`** is the entire morph mechanism. One parameter. Everything else is infrastructure to feed it correctly.
- **Start conditioning with just jawOpen + headYaw.** Get those two working and visible before implementing all 52 blend shapes. Verify the conditioning canvas looks correct (save to disk, open in Preview) before connecting it to SD.
- **10fps throttle on ARKit is correct.** 60fps over WebSocket would saturate the connection and the Mac can't generate faster than ~4fps anyway. Throttle in `FaceTracker.swift`, not on the Mac side.
- **JPEG quality 85** is the right tradeoff. Quality 95+ makes frames too large for smooth WebSocket delivery. Quality 70 shows compression artifacts on skin.
