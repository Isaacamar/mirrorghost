# mirrorghost

A real-time generative face installation. A person stands in front of an iPad Pro. On screen, their face slowly builds from TV static — recognizable but wrong, always mid-process, never fully resolving. Intended for projection onto a mannequin head.

---

## Concept

The system runs two interlocked loops:

**Slow loop — identity generation (~30s cycle, Mac)**
Stable Diffusion 1.5 + IP-Adapter FaceID generates a portrait conditioned on the viewer's face embedding. The generated image is not a photograph — it is a hallucinated interpretation of the face, constrained by an identity signal that accumulates over the session. Early in a session the face is ambiguous. Over time it becomes more specifically *you*.

**Fast loop — reenactment (30fps, iPad)**
A Core ML model (future) animates the generated portrait in real time from ARKit face tracking data. The portrait moves when you move. The uncanny valley is the point.

TV static overlays the output, inversely proportional to identity confidence. At zero confidence: pure noise. At full confidence: a minimum residual grain that never disappears.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  iPad Pro M4                                                │
│                                                             │
│  ARKit (TrueDepth)                                          │
│    └── FaceFrame ──────────────────────────────────────────►│
│         ├── wireframe image (512×512, ControlNet input)     │
│         ├── isolation mask  (filled face region)            │
│         ├── aligned crop    (eye-aligned, InsightFace input)│
│         ├── BlendShapeSet   (52 typed coefficients)         │
│         ├── euler angles    (pitch / yaw / roll)            │
│         └── eye transforms  (left + right gaze vectors)     │
│                                                             │
│  [future] Core ML reenactment model                         │
│    └── drives the generated portrait at 30fps               │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket (LAN WiFi)
                           │ face mesh JPEG + blend shapes
                           │ ← generated portrait JPEG
┌──────────────────────────▼──────────────────────────────────┐
│  Mac (Apple Silicon M-series)                               │
│                                                             │
│  InsightFace                                                │
│    └── ArcFace 512-dim identity embedding from face crop    │
│                                                             │
│  FairFace (future)                                          │
│    └── age / gender → text prompt tokens                    │
│                                                             │
│  SD 1.5 + IP-Adapter FaceID + ControlNet                    │
│    ├── identity: face embedding (accumulates over session)  │
│    ├── conditioning: wireframe mesh image                   │
│    └── prompt: measured physical attributes                 │
│                                                             │
│  Generates every ~30s → JPEG → WebSocket → iPad             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Voice (in progress — see voice/)                           │
│  Teammate component. Feeds into prompt conditioning.        │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
mirrorghost/
├── ipad-app/                  iPad Swift/SwiftUI app (Xcode project)
│   ├── MirrorGhost.xcodeproj
│   ├── project.yml            xcodegen spec — regenerate with `xcodegen generate`
│   └── MirrorGhost/
│       ├── ARKit/
│       │   ├── FaceTracker.swift       ARKit session + ARSessionDelegate
│       │   ├── FaceMeshRenderer.swift  1,220-vertex wireframe → 512×512 UIImage
│       │   └── FaceIsolator.swift      Filled mask + eye-aligned crop
│       ├── App/
│       │   ├── MirrorGhostApp.swift    @main entry point
│       │   ├── TrackingState.swift     @MainActor ObservableObject
│       │   ├── FaceFrame.swift         Central per-frame data snapshot
│       │   └── BlendShapeSet.swift     Typed struct for all 52 ARKit blend shapes
│       └── Views/
│           └── RootView.swift          Wireframe | mask | crop + debug overlay
│
├── mac-server/                Mac inference server (Python)
│   ├── main.py                FastAPI + WebSocket entry point
│   ├── config.py              All tuning constants
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── generation.py      SD 1.5 + ControlNet + IP-Adapter FaceID
│   │   ├── conditioning.py    Wireframe mesh → ControlNet input
│   │   ├── embeddings.py      InsightFace identity accumulation
│   │   ├── morph.py           Morph weight state machine
│   │   └── face_utils.py      PIL helpers, JPEG encode
│   └── websocket/
│       ├── server.py          WebSocket connection handler
│       └── protocol.py        Pydantic message schemas
│
├── voice/                     Voice component (teammate — in progress)
│
├── models/
│   └── face_landmarker.task   MediaPipe face landmarker task file
│
├── archive/                   Previous iterations (reference only)
│   ├── ipad-app-v1/           Original Mirror app (no FaceFrame, no isolation)
│   └── CLAUDE_face_only.md    Early build spec
│
├── reflection.py              Standalone webcam version (no iPad required)
├── download_models.py         Script to pre-download HuggingFace models
├── requirements.txt           Top-level Python dependencies
└── .gitignore
```

---

## Hardware

| Device | Role |
|---|---|
| iPad Pro M4 | TrueDepth face tracking, full-screen display, future Core ML reenactment |
| MacBook Pro (Apple Silicon) | SD inference on MPS, InsightFace embedding, WebSocket server |
| Local WiFi | LAN bridge — no internet required during operation |

---

## iPad App — Current State

**Step 1 is complete.** The iPad app does face tracking only — no networking, no model inference.

Every ARKit frame produces a `FaceFrame` containing:

| Field | Description |
|---|---|
| `wireframe` | 512×512 UIImage — 1,220 ARKit face vertices projected to screen, white dots + gray triangle mesh on black. This is the ControlNet conditioning input. |
| `isolationMask` | 512×512 UIImage — all face triangles filled white on black. Defines the face region for downstream processing. |
| `alignedCrop` | UIImage? — Eye-aligned face crop from the TrueDepth camera image (2.8× inter-ocular distance square, eyes at 38% from top). InsightFace input. `nil` if head angle is degenerate. |
| `blendShapes` | `BlendShapeSet` — typed struct with all 52 ARKit blend shape coefficients. No string lookups at runtime. |
| `euler` | `SIMD3<Float>` — pitch, yaw, roll in degrees. |
| `leftEyeTransform` / `rightEyeTransform` | Full 4×4 eye pose transforms — gaze direction. |
| `screenVertices` | `[CGPoint]` — all 1,220 vertices projected to 512×512 canvas. Shared between wireframe renderer and isolation mask renderer. |
| `depthAvailable` | `Bool` — whether `ARFrame.capturedDepthData` was populated. Should be `true` on iPad Pro M4. |

The debug view shows the wireframe and isolation mask side-by-side, the aligned crop as a 180×180 thumbnail, live euler angles, fps, and the top 8 blend shapes as bar charts.

### Xcode Setup

Requirements: Xcode 16+, iPad Pro with TrueDepth camera, macOS 15.

```bash
# If you have xcodegen installed:
cd ipad-app
xcodegen generate

# Otherwise, open the existing project:
open ipad-app/MirrorGhost.xcodeproj
```

1. Open `MirrorGhost.xcodeproj` in Xcode
2. **Signing & Capabilities** → set your development team
3. Select your iPad as the run destination
4. Build and run

The ARKit face tracking session starts immediately on launch. No connection required for this step.

**Constraint:** `ARFaceTrackingConfiguration` requires the TrueDepth (front) camera. Simulator will not work.

### Key implementation notes

- `session(_:didUpdate:anchors:)` does not give you the camera — use `session.currentFrame` to get `ARCamera` for `projectPoint`
- `ARFaceGeometry.triangleIndices` is `UnsafeBufferPointer<Int16>` — index directly, don't convert to Array
- `ObservableObject` + `@Published` throughout — no `@Observable` macro (Swift 6 compatibility)
- `CIContext` is shared as `nonisolated(unsafe) static let` — it's thread-safe per Apple docs but doesn't conform to `Sendable`
- All `@MainActor`-isolated callbacks use `Task { @MainActor [weak self] in }` not `DispatchQueue.main.async`

---

## Mac Server — Current State

The Python server is in `mac-server/`. It handles:

- WebSocket connection from iPad
- Receiving face mesh JPEGs and blend shapes at 10fps
- Running InsightFace for identity embedding extraction
- Running SD 1.5 + IP-Adapter FaceID + ControlNet for portrait generation
- Sending generated portraits back to iPad

```bash
cd mac-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Server prints the Mac's local IP on startup. Enter that IP in the iPad app connect screen.

### Critical constraints (learned the hard way)

| Constraint | Reason |
|---|---|
| `torch_dtype=torch.float32` | float16 produces black images on MPS |
| No `enable_attention_slicing()` | Destroys `IPAdapterAttnProcessor` at runtime |
| `guidance_scale=1.0` for LCM | LCM requires this (0.0 is SDXL-Turbo's requirement) |
| Load all ML models before tkinter | Avoids Metal GPU conflict |
| Fixed seed per session | Without it, base face identity jumps every frame instead of drifting |

---

## Standalone Version

`reflection.py` runs the full generative loop from a webcam, no iPad required.

```bash
source .venv/bin/activate
python reflection.py
```

| Key | Action |
|---|---|
| `R` | Noise flood — resets embedding accumulator and canvas |
| `Q` / `Escape` | Quit |

---

## Voice Component

`voice/` is reserved for the voice integration (teammate in progress). This component will feed into the prompt conditioning layer on the Mac — spoken content or voice characteristics may influence the generated portrait's affect or style.

---

## Data Pipeline — Physical Feature Extraction (planned)

The current IP-Adapter FaceID pipeline can misattribute demographic features because the ArcFace embedding is optimized for identity recognition, not demographic description. When the embedding signal is weak (misaligned crop, extreme angle), SD 1.5 falls back to its training distribution.

The planned fix is a layered measurement approach:

**Layer 1 — objective measurement (iPad, no model)**
- Skin tone: sample cheek pixels through the isolation mask → LAB colorspace → ITA angle → Fitzpatrick-scale descriptor
- Face geometry ratios from mesh: inter-ocular distance, jaw angle, brow prominence — computed from known vertex positions in 3D
- Eye / hair color: sample from mesh-guided regions

**Layer 2 — attribute classification (Mac, alongside InsightFace)**
- [FairFace](https://github.com/joojs/fairface) for age group and perceived gender → text tokens injected into SD prompt
- These are the two attributes that can't be measured directly

**Layer 3 — explicit SD conditioning**
Measured attributes become text tokens: `"portrait of a 30-year-old [gender], [skin_tone] skin tone"`. This overrides the model's distributional default on every generation instead of relying on the embedding alone.

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Complete | iPad face tracking — FaceFrame data pipeline, wireframe, isolation mask, aligned crop |
| 2 | Planned | Mac connection — WebSocket, send FaceFrame data, receive generated portrait |
| 3 | Planned | SD generation loop — ControlNet from wireframe, IP-Adapter FaceID from aligned crop |
| 4 | Planned | Physical attribute measurement — skin tone, geometry ratios, FairFace |
| 5 | Planned | Core ML reenactment — fast loop animating portrait at 30fps from blend shapes |
| 6 | Planned | Voice integration — teammate component wired into prompt conditioning |

---

## Models

| Model | Size | Location |
|---|---|---|
| `runwayml/stable-diffusion-v1-5` | ~4GB | `~/.cache/huggingface/hub/` |
| `h94/IP-Adapter-FaceID` | ~200MB | `~/.cache/huggingface/hub/` |
| `latent-consistency/lcm-lora-sdv1-5` | ~67MB | `~/.cache/huggingface/hub/` |
| InsightFace `buffalo_l` | ~500MB | `~/.insightface/models/` |
| MediaPipe face landmarker | 3.6MB | `models/face_landmarker.task` |

First run downloads ~6GB from HuggingFace. Use `python download_models.py` to pre-fetch.

---

## Dependencies

### Mac (Python)

```
torch (MPS, float32)
diffusers + transformers + accelerate + peft
insightface
fastapi + uvicorn[standard]
opencv-python-headless
pillow
python-dotenv
```

See `mac-server/requirements.txt` and top-level `requirements.txt`.

### iPad (Swift)

No external packages. System frameworks only:
- `ARKit`
- `SwiftUI`
- `CoreImage`
- `AVFoundation`
- `simd`

---

## WebSocket Protocol

```
iPad → Mac  (10fps)
  wireframe JPEG (multipart or binary frame)
  { type: "face_data", blend_shapes: {...52...}, euler: {pitch, yaw, roll} }

Mac → iPad
  { type: "portrait", jpeg_b64: "...", generation_ms: 4200 }
```

Full protocol spec TBD when networking phase begins.
