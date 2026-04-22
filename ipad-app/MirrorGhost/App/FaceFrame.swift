import ARKit
import UIKit

struct FaceFrame {
    let timestamp: TimeInterval

    // Head pose
    let transform: simd_float4x4
    let euler: SIMD3<Float>             // pitch, yaw, roll — degrees

    // Eye gaze (world-space transforms, translation = eye center position)
    let leftEyeTransform:  simd_float4x4
    let rightEyeTransform: simd_float4x4

    // Animation driving data — typed, no string lookups at 30fps
    let blendShapes: BlendShapeSet

    // Projected geometry onto the 512×512 conditioning canvas
    let screenVertices: [CGPoint]       // 1,220 points, same order as ARFaceGeometry.vertices

    // --- Rendered outputs (computed once per frame in FaceTracker) ---

    // 512×512 gray wireframe on black — ControlNet conditioning input
    let wireframe: UIImage

    // 512×512 white filled face mask on black — face region for downstream isolation
    let isolationMask: UIImage

    // Eye-aligned face crop from the camera image, ~512×512.
    // nil when face geometry is degenerate (extreme angle, too close, etc.)
    // Sent to InsightFace / IP-Adapter at 1fps for identity extraction.
    let alignedCrop: UIImage?

    // Whether the TrueDepth depth map was available this frame.
    // Depth fusion is a future refinement; this flag tells us if the hardware supports it.
    let depthAvailable: Bool
}
