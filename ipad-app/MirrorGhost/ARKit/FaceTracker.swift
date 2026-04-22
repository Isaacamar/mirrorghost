import ARKit
import simd

final class FaceTracker: NSObject, ObservableObject, ARSessionDelegate {

    private let session = ARSession()

    var onFrame:  ((FaceFrame) -> Void)?
    var onNoFace: (() -> Void)?

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("[FaceTracker] ARFaceTracking not supported — requires TrueDepth camera")
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() { session.pause() }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard
            let anchor = anchors.first(where: { $0 is ARFaceAnchor }) as? ARFaceAnchor,
            let frame  = session.currentFrame
        else {
            onNoFace?()
            return
        }

        let canvasSize = CGSize(width: 512, height: 512)

        // Project vertices + render wireframe in one pass — reuse screenVertices for
        // the isolation mask so we don't project twice.
        let meshOutput = FaceMeshRenderer.render(
            geometry:   anchor.geometry,
            transform:  anchor.transform,
            camera:     frame.camera,
            outputSize: canvasSize
        )

        let isolMask = FaceIsolator.mask(
            geometry:      anchor.geometry,
            screenVertices: meshOutput.screenVertices,
            canvasSize:    canvasSize
        )

        let crop = FaceIsolator.alignedCrop(anchor: anchor, frame: frame)

        let faceFrame = FaceFrame(
            timestamp:         frame.timestamp,
            transform:         anchor.transform,
            euler:             euler(from: anchor.transform),
            leftEyeTransform:  anchor.leftEyeTransform,
            rightEyeTransform: anchor.rightEyeTransform,
            blendShapes:       BlendShapeSet(from: anchor),
            screenVertices:    meshOutput.screenVertices,
            wireframe:         meshOutput.image,
            isolationMask:     isolMask,
            alignedCrop:       crop,
            depthAvailable:    frame.capturedDepthData != nil
        )

        onFrame?(faceFrame)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[FaceTracker] session error: \(error)")
    }

    // MARK: - Euler angles (degrees) from head pose matrix

    private func euler(from m: simd_float4x4) -> SIMD3<Float> {
        let d: Float = 180 / .pi
        return SIMD3<Float>(
            asin(-m.columns.2.y)                     * d,
            atan2(m.columns.2.x, m.columns.2.z)     * d,
            atan2(m.columns.0.y, m.columns.1.y)     * d
        )
    }
}
