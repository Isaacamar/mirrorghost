import ARKit
import CoreImage
import UIKit
import simd

class FaceTracker: NSObject, ObservableObject, ARSessionDelegate {
    private let session   = ARSession()
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let sendQueue = DispatchQueue(label: "facetracker.send", qos: .userInteractive)

    // Blend shapes + euler → sent at 10fps
    private var lastBlendSendTime: Date = .distantPast
    private let blendSendInterval: TimeInterval = 0.1

    // Face image → sent at 1fps for InsightFace on Mac
    private var lastImageSendTime: Date = .distantPast
    private let imageSendInterval: TimeInterval = 1.0

    var onBlendShapes: (([String: Float], (pitch: Float, yaw: Float, roll: Float)) -> Void)?
    var onFaceImage:   ((Data) -> Void)?
    var onNoFace:      (() -> Void)?

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("[FaceTracker]  ARFaceTracking not supported — requires TrueDepth camera")
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        print("[FaceTracker]  started")
    }

    func stop() {
        session.pause()
        print("[FaceTracker]  stopped")
    }

    // MARK: - ARSessionDelegate: face anchor (blend shapes + pose)

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let anchor = anchors.first(where: { $0 is ARFaceAnchor }) as? ARFaceAnchor else {
            onNoFace?()
            return
        }

        let now = Date()
        guard now.timeIntervalSince(lastBlendSendTime) >= blendSendInterval else { return }
        lastBlendSendTime = now

        let blendShapes = BlendShapeMapper.extract(from: anchor)
        let euler       = extractEulerAngles(from: anchor.transform)

        sendQueue.async { [weak self] in
            self?.onBlendShapes?(blendShapes, euler)
        }
    }

    // MARK: - ARSessionDelegate: camera frame (face image for InsightFace)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = Date()
        guard now.timeIntervalSince(lastImageSendTime) >= imageSendInterval else { return }

        // Only send if a face anchor exists in this frame
        let hasFace = frame.anchors.contains(where: { $0 is ARFaceAnchor })
        guard hasFace else { return }

        lastImageSendTime = now

        sendQueue.async { [weak self] in
            self?.captureAndSendFaceImage(from: frame)
        }
    }

    private func captureAndSendFaceImage(from frame: ARFrame) {
        // Front camera delivers YCbCr pixel buffer — convert to RGB via CIImage
        let pixelBuffer = frame.capturedImage
        let ciImage     = CIImage(cvPixelBuffer: pixelBuffer)

        // Scale down to 640-wide for network efficiency (InsightFace handles it fine)
        let scale      = 640.0 / ciImage.extent.width
        let scaled     = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        guard let cgImage = ciContext.createCGImage(scaled, from: scaled.extent) else { return }

        // Front camera on iPad in portrait is rotated 90° — correct to upright
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.6) else { return }

        onFaceImage?(jpegData)
    }

    // MARK: - Euler angles from 4×4 transform (degrees)

    private func extractEulerAngles(from transform: simd_float4x4) -> (pitch: Float, yaw: Float, roll: Float) {
        let m     = transform
        let pitch = asin(-m.columns.2.y)
        let yaw   = atan2(m.columns.2.x, m.columns.2.z)
        let roll  = atan2(m.columns.0.y, m.columns.1.y)
        let toDeg: Float = 180.0 / .pi
        return (pitch * toDeg, yaw * toDeg, roll * toDeg)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[FaceTracker]  session error: \(error)")
    }
}
