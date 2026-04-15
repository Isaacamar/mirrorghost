import ARKit
import simd

class FaceTracker: NSObject, ObservableObject, ARSessionDelegate {
    private let session    = ARSession()
    private let sendQueue  = DispatchQueue(label: "facetracker.send", qos: .userInteractive)

    // Throttle to 10fps — ARKit delivers at 60fps
    private var lastSendTime: Date = .distantPast
    private let sendInterval: TimeInterval = 0.1

    var onBlendShapes: (([String: Float], (pitch: Float, yaw: Float, roll: Float)) -> Void)?
    var onNoFace: (() -> Void)?

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("[FaceTracker]  ARFaceTracking not supported — requires TrueDepth camera (iPad Pro / iPhone X+)")
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

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let anchor = anchors.first(where: { $0 is ARFaceAnchor }) as? ARFaceAnchor else {
            onNoFace?()
            return
        }

        let now = Date()
        guard now.timeIntervalSince(lastSendTime) >= sendInterval else { return }
        lastSendTime = now

        let blendShapes = BlendShapeMapper.extract(from: anchor)
        let euler       = extractEulerAngles(from: anchor.transform)

        sendQueue.async { [weak self] in
            self?.onBlendShapes?(blendShapes, euler)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[FaceTracker]  session error: \(error)")
    }

    // MARK: - Euler angles from 4×4 transform matrix

    private func extractEulerAngles(from transform: simd_float4x4) -> (pitch: Float, yaw: Float, roll: Float) {
        // Column-major: columns[i] is the i-th column
        let m = transform

        // Pitch (X rotation): arcsin(-m[2][1]) — m.columns[2].y
        let pitch = asin(-m.columns.2.y)

        // Yaw (Y rotation): atan2(m[2][0], m[2][2])
        let yaw = atan2(m.columns.2.x, m.columns.2.z)

        // Roll (Z rotation): atan2(m[0][1], m[1][1])
        let roll = atan2(m.columns.0.y, m.columns.1.y)

        let toDeg: Float = 180.0 / .pi
        return (pitch * toDeg, yaw * toDeg, roll * toDeg)
    }
}
