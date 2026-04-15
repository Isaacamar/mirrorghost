import SwiftUI
import Combine

class AppState: ObservableObject {
    @Published var connected: Bool = false
    @Published var serverIP: String = UserDefaults.standard.string(forKey: "serverIP") ?? ""

    @Published var currentFaceImage: UIImage? = nil
    @Published var morphWeight: Double = 0.0
    @Published var frameIndex: Int = 0
    @Published var generationMs: Int = 0

    @Published var faceDetected: Bool = false
    @Published var blendShapeCount: Int = 0

    @Published var fps: Double = 0.0
    private var lastFrameTime: Date = .distantPast
    private var frameTimes: [TimeInterval] = []

    let wsClient    = WSClient()
    let faceTracker = FaceTracker()

    init() {
        wsClient.onFaceFrame = { [weak self] image, morph, index, ms in
            DispatchQueue.main.async {
                self?.currentFaceImage = image
                self?.morphWeight      = morph
                self?.frameIndex       = index
                self?.generationMs     = ms
                self?.updateFPS()
            }
        }
        wsClient.onConnected = { [weak self] in
            DispatchQueue.main.async { self?.connected = true }
        }
        wsClient.onDisconnected = { [weak self] in
            DispatchQueue.main.async { self?.connected = false }
        }

        faceTracker.onBlendShapes = { [weak self] blendShapes, euler in
            // @Published mutations on main thread
            DispatchQueue.main.async {
                self?.blendShapeCount = blendShapes.count
                self?.faceDetected    = true
            }
            // Network sends are fine on background thread
            self?.wsClient.sendFaceFrame(blendShapes: blendShapes, euler: euler)
        }

        // Face image for InsightFace identity extraction on Mac (1fps)
        faceTracker.onFaceImage = { [weak self] jpegData in
            self?.wsClient.sendFaceImage(jpegData)
        }

        faceTracker.onNoFace = { [weak self] in
            DispatchQueue.main.async { self?.faceDetected = false }
        }
    }

    func connect() {
        UserDefaults.standard.set(serverIP, forKey: "serverIP")
        wsClient.connect(to: serverIP)
        faceTracker.start()
    }

    func disconnect() {
        wsClient.disconnect()
        faceTracker.stop()
    }

    func advanceMorph() { wsClient.sendAdvanceMorph() }
    func reset()        { wsClient.sendReset() }

    private func updateFPS() {
        let now      = Date()
        let interval = now.timeIntervalSince(lastFrameTime)
        lastFrameTime = now
        frameTimes.append(interval)
        if frameTimes.count > 10 { frameTimes.removeFirst() }
        let avg = frameTimes.reduce(0, +) / Double(frameTimes.count)
        fps = avg > 0 ? (1.0 / avg) : 0
    }
}
