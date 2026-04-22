import SwiftUI

@MainActor
final class TrackingState: ObservableObject {

    @Published var faceFrame:    FaceFrame? = nil
    @Published var faceDetected: Bool       = false
    @Published var fps:          Double     = 0.0

    let tracker = FaceTracker()

    private var lastFrameTime:    Date           = .distantPast
    private var recentIntervals: [TimeInterval]  = []

    init() {
        tracker.onFrame = { [weak self] frame in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.faceFrame    = frame
                self.faceDetected = true
                self.tickFPS()
            }
        }
        tracker.onNoFace = { [weak self] in
            Task { @MainActor [weak self] in self?.faceDetected = false }
        }
    }

    func start() { tracker.start() }
    func stop()  { tracker.stop()  }

    private func tickFPS() {
        let now = Date()
        let dt  = now.timeIntervalSince(lastFrameTime)
        lastFrameTime = now
        recentIntervals.append(dt)
        if recentIntervals.count > 30 { recentIntervals.removeFirst() }
        let avg = recentIntervals.reduce(0, +) / Double(recentIntervals.count)
        fps = avg > 0 ? 1.0 / avg : 0
    }
}
