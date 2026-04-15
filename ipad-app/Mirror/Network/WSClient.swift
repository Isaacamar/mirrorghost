import Foundation
import UIKit

class WSClient: NSObject, ObservableObject {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?

    // Callbacks — called on background thread, dispatch to main if needed
    var onFaceFrame:    ((UIImage?, Double, Int, Int) -> Void)?
    var onConnected:    (() -> Void)?
    var onDisconnected: (() -> Void)?

    func connect(to host: String, port: Int = 8765) {
        guard let url = URL(string: "ws://\(host):\(port)/ws") else {
            print("[WSClient]  invalid URL for host: \(host)")
            return
        }
        print("[WSClient]  connecting to \(url)")
        urlSession    = URLSession(configuration: .default,
                                  delegate: self,
                                  delegateQueue: OperationQueue())
        webSocketTask = urlSession?.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveLoop()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
    }

    // MARK: - Sending

    func sendFaceFrame(blendShapes: [String: Float], euler: (pitch: Float, yaw: Float, roll: Float)) {
        let payload: [String: Any] = [
            "type":      "face_frame",
            "timestamp": Date().timeIntervalSince1970,
            "blend_shapes": blendShapes,
            "head_euler": [
                "pitch": euler.pitch,
                "yaw":   euler.yaw,
                "roll":  euler.roll,
            ],
        ]
        send(payload)
    }

    func sendAdvanceMorph() {
        send(["type": "advance_morph"])
    }

    func sendReset() {
        send(["type": "reset"])
    }

    private func send(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let str  = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(str)) { error in
            if let error = error { print("[WSClient]  send error: \(error)") }
        }
    }

    // MARK: - Receiving

    private func receiveLoop() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text): self?.handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        self?.handleMessage(text)
                    }
                @unknown default: break
                }
                self?.receiveLoop()   // reschedule
            case .failure(let error):
                print("[WSClient]  receive error: \(error)")
                self?.onDisconnected?()
            }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "server_ready":
            let sd  = json["sd_model"] as? String ?? ""
            let cn  = json["controlnet_model"] as? String ?? ""
            print("[WSClient]  server_ready — sd=\(sd)  controlnet=\(cn)")
            onConnected?()

        case "face_frame":
            guard let b64   = json["jpeg_b64"] as? String,
                  let imgData = Data(base64Encoded: b64),
                  let image  = UIImage(data: imgData) else { return }
            let morph  = json["morph_weight"]  as? Double ?? 0
            let index  = json["frame_index"]   as? Int    ?? 0
            let ms     = json["generation_ms"] as? Int    ?? 0
            onFaceFrame?(image, morph, index, ms)

        case "morph_update":
            // AppState handles this via onFaceFrame; or expose a separate callback here
            break

        default:
            break
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension WSClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        print("[WSClient]  connection opened")
        // server_ready message triggers onConnected — don't double-fire here
    }

    func urlSession(_ session: URLSession,
                    webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
                    reason: Data?) {
        print("[WSClient]  connection closed: \(closeCode)")
        onDisconnected?()
    }
}
