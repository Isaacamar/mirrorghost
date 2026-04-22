import SwiftUI

struct DebugOverlayView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle()
                    .fill(state.connected ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(state.connected ? "connected" : "disconnected")
                Spacer()
                Text(String(format: "%.1f fps", state.fps))
            }
            Text(String(format: "morph  %.3f", state.morphWeight))
            Text("frame  \(state.frameIndex)   gen \(state.generationMs)ms")
            Text("face   \(state.faceDetected ? "detected" : "none")  (\(state.blendShapeCount) shapes)")
        }
        .font(.system(size: 11, design: .monospaced))
        .foregroundColor(.white.opacity(0.7))
        .padding(12)
        .background(Color.black.opacity(0.5))
        .cornerRadius(8)
        .padding()
    }
}
