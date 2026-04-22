import SwiftUI

struct RootView: View {
    @StateObject private var state = TrackingState()

    var body: some View {
        ZStack(alignment: .topLeading) {
            Color.black.ignoresSafeArea()

            if let frame = state.faceFrame {
                VStack(spacing: 4) {
                    // Wireframe | Mask side-by-side
                    HStack(spacing: 4) {
                        MeshImageView(image: frame.wireframe,    label: "wireframe")
                        MeshImageView(image: frame.isolationMask, label: "mask")
                    }
                    .frame(maxWidth: .infinity)
                    .aspectRatio(2, contentMode: .fit)

                    // Aligned crop (nil-safe)
                    if let crop = frame.alignedCrop {
                        Image(uiImage: crop)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(1, contentMode: .fit)
                            .frame(width: 180, height: 180)
                            .overlay(alignment: .bottomLeading) {
                                Text("crop")
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.5))
                                    .padding(3)
                            }
                    } else {
                        Rectangle()
                            .fill(Color.white.opacity(0.05))
                            .frame(width: 180, height: 180)
                            .overlay(Text("crop unavailable")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.white.opacity(0.3)))
                    }

                    Spacer()
                }
            } else {
                Text(state.faceDetected ? "rendering…" : "no face detected")
                    .foregroundColor(.white.opacity(0.3))
                    .font(.system(size: 13, design: .monospaced))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            // Debug panel — always visible in top-right
            DebugPanel(state: state)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .padding(12)
        }
        .onAppear   { state.start() }
        .onDisappear { state.stop() }
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
    }
}

// MARK: - Mesh image tile

private struct MeshImageView: View {
    let image: UIImage
    let label: String

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .interpolation(.none)
            .aspectRatio(1, contentMode: .fit)
            .overlay(alignment: .bottomLeading) {
                Text(label)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
                    .padding(3)
            }
    }
}

// MARK: - Debug panel

private struct DebugPanel: View {
    @ObservedObject var state: TrackingState

    var body: some View {
        VStack(alignment: .trailing, spacing: 2) {
            // Status row
            HStack(spacing: 6) {
                Circle()
                    .fill(state.faceDetected ? Color.green : Color.red)
                    .frame(width: 6, height: 6)
                mono(String(format: "%.1f fps", state.fps))
            }

            if let f = state.faceFrame {
                mono(String(format: "p %+.1f° y %+.1f° r %+.1f°",
                            f.euler.x, f.euler.y, f.euler.z))
                mono("depth \(f.depthAvailable ? "yes" : "no")")
                    .foregroundColor(f.depthAvailable ? .white.opacity(0.7) : .orange.opacity(0.7))
                mono("crop \(f.alignedCrop != nil ? "ok" : "—")")

                Divider().background(Color.white.opacity(0.15)).frame(width: 180)

                // Top 8 blend shapes as bar-style display
                ForEach(f.blendShapes.top(8), id: \.name) { item in
                    HStack(spacing: 4) {
                        mono(padded(item.name, to: 20))
                        GeometryReader { geo in
                            Rectangle()
                                .fill(Color.white.opacity(0.5))
                                .frame(width: geo.size.width * CGFloat(item.value))
                        }
                        .frame(width: 60, height: 8)
                        mono(String(format: "%.2f", item.value))
                    }
                }
            }
        }
        .padding(10)
        .background(Color.black.opacity(0.6))
        .cornerRadius(8)
    }

    private func mono(_ s: String) -> some View {
        Text(s)
            .font(.system(size: 10, design: .monospaced))
            .foregroundColor(.white.opacity(0.75))
            .lineLimit(1)
    }

    private func padded(_ s: String, to width: Int) -> String {
        s.count >= width ? String(s.prefix(width)) : s + String(repeating: " ", count: width - s.count)
    }
}
