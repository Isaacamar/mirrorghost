import SwiftUI

struct SessionView: View {
    @EnvironmentObject var state: AppState
    @State private var showDebug = false
    @State private var showConnect = true

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Main face display
            FaceDisplayView(image: state.currentFaceImage)

            // Debug overlay (tap to toggle)
            if showDebug {
                VStack {
                    DebugOverlayView()
                    Spacer()
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            // Gesture controls
            Color.clear
                .contentShape(Rectangle())
                .gesture(
                    TapGesture(count: 2).onEnded {
                        showDebug.toggle()
                    }
                )
                .gesture(
                    LongPressGesture(minimumDuration: 1.0).onEnded { _ in
                        showConnect = true
                    }
                )
        }
        .sheet(isPresented: $showConnect) {
            ConnectView(onConnect: {
                showConnect = false
                state.connect()
            })
            .environmentObject(state)
        }
        .onDisappear {
            state.disconnect()
        }
    }
}

// ── Connection setup sheet ────────────────────────────────────────────────────

struct ConnectView: View {
    @EnvironmentObject var state: AppState
    var onConnect: () -> Void

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Mac Server")) {
                    TextField("IP Address  (e.g. 192.168.1.42)", text: $state.serverIP)
                        .keyboardType(.decimalPad)
                        .autocorrectionDisabled()
                    Text("Run  python main.py  in mac-server/ to see your IP.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Section {
                    Button("Connect") {
                        onConnect()
                    }
                    .disabled(state.serverIP.isEmpty)
                }
                Section(header: Text("Controls")) {
                    Text("Double-tap  — toggle debug overlay")
                    Text("Long-press  — return to this screen")
                    Text("Tap screen  — advance morph manually")
                }
                .font(.caption)
            }
            .navigationTitle("Mirror")
        }
        .onTapGesture {
            // advance morph on single tap (forwarded from SessionView)
        }
    }
}
