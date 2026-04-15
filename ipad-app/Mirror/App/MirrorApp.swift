import SwiftUI

@main
struct MirrorApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            SessionView()
                .environmentObject(appState)
                .preferredColorScheme(.dark)
        }
    }
}
