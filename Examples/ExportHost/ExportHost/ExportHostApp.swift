import SwiftUI

/// App host for reading a USDZ back through Model I/O and RealityKit.
/// This target does not generate meshes or load weights.
@main
struct ExportHostApp: App {
    var body: some Scene {
        WindowGroup {
            Text("USDZ export host")
        }
    }
}
