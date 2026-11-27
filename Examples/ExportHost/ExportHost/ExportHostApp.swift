import SwiftUI

/// App host so Model I/O registers its USD exporter. The SwiftPM test
/// process does not. This target does not generate meshes or load weights.
@main
struct ExportHostApp: App {
    var body: some Scene {
        WindowGroup {
            Text("USDZ export host")
        }
    }
}
