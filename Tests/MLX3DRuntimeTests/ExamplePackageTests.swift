import Foundation
import Testing

struct ExamplePackageTests {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test func exampleImportsBothRuntimesThroughARelativePackage() throws {
        let root = repositoryRoot
        let manifest = try String(contentsOf: root.appendingPathComponent("Package.swift"), encoding: .utf8)
        let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
        let exampleManifest = try String(
            contentsOf: root.appendingPathComponent("Examples/Integration/Package.swift"),
            encoding: .utf8
        )
        let exampleSource = try String(
            contentsOf: root.appendingPathComponent("Examples/Integration/Sources/MLX3DExample/MLX3DExample.swift"),
            encoding: .utf8
        )

        #expect(manifest.contains("branch: \"main\""))
        #expect(!manifest.contains("cursor/shap-e-runtime-package-ea5b"))
        #expect(!manifest.contains("cursor/triposr-mlx-swift-package-45c1"))
        #expect(manifest.contains("from: \"0.1.0\""))
        #expect(manifest.contains("\"0.31.3\"..<\"0.31.5\""))
        #expect(readme.contains("from: \"0.1.0\""))
        #expect(readme.contains("https://huggingface.co/signalNotNoise/shap-e-mlx"))
        #expect(readme.contains("https://huggingface.co/signalNotNoise/triposr-mlx"))

        #expect(exampleManifest.contains("branch: \"main\""))
        #expect(!exampleManifest.contains("cursor/shap-e-runtime-package-ea5b"))
        #expect(!exampleManifest.contains("cursor/triposr-mlx-swift-package-45c1"))
        #expect(exampleManifest.contains("path: \"../..\""))
        #expect(exampleManifest.contains("ShapERuntime"))
        #expect(exampleManifest.contains("TripoSRRuntime"))
        #expect(exampleManifest.contains("MLX3DRuntime"))
        #expect(exampleSource.contains("import MLX3DRuntime"))
        #expect(exampleSource.contains("import ShapERuntime"))
        #expect(exampleSource.contains("import TripoSRRuntime"))
    }
}
