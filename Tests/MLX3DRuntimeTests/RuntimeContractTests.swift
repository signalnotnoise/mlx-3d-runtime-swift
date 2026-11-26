import Foundation
import MLX
import ShapERuntime
import Testing
import TripoSRRuntime

@testable import MLX3DRuntime

struct RuntimeContractTests {
    @Test func imageToMeshReportsMissingWeights() {
        #expect(throws: MLX3DRuntimeError.self) {
            try MLX3DRuntime.imageToMesh(
                imageURL: URL(fileURLWithPath: "/nonexistent/mlx3d.png"),
                weightsURL: URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors")
            )
        }
    }

    @Test func textToMeshReportsMissingBundle() async {
        await #expect(throws: MLX3DRuntimeError.self) {
            try await MLX3DRuntime.textToMesh(
                prompt: "a chair",
                modelBundleURL: URL(fileURLWithPath: "/nonexistent/runtime_bundle")
            )
        }
    }

    @Test func marchingCubesRejectsAShortField() {
        #expect(throws: MLX3DRuntimeError.self) {
            try MarchingCubesExtractor.extract(
                field: [0, 1],
                resolution: 2,
                bboxMin: SIMD3<Float>(-1, -1, -1),
                bboxMax: SIMD3<Float>(1, 1, 1)
            )
        }
    }
}

struct PublishedWeightTests {
    @Test(
        .enabled(
            if: tripoSRWeightsURL() != nil,
            "triposr_swift_ready.safetensors is absent"
        )
    )
    func tripoSRWeightsLoadWhenPresent() throws {
        Device.setDefault(device: Device(.cpu))
        let url = try #require(tripoSRWeightsURL())
        let model = TripoSRModel()
        try model.loadWeights(from: url)
    }

    @Test(
        .enabled(
            if: shapEBundleURL() != nil,
            "Shap-E runtime bundle is absent"
        )
    )
    func shapERuntimeInitializesWhenBundleIsPresent() throws {
        let bundle = try #require(shapEBundleURL())
        let vocabulary = bundle.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz")
        _ = try ShapERuntime(modelBundleURL: bundle, bpeVocabularyURL: vocabulary)
    }
}

private func environmentValue(_ key: String) -> String? {
    let environment = ProcessInfo.processInfo.environment
    if let value = environment[key], !value.isEmpty {
        return value
    }
    if let value = environment["TEST_RUNNER_\(key)"], !value.isEmpty {
        return value
    }
    return nil
}

private func tripoSRWeightsURL() -> URL? {
    if let path = environmentValue("TRIPOSR_WEIGHTS_PATH"),
        FileManager.default.fileExists(atPath: path)
    {
        return URL(fileURLWithPath: path)
    }
    for name in ["triposr_swift_ready.safetensors", "triposr_weights/triposr_swift_ready.safetensors"] {
        let url = URL(fileURLWithPath: name).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
    }
    return nil
}

private func shapEBundleURL() -> URL? {
    var candidates: [URL] = []
    if let path = environmentValue("SHAP_E_BUNDLE"), !path.isEmpty {
        candidates.append(URL(fileURLWithPath: path))
    }
    candidates.append(URL(fileURLWithPath: "runtime_bundle"))
    for candidate in candidates {
        let manifest = candidate.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: manifest.path) {
            return candidate
        }
    }
    return nil
}
