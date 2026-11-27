import Foundation

/// Which runtime a cached model belongs to.
public enum MLX3DBackend: String, Sendable, Hashable, Codable {
    case shapE
    case tripoSR

    public var label: String {
        switch self {
        case .shapE:
            return "shap-e"
        case .tripoSR:
            return "triposr"
        }
    }
}

/// Identity of one loaded model inside an ``MLX3DSession``.
///
/// The key is the backend, the bundle URL, and every other file that loader
/// reads into the cached object. Prompt, seed, grid size, steps, and threshold
/// are not part of the key: they do not change the loaded weights.
///
/// Shap-E neural weights come from the bundle directory (`manifest.json` and
/// the artifacts next to it). The CLIP BPE vocabulary is an auxiliary URL
/// because ``ShapERuntime`` loads that file into the cached runtime. Two BPE
/// paths do not share a runtime. TripoSR has no auxiliary file. Its
/// `bundleURL` is the `.safetensors` file.
public struct MLX3DModelKey: Sendable, Hashable {
    public let backend: MLX3DBackend
    public let bundleURL: URL
    /// Files the loader reads besides ``bundleURL``.
    public let auxiliaryURLs: [URL]

    public init(backend: MLX3DBackend, bundleURL: URL, auxiliaryURLs: [URL] = []) {
        self.backend = backend
        self.bundleURL = Self.canonical(bundleURL)
        self.auxiliaryURLs = auxiliaryURLs.map(Self.canonical)
    }

    /// Shap-E bundle plus the BPE file the runtime loads.
    ///
    /// `bpeVocabularyURL` defaults to `bpe_simple_vocab_16e6.txt.gz` inside
    /// the bundle, matching ``MLX3DRuntime/textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:)``.
    public static func shapE(modelBundleURL: URL, bpeVocabularyURL: URL? = nil) -> MLX3DModelKey {
        let vocabulary = bpeVocabularyURL
            ?? modelBundleURL.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz")
        return MLX3DModelKey(backend: .shapE, bundleURL: modelBundleURL, auxiliaryURLs: [vocabulary])
    }

    /// TripoSR weights file. The safetensors URL is the bundle.
    public static func tripoSR(weightsURL: URL) -> MLX3DModelKey {
        MLX3DModelKey(backend: .tripoSR, bundleURL: weightsURL)
    }

    public static func == (lhs: MLX3DModelKey, rhs: MLX3DModelKey) -> Bool {
        lhs.backend == rhs.backend
            && lhs.bundleURL.path == rhs.bundleURL.path
            && lhs.auxiliaryURLs.map(\.path) == rhs.auxiliaryURLs.map(\.path)
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(backend)
        hasher.combine(bundleURL.path)
        hasher.combine(auxiliaryURLs.map(\.path))
    }

    /// `standardizedFileURL` with a trailing slash removed, so `..` and a
    /// trailing slash do not split one model into two cache entries.
    static func canonical(_ url: URL) -> URL {
        let standardized = url.standardizedFileURL
        let path = standardized.path
        guard path.count > 1, path.hasSuffix("/") else {
            return standardized
        }
        return URL(fileURLWithPath: String(path.dropLast()), isDirectory: true)
    }
}
