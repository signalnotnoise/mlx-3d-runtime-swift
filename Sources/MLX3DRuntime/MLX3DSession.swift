import Foundation
@preconcurrency import MLX
import ShapERuntime
@preconcurrency import TripoSRRuntime

/// Caller-owned cache of loaded Shap-E and TripoSR weights.
///
/// Create one session and hold it for as long as the loaded models should
/// stay resident. ``load(_:)``, ``textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:)``,
/// and ``imageToMesh(image:weightsURL:options:)`` reuse a model that is already
/// loaded for that ``MLX3DModelKey``. Concurrent callers of the same key share
/// one in-flight load.
///
/// This type does not download weights, choose an on-disk directory, or decide
/// when to preload. The caller drives ``load(_:)``, ``unload(_:)``,
/// ``unloadAll()``, ``evict(_:)``, and ``evictAll()``.
///
/// ``unload(_:)`` is authoritative. It drops the cached model and bumps a
/// generation token. A load that was already running for that key does not
/// put the model back when it finishes. Each caller waiting on that load
/// throws ``MLX3DRuntimeError/modelUnloaded(backend:bundlePath:)``. A later
/// ``load(_:)`` or generate call starts a new load. ``evict(_:)`` only drops a
/// model that is already ready. An in-flight load survives `evict` and can
/// still be cached.
///
/// The static ``MLX3DRuntime`` methods do not share this cache. Text-to-mesh
/// builds a throwaway session for that one call. Image-to-mesh on the enum
/// still loads weights on every call.
///
/// The session is a class, not an actor. Load coalescing lives in an internal
/// actor. Image tensors stay on the caller side of that actor because
/// `MLXArray` is not `Sendable`.
public final class MLX3DSession: Sendable {
    private let shapECache: MLX3DModelCache<ShapERuntime>
    private let tripoCache: MLX3DModelCache<CachedTripoSRModel>

    public init() {
        self.shapECache = MLX3DModelCache { key in
            try Self.loadShapE(key)
        }
        self.tripoCache = MLX3DModelCache { key in
            try Self.loadTripoSR(key)
        }
    }

    /// Loads and caches the model for `key`. A ready entry returns without
    /// calling the loader. A key that is already loading waits for that load.
    public func load(_ key: MLX3DModelKey) async throws {
        switch key.backend {
        case .shapE:
            _ = try await shapECache.model(for: key)
        case .tripoSR:
            _ = try await tripoCache.model(for: key)
        }
    }

    /// Drops one cached model. See the type discussion for the in-flight rule.
    public func unload(_ key: MLX3DModelKey) async {
        switch key.backend {
        case .shapE:
            await shapECache.unload(key)
        case .tripoSR:
            await tripoCache.unload(key)
        }
    }

    /// ``unload(_:)`` for every key this session has loaded or unloaded.
    public func unloadAll() async {
        await shapECache.unloadAll()
        await tripoCache.unloadAll()
    }

    /// Drops a ready model. Does not invalidate an in-flight load.
    public func evict(_ key: MLX3DModelKey) async {
        switch key.backend {
        case .shapE:
            await shapECache.evict(key)
        case .tripoSR:
            await tripoCache.evict(key)
        }
    }

    /// ``evict(_:)`` for every ready model.
    public func evictAll() async {
        await shapECache.evictAll()
        await tripoCache.evictAll()
    }

    /// Text to mesh through the cached Shap-E runtime for this bundle and BPE file.
    public func textToMesh(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: TextToMeshOptions = TextToMeshOptions()
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let key = MLX3DModelKey.shapE(
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL
        )
        let runtime = try await shapECache.model(for: key)
        let generated = try await runtime.generate(
            prompt: prompt,
            guidanceScale: options.guidanceScale,
            numInferenceSteps: options.numInferenceSteps,
            gridSize: options.gridSize,
            queryBatchSize: options.queryBatchSize,
            seed: options.seed
        )
        let mesh = try MLX3DRuntime.mesh(from: generated)
        guard mesh.faceCount > 0 else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return mesh
    }

    /// Image file to mesh through the cached TripoSR model.
    public func imageToMesh(
        imageURL: URL,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions()
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let cached = try await cachedTripoSR(weightsURL: weightsURL)
        let image = try ImageTensor.rgb(
            contentsOf: imageURL,
            side: options.imageSide,
            background: options.background
        )
        return try MLX3DRuntime.renderTripoSR(model: cached.model, image: image, options: options)
    }

    /// Image tensor to mesh through the cached TripoSR model.
    ///
    /// `image` is float RGB in `0...1`, either `[height, width, 3]` or
    /// `[1, height, width, 3]`. This overload does not resize.
    nonisolated(nonsending) public func imageToMesh(
        image: MLXArray,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions()
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let cached = try await cachedTripoSR(weightsURL: weightsURL)
        return try MLX3DRuntime.renderTripoSR(model: cached.model, image: image, options: options)
    }

    private func cachedTripoSR(weightsURL: URL) async throws -> CachedTripoSRModel {
        try await tripoCache.model(for: MLX3DModelKey.tripoSR(weightsURL: weightsURL))
    }

    private static func loadShapE(_ key: MLX3DModelKey) throws -> ShapERuntime {
        let manifest = key.bundleURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            throw MLX3DRuntimeError.bundleNotFound(key.bundleURL.path)
        }
        let vocabulary = key.auxiliaryURLs.first
            ?? key.bundleURL.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz")
        guard FileManager.default.fileExists(atPath: vocabulary.path) else {
            throw MLX3DRuntimeError.bpeNotFound(vocabulary.path)
        }
        return try ShapERuntime(modelBundleURL: key.bundleURL, bpeVocabularyURL: vocabulary)
    }

    private static func loadTripoSR(_ key: MLX3DModelKey) throws -> CachedTripoSRModel {
        try MLX3DRuntime.requireWeights(key.bundleURL)
        return CachedTripoSRModel(try MLX3DRuntime.makeTripoSRModel(weightsURL: key.bundleURL))
    }
}

/// `TripoSRModel` is not `Sendable`. The cache actor only stores this box.
/// Inference reads `model` on the session's caller side.
final class CachedTripoSRModel: @unchecked Sendable {
    let model: TripoSRModel

    init(_ model: TripoSRModel) {
        self.model = model
    }
}
