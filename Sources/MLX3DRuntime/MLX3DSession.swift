import CoreGraphics
import Foundation
import ImageIO
@preconcurrency import MLX
import ShapERuntime
@preconcurrency import TripoSRRuntime
#if canImport(UIKit)
import UIKit
#endif

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
    private let shapECache: MLX3DModelCache<any SessionShapE>
    private let tripoCache: MLX3DModelCache<any SessionTripoSR>

    public convenience init() {
        self.init(
            shapELoader: { key in try Self.loadShapE(key) },
            tripoLoader: { key in try Self.loadTripoSR(key) }
        )
    }

    /// Test seam. Production `init` uses the real weight loaders.
    init(
        shapELoader: @escaping MLX3DModelCache<any SessionShapE>.Loader,
        tripoLoader: @escaping MLX3DModelCache<any SessionTripoSR>.Loader
    ) {
        self.shapECache = MLX3DModelCache(loader: shapELoader)
        self.tripoCache = MLX3DModelCache(loader: tripoLoader)
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

    /// How many times this session started a new load for `backend`.
    /// A reused slot does not increment the count.
    func loadsStarted(for backend: MLX3DBackend) async -> Int {
        switch backend {
        case .shapE:
            await shapECache.loadsStarted
        case .tripoSR:
            await tripoCache.loadsStarted
        }
    }

    /// Text to one mesh through the cached Shap-E runtime for this bundle and BPE file.
    ///
    /// Always samples one mesh. ``textToMeshBatch(prompt:modelBundleURL:bpeVocabularyURL:batchSize:options:progress:)``
    /// returns one mesh per row. `progress` is forwarded to Shap-E. Partial meshes
    /// are produced only when `options.progressInterval` is greater than zero.
    public func textToMesh(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: TextToMeshOptions = TextToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        let meshes = try await textToMeshBatch(
            prompt: prompt,
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL,
            batchSize: 1,
            options: options,
            progress: progress
        )
        guard let mesh = meshes.first else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return mesh
    }

    /// Text to `batchSize` meshes through the cached Shap-E runtime.
    /// Each row uses `options.seed + row`.
    public func textToMeshBatch(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        batchSize: Int,
        options: TextToMeshOptions = TextToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> [Mesh] {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        guard batchSize > 0 else {
            throw MLX3DRuntimeError.invalidBatchSize(batchSize)
        }
        let key = MLX3DModelKey.shapE(
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL
        )
        let runtime = try await shapECache.model(for: key)
        return try await runtime.textMesh(
            prompt: prompt,
            options: options,
            batchSize: batchSize,
            progress: progress
        )
    }

    /// Image file to mesh through the cached TripoSR model.
    public func imageToMesh(
        imageURL: URL,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let cached = try await cachedTripoSR(weightsURL: weightsURL)
        let image = try ImageTensor.rgb(
            contentsOf: imageURL,
            side: options.imageSide,
            background: options.background,
            resizeMode: options.resizeMode
        )
        return try cached.mesh(image: image, options: options, progress: progress)
    }

    /// CGImage to mesh through the cached TripoSR model.
    ///
    /// Cutout behavior matches ``MLX3DRuntime/imageToMesh(cgImage:weightsURL:options:)``:
    /// ``CutoutFailurePolicy/strict`` by default, ``ImageBackgroundRemoval/foregroundCutout(from:ratio:orientation:)``
    /// when the flag is on, then gray compositing. This method never calls `removeBackground`.
    public func imageToMesh(
        cgImage: CGImage,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try await tripoCGImage(
            cgImage,
            weightsURL: weightsURL,
            options: options,
            orientation: .up,
            foregroundCutout: nil,
            progress: progress
        )
    }

    #if canImport(UIKit)
    /// UIImage to mesh through the cached TripoSR model.
    ///
    /// The explicit `UIImage.Orientation` switch renders an upright bitmap first.
    /// Cutout, a nil cutout, and fallback all use that upright image. Vision is
    /// then asked for `.up`, because the pixels are already oriented.
    public func imageToMesh(
        uiImage: UIImage,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try await imageToMesh(
            uiImage: uiImage,
            weightsURL: weightsURL,
            options: options,
            foregroundCutout: nil,
            progress: progress
        )
    }

    func imageToMesh(
        uiImage: UIImage,
        weightsURL: URL,
        options: ImageToMeshOptions,
        foregroundCutout: (@Sendable (CGImage, CGFloat, CGImagePropertyOrientation) async throws -> CGImage?)?,
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        let upright = try ImageBackgroundRemoval.uprightCGImage(from: uiImage)
        return try await tripoCGImage(
            upright,
            weightsURL: weightsURL,
            options: options,
            orientation: .up,
            foregroundCutout: foregroundCutout,
            progress: progress
        )
    }

    /// CGImage that would be passed to TripoSR after orientation and cutout policy.
    /// Does not load weights.
    func modelInputCGImage(
        uiImage: UIImage,
        options: ImageToMeshOptions,
        foregroundCutout: (@Sendable (CGImage, CGFloat, CGImagePropertyOrientation) async throws -> CGImage?)?
    ) async throws -> CGImage {
        let upright = try ImageBackgroundRemoval.uprightCGImage(from: uiImage)
        return try await Self.preparedTripoImage(
            upright,
            options: options,
            orientation: .up,
            foregroundCutout: foregroundCutout
        )
    }
    #endif

    /// CGImage to meshes through the cached Shap-E runtime.
    ///
    /// White-background prep matches ``MLX3DRuntime/imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:)``.
    /// This method never calls `foregroundCutout`.
    public func imageToMeshShapE(
        cgImage: CGImage,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: ShapEImageToMeshOptions = ShapEImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> [Mesh] {
        try await shapEImage(
            cgImage,
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL,
            options: options,
            removeBackground: nil,
            progress: progress
        )
    }

    func shapEImage(
        _ cgImage: CGImage,
        modelBundleURL: URL,
        bpeVocabularyURL: URL?,
        options: ShapEImageToMeshOptions,
        removeBackground: (@Sendable (CGImage) async throws -> CGImage?)?,
        progress: MLX3DProgressHandler? = nil
    ) async throws -> [Mesh] {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        guard options.batchSize > 0 else {
            throw MLX3DRuntimeError.invalidBatchSize(options.batchSize)
        }
        let prepared = await MLX3DRuntime.prepareShapEImage(
            cgImage,
            applyWhiteBackgroundRemoval: options.applyWhiteBackgroundRemoval,
            removeBackground: removeBackground
        )
        let key = MLX3DModelKey.shapE(
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL
        )
        let runtime = try await shapECache.model(for: key)
        return try await runtime.imageMeshes(image: prepared, options: options, progress: progress)
    }

    func tripoCGImage(
        _ cgImage: CGImage,
        weightsURL: URL,
        options: ImageToMeshOptions,
        orientation: CGImagePropertyOrientation,
        foregroundCutout: (@Sendable (CGImage, CGFloat, CGImagePropertyOrientation) async throws -> CGImage?)?,
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let prepared = try await Self.preparedTripoImage(
            cgImage,
            options: options,
            orientation: orientation,
            foregroundCutout: foregroundCutout
        )
        let tensor = try ImageTensor.rgb(
            cgImage: prepared,
            side: options.imageSide,
            background: options.background,
            resizeMode: options.resizeMode
        )
        let cached = try await cachedTripoSR(weightsURL: weightsURL)
        return try cached.mesh(image: tensor, options: options, progress: progress)
    }

    private static func preparedTripoImage(
        _ cgImage: CGImage,
        options: ImageToMeshOptions,
        orientation: CGImagePropertyOrientation,
        foregroundCutout: (@Sendable (CGImage, CGFloat, CGImagePropertyOrientation) async throws -> CGImage?)?
    ) async throws -> CGImage {
        guard options.applyForegroundCutout else {
            return cgImage
        }
        let cut = foregroundCutout ?? { image, ratio, orientation in
            try await ImageBackgroundRemoval.foregroundCutout(
                from: image,
                ratio: ratio,
                orientation: orientation
            )
        }
        let outcome: Result<CGImage?, Error>
        do {
            outcome = .success(try await cut(cgImage, options.foregroundRatio, orientation))
        } catch {
            outcome = .failure(error)
        }
        return try MLX3DRuntime.resolveForegroundCutout(
            source: cgImage,
            outcome: outcome,
            policy: options.cutoutFailurePolicy
        )
    }

    /// Image tensor to mesh through the cached TripoSR model.
    ///
    /// `image` is float RGB in `0...1`, either `[height, width, 3]` or
    /// `[1, height, width, 3]`. This overload does not resize.
    nonisolated(nonsending) public func imageToMesh(
        image: MLXArray,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let cached = try await cachedTripoSR(weightsURL: weightsURL)
        return try cached.mesh(image: image, options: options, progress: progress)
    }

    private func cachedTripoSR(weightsURL: URL) async throws -> any SessionTripoSR {
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

protocol SessionShapE: Sendable {
    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh]
    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh]
}

protocol SessionTripoSR: Sendable {
    func mesh(
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh
}

extension ShapERuntime: SessionShapE {
    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        guard batchSize > 0 else {
            throw MLX3DRuntimeError.invalidBatchSize(batchSize)
        }
        let generated = try await generate(
            prompt: prompt,
            batchSize: batchSize,
            guidanceScale: options.guidanceScale,
            numInferenceSteps: options.numInferenceSteps,
            gridSize: options.gridSize,
            queryBatchSize: options.queryBatchSize,
            seed: options.seed,
            useMetalMarchingCubes: options.useMetalMarchingCubes,
            progressInterval: options.progressInterval,
            progress: MLX3DRuntime.shapEProgress(progress)
        )
        return try MLX3DRuntime.meshes(from: generated)
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        let generated = try await generate(
            image: image,
            batchSize: options.batchSize,
            guidanceScale: options.guidanceScale,
            numInferenceSteps: options.numInferenceSteps,
            sigmaMin: options.sigmaMin,
            sigmaMax: options.sigmaMax,
            gridSize: options.gridSize,
            queryBatchSize: options.queryBatchSize,
            seed: options.seed,
            useMetalMarchingCubes: options.useMetalMarchingCubes,
            progressInterval: options.progressInterval,
            progress: MLX3DRuntime.shapEProgress(progress)
        )
        return try MLX3DRuntime.meshes(from: generated)
    }
}

extension CachedTripoSRModel: SessionTripoSR {
    func mesh(
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        try MLX3DRuntime.renderTripoSR(model: model, image: image, options: options, progress: progress)
    }
}
