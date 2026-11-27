import CoreGraphics
import Foundation
import MLX
import MLXNN
import os
import ShapERuntime
import TripoSRRuntime

/// Same subsystem and category Mesquette uses at the cutout call site
/// (`ContentViewViewModel.swift` and the image pickers, commit 971d381e).
private let imagePreparationLogger = Logger(
    subsystem: Bundle.main.bundleIdentifier ?? "com.visionarydevops.Mesquette",
    category: "image"
)

public struct TextToMeshOptions: Sendable {
    public var guidanceScale: Float
    public var numInferenceSteps: Int
    public var gridSize: Int
    public var queryBatchSize: Int
    public var seed: Int
    /// Upper bound checked before Shap-E loads. Cannot exceed `VolumeGrid.maximumResolution`.
    public var maximumGridSize: Int
    /// Metal marching cubes for the final extract. Partials stay on the CPU inside Shap-E.
    /// Default false, which is the extract the previous single-mesh call used.
    public var useMetalMarchingCubes: Bool
    /// How often Shap-E attaches a latent for a partial mesh. `0` reports steps and
    /// does not extract partials. Ignored when the progress handler is nil.
    public var progressInterval: Int

    public init(
        guidanceScale: Float = 15,
        numInferenceSteps: Int = 64,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0,
        maximumGridSize: Int = VolumeGrid.maximumResolution,
        useMetalMarchingCubes: Bool = false,
        progressInterval: Int = 0
    ) {
        self.guidanceScale = guidanceScale
        self.numInferenceSteps = numInferenceSteps
        self.gridSize = gridSize
        self.queryBatchSize = queryBatchSize
        self.seed = seed
        self.maximumGridSize = maximumGridSize
        self.useMetalMarchingCubes = useMetalMarchingCubes
        self.progressInterval = progressInterval
    }
}

/// What `imageToMesh(cgImage:)` does when `foregroundCutout` returns nil or throws.
///
/// The default is ``strict``. That keeps the throw this entry already had: a missing
/// subject is visible to the caller. Mesquette passes ``fallbackToOriginal``.
public enum CutoutFailurePolicy: Sendable, Equatable {
    /// Log and pass the original `CGImage` through. Matches Mesquette
    /// `ContentViewViewModel.swift` lines 198–211 at commit 971d381e: a nil cutout
    /// logs "Foreground cutout returned nil; TripoSR will use the unprepared image",
    /// and a thrown error logs "Foreground cutout failed, using unprepared image:"
    /// plus `error.localizedDescription`. Either way the original image is the one
    /// composited over gray.
    case fallbackToOriginal
    /// A nil cutout throws `imageUnsupported`. A thrown error propagates.
    case strict
}

/// Shap-E image-to-mesh options. Not used by any TripoSR entry.
///
/// Defaults follow Mesquette's image mode at commit 971d381e: guidance 3
/// (`AppModel.resetGenerationParameters`), 32 steps and grid 64
/// (`QualityPreset.standard`), seed 0, batch 1, Metal marching cubes on
/// (`QualityPreset.standard.usesGPUForFinalExtract`), live partials off
/// (`enableLivePartials` is false, so `progressInterval` is 0), and query batch
/// 4096 on macOS or 1024 elsewhere (`AppModel`). `sigmaMin` and `sigmaMax` are
/// the values `ShapERuntime.generate(image:)` already uses; Mesquette does not
/// override them.
public struct ShapEImageToMeshOptions: Sendable {
    public var batchSize: Int
    public var guidanceScale: Float
    public var numInferenceSteps: Int
    public var gridSize: Int
    public var queryBatchSize: Int
    public var seed: Int
    public var sigmaMin: Float
    public var sigmaMax: Float
    public var useMetalMarchingCubes: Bool
    public var progressInterval: Int
    /// Upper bound checked before the Shap-E bundle is opened.
    public var maximumGridSize: Int
    /// When true, `imageToMeshShapE` runs the white-background `removeBackground` prep.
    /// A nil or thrown removal keeps the original image (Mesquette's image pickers).
    /// This flag is never read by TripoSR.
    public var applyWhiteBackgroundRemoval: Bool

    public init(
        batchSize: Int = 1,
        guidanceScale: Float = 3,
        numInferenceSteps: Int = 32,
        gridSize: Int = 64,
        queryBatchSize: Int = ShapEImageToMeshOptions.mesquetteQueryBatchSize,
        seed: Int = 0,
        sigmaMin: Float = 0.002,
        sigmaMax: Float = 160,
        useMetalMarchingCubes: Bool = true,
        progressInterval: Int = 0,
        maximumGridSize: Int = VolumeGrid.maximumResolution,
        applyWhiteBackgroundRemoval: Bool = true
    ) {
        self.batchSize = batchSize
        self.guidanceScale = guidanceScale
        self.numInferenceSteps = numInferenceSteps
        self.gridSize = gridSize
        self.queryBatchSize = queryBatchSize
        self.seed = seed
        self.sigmaMin = sigmaMin
        self.sigmaMax = sigmaMax
        self.useMetalMarchingCubes = useMetalMarchingCubes
        self.progressInterval = progressInterval
        self.maximumGridSize = maximumGridSize
        self.applyWhiteBackgroundRemoval = applyWhiteBackgroundRemoval
    }

    /// `AppModel` query batch: 4096 on macOS, 1024 on iOS and visionOS.
    public static var mesquetteQueryBatchSize: Int {
        #if os(macOS)
        4096
        #else
        1024
        #endif
    }
}

public struct ImageToMeshOptions: Sendable {
    /// Density grid resolution. The default is 128. A caller passes its own value.
    /// Mesquette's previous TripoSR facade used 256.
    public var gridSize: Int
    public var threshold: Float
    public var queryBatchSize: Int
    /// World-space half-extent of the density grid. `nil` uses the model's renderer radius.
    public var radius: Float?
    public var imageSide: Int
    public var background: Float
    /// Upper bound checked before TripoSR weights load. Cannot exceed `VolumeGrid.maximumResolution`.
    public var maximumGridSize: Int
    /// Metal marching cubes for the final extract. Default false keeps the CPU extractor.
    public var useMetalMarchingCubes: Bool
    /// How a non-square image is drawn into the square tensor. Default ``ImageResizeMode/aspectFit``.
    /// The tensor `imageToMesh` overload does not resize, so it ignores this flag.
    public var resizeMode: ImageResizeMode
    /// When true, `imageToMesh(cgImage:)` runs `foregroundCutout` before TripoSR preprocessing.
    /// `imageToMesh(imageURL:)` and `imageToMesh(image:)` ignore this flag.
    public var applyForegroundCutout: Bool
    /// Foreground fill of the square cutout. Default matches upstream `resize_foreground`.
    public var foregroundRatio: CGFloat
    /// Used only by `imageToMesh(cgImage:)` when the cutout is enabled. Default ``CutoutFailurePolicy/strict``.
    public var cutoutFailurePolicy: CutoutFailurePolicy

    public init(
        gridSize: Int = 128,
        threshold: Float = 25,
        queryBatchSize: Int = 8192,
        radius: Float? = nil,
        imageSide: Int = 512,
        background: Float = 0.5,
        maximumGridSize: Int = VolumeGrid.maximumResolution,
        useMetalMarchingCubes: Bool = false,
        resizeMode: ImageResizeMode = .aspectFit,
        applyForegroundCutout: Bool = true,
        foregroundRatio: CGFloat = 0.85,
        cutoutFailurePolicy: CutoutFailurePolicy = .strict
    ) {
        self.gridSize = gridSize
        self.threshold = threshold
        self.queryBatchSize = queryBatchSize
        self.radius = radius
        self.imageSide = imageSide
        self.background = background
        self.maximumGridSize = maximumGridSize
        self.useMetalMarchingCubes = useMetalMarchingCubes
        self.resizeMode = resizeMode
        self.applyForegroundCutout = applyForegroundCutout
        self.foregroundRatio = foregroundRatio
        self.cutoutFailurePolicy = cutoutFailurePolicy
    }
}

public enum MLX3DRuntime {
    /// Text to one mesh through Shap-E.
    ///
    /// `bpeVocabularyURL` defaults to `bpe_simple_vocab_16e6.txt.gz` inside `modelBundleURL`.
    /// Vertex colors are RGB copied from Shap-E's RGBA channel. This entry always
    /// samples one mesh. Use ``textToMeshBatch(prompt:modelBundleURL:bpeVocabularyURL:batchSize:options:progress:)``
    /// for more than one. `progress` receives Shap-E's phase and step. Partial meshes
    /// are included only when `options.progressInterval` is greater than zero.
    public static func textToMesh(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: TextToMeshOptions = TextToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try await MLX3DSession().textToMesh(
            prompt: prompt,
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL,
            options: options,
            progress: progress
        )
    }

    /// Text to `batchSize` meshes through Shap-E. Each row uses `options.seed + row`.
    public static func textToMeshBatch(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        batchSize: Int,
        options: TextToMeshOptions = TextToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> [Mesh] {
        try await MLX3DSession().textToMeshBatch(
            prompt: prompt,
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: bpeVocabularyURL,
            batchSize: batchSize,
            options: options,
            progress: progress
        )
    }

    /// Image file to mesh through TripoSR.
    ///
    /// The file is decoded, composited over `options.background`, and drawn with
    /// `options.resizeMode` (aspect-fit by default) into `options.imageSide` (512).
    /// Density is sampled on a `gridSize`³ grid (128 by default) in batches of
    /// `queryBatchSize` (8192). Marching cubes uses `threshold` (25). Pass
    /// `useMetalMarchingCubes` to select the extractor. Vertex colors come from
    /// TripoSR's color query. `progress` receives `encoding`, `volume`, `meshing`,
    /// and `colors`. This entry does not run `foregroundCutout`. Cutout options
    /// on `options` are ignored.
    public static func imageToMesh(
        imageURL: URL,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        try requireWeights(weightsURL)
        let image = try ImageTensor.rgb(
            contentsOf: imageURL,
            side: options.imageSide,
            background: options.background,
            resizeMode: options.resizeMode
        )
        return try imageToMesh(image: image, weightsURL: weightsURL, options: options, progress: progress)
    }

    /// CGImage to mesh through TripoSR.
    ///
    /// When `options.applyForegroundCutout` is true (the default), this runs
    /// `ImageBackgroundRemoval.foregroundCutout` at `options.foregroundRatio`
    /// (0.85). The cutout stays transparent; ``ImageTensor/rgb(cgImage:side:background:resizeMode:)``
    /// then composites it over gray (`background` 0.5) using `options.resizeMode`.
    /// ``CutoutFailurePolicy/strict``
    /// (the default) throws when Vision returns nil or throws. ``CutoutFailurePolicy/fallbackToOriginal``
    /// logs and passes the original image, matching Mesquette. Set the cutout flag
    /// false for an image that is already prepared. This path never calls `removeBackground`.
    public static func imageToMesh(
        cgImage: CGImage,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let prepared: CGImage
        if options.applyForegroundCutout {
            let outcome: Result<CGImage?, Error>
            do {
                outcome = .success(
                    try await ImageBackgroundRemoval.foregroundCutout(
                        from: cgImage,
                        ratio: options.foregroundRatio
                    )
                )
            } catch {
                outcome = .failure(error)
            }
            prepared = try resolveForegroundCutout(
                source: cgImage,
                outcome: outcome,
                policy: options.cutoutFailurePolicy
            )
        } else {
            prepared = cgImage
        }
        let image = try ImageTensor.rgb(
            cgImage: prepared,
            side: options.imageSide,
            background: options.background,
            resizeMode: options.resizeMode
        )
        return try imageToMesh(image: image, weightsURL: weightsURL, options: options, progress: progress)
    }

    /// CGImage to one or more meshes through Shap-E's image pipeline.
    ///
    /// When `options.applyWhiteBackgroundRemoval` is true (the default), this runs
    /// `ImageBackgroundRemoval.removeBackground` (mask composited on solid white).
    /// A nil mask or a thrown error keeps the original image and, on a throw, logs
    /// Mesquette's picker message. The flag false skips that prep. The resulting
    /// image is passed to `ShapERuntime.generate(image:)`. `progress` receives
    /// Shap-E's phase and step. Partial meshes ride along only when
    /// `options.progressInterval` is greater than zero. This method never calls
    /// `foregroundCutout` and is not reachable from the TripoSR entries.
    public static func imageToMeshShapE(
        cgImage: CGImage,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: ShapEImageToMeshOptions = ShapEImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) async throws -> [Mesh] {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        guard options.batchSize > 0 else {
            throw MLX3DRuntimeError.invalidBatchSize(options.batchSize)
        }
        let prepared = await prepareShapEImage(
            cgImage,
            applyWhiteBackgroundRemoval: options.applyWhiteBackgroundRemoval
        )
        let vocabulary = try requireShapEBundle(modelBundleURL, bpeVocabularyURL: bpeVocabularyURL)
        let runtime = try ShapERuntime(
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: vocabulary
        )
        let generated = try await runtime.generate(
            image: prepared,
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
            progress: shapEProgress(progress)
        )
        return try meshes(from: generated)
    }

    /// Image tensor to mesh through TripoSR.
    ///
    /// `image` is float RGB in `0...1`, either `[height, width, 3]` or `[1, height, width, 3]`.
    /// This overload does not resize or cut out, so `options.resizeMode` is ignored.
    /// Use ``imageToMesh(imageURL:weightsURL:options:progress:)`` to decode a file.
    public static func imageToMesh(
        image: MLXArray,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions(),
        progress: MLX3DProgressHandler? = nil
    ) throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        try requireWeights(weightsURL)
        let model = try makeTripoSRModel(weightsURL: weightsURL)
        return try renderTripoSR(model: model, image: image, options: options, progress: progress)
    }

    static func makeTripoSRModel(weightsURL: URL) throws -> TripoSRModel {
        let model = TripoSRModel()
        MLX.eval(model.parameters())
        try model.loadWeights(from: weightsURL)
        return model
    }

    static func renderTripoSR(
        model: TripoSRModel,
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        let batched = try batchedImage(image)
        GenerationProgress.emit(progress, phase: "encoding", step: 0, totalSteps: 1)
        let triplanes = model.forwardTriplanes(image: batched)
        eval(triplanes)
        GenerationProgress.emit(progress, phase: "encoding", step: 1, totalSteps: 1)
        let mesh = try TripoSRMeshing.extract(
            model: model,
            triplanes: triplanes,
            options: options,
            progress: progress
        )
        guard mesh.faceCount > 0 else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return mesh
    }

    /// Applies ``CutoutFailurePolicy`` to one `foregroundCutout` result.
    /// Fallback logs the two Mesquette strings and returns `source`.
    static func resolveForegroundCutout(
        source: CGImage,
        outcome: Result<CGImage?, Error>,
        policy: CutoutFailurePolicy
    ) throws -> CGImage {
        switch (outcome, policy) {
        case (.success(let cutout?), _):
            return cutout
        case (.success(nil), .strict):
            throw MLX3DRuntimeError.imageUnsupported("Vision foreground cutout produced no subject.")
        case (.success(nil), .fallbackToOriginal):
            imagePreparationLogger.error(
                "Foreground cutout returned nil; TripoSR will use the unprepared image"
            )
            return source
        case (.failure(let error), .strict):
            throw error
        case (.failure(let error), .fallbackToOriginal):
            imagePreparationLogger.error(
                "Foreground cutout failed, using unprepared image: \(error.localizedDescription, privacy: .public)"
            )
            return source
        }
    }

    /// White-background prep for Shap-E. `removeBackground` nil uses the copied helper.
    /// A nil result keeps `cgImage` with no log. A throw logs Mesquette's picker message
    /// (`MacChatPanelView.pickImage`, commit 971d381e) and keeps `cgImage`.
    static func prepareShapEImage(
        _ cgImage: CGImage,
        applyWhiteBackgroundRemoval: Bool,
        removeBackground: ((CGImage) async throws -> CGImage?)? = nil
    ) async -> CGImage {
        guard applyWhiteBackgroundRemoval else {
            return cgImage
        }
        let remove = removeBackground ?? { image in
            try await ImageBackgroundRemoval.removeBackground(from: image)
        }
        do {
            if let stripped = try await remove(cgImage) {
                return stripped
            }
            return cgImage
        } catch {
            imagePreparationLogger.error(
                "Background removal failed: \(error.localizedDescription, privacy: .public)"
            )
            return cgImage
        }
    }

    private static func requireShapEBundle(_ modelBundleURL: URL, bpeVocabularyURL: URL?) throws -> URL {
        let manifest = modelBundleURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            throw MLX3DRuntimeError.bundleNotFound(modelBundleURL.path)
        }
        let vocabulary = bpeVocabularyURL
            ?? modelBundleURL.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz")
        guard FileManager.default.fileExists(atPath: vocabulary.path) else {
            throw MLX3DRuntimeError.bpeNotFound(vocabulary.path)
        }
        return vocabulary
    }

    static func meshes(from generated: [ShapEMeshResult]) throws -> [Mesh] {
        guard !generated.isEmpty else {
            throw MLX3DRuntimeError.emptyMesh
        }
        let meshes = try generated.map { try mesh(from: $0) }
        guard meshes.allSatisfy({ $0.faceCount > 0 }) else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return meshes
    }

    /// Nil stays nil, so Shap-E does not build a sampler callback or extract partials.
    static func shapEProgress(
        _ progress: MLX3DProgressHandler?
    ) -> ShapERuntime.ShapEProgressCallback? {
        guard let progress else { return nil }
        return { @Sendable phase, step, total, partials in
            let meshes: [Mesh]?
            if let partials {
                meshes = ShapEMeshArrays.kept(from: partials)
            } else {
                meshes = nil
            }
            progress(
                MLX3DGenerationEvent(
                    phase: phase,
                    step: step,
                    totalSteps: total,
                    partialMeshes: meshes
                )
            )
        }
    }

    static func mesh(from generated: ShapEMeshResult) throws -> Mesh {
        try ShapEMeshArrays.prepare(generated.meshResult.faces)
        var mesh = try ShapEMeshArrays.mesh(from: generated)
        mesh.vertexColors = try rgbColors(from: generated)
        return mesh
    }

    private static func rgbColors(from generated: ShapEMeshResult) throws -> [Float]? {
        guard let rgba = try ShapEMeshArrays.rgba(from: generated) else {
            return nil
        }
        let vertexCount = rgba.count / 4
        guard rgba.count == vertexCount * 4 else {
            throw MLX3DRuntimeError.colorCountMismatch(expected: vertexCount * 4, actual: rgba.count)
        }
        var rgb = [Float]()
        rgb.reserveCapacity(vertexCount * 3)
        for vertex in 0..<vertexCount {
            rgb.append(rgba[vertex * 4])
            rgb.append(rgba[vertex * 4 + 1])
            rgb.append(rgba[vertex * 4 + 2])
        }
        return rgb
    }

    static func requireWeights(_ weightsURL: URL) throws {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: weightsURL.path, isDirectory: &isDirectory)
        guard exists, !isDirectory.boolValue else {
            throw MLX3DRuntimeError.weightsNotFound(weightsURL.path)
        }
    }

    private static func batchedImage(_ image: MLXArray) throws -> MLXArray {
        switch image.ndim {
        case 3:
            guard image.shape[2] == 3 else {
                throw MLX3DRuntimeError.imageUnsupported(
                    "Expected [height, width, 3], got \(image.shape)."
                )
            }
            return image.reshaped([1, image.shape[0], image.shape[1], 3])
        case 4:
            guard image.shape[0] == 1, image.shape[3] == 3 else {
                throw MLX3DRuntimeError.imageUnsupported(
                    "Expected [1, height, width, 3], got \(image.shape)."
                )
            }
            return image
        default:
            throw MLX3DRuntimeError.imageUnsupported(
                "Expected 3 or 4 dimensions, got \(image.shape)."
            )
        }
    }
}
