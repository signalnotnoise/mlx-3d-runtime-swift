import Foundation
import MLX
import MLXNN
import ShapERuntime
import TripoSRRuntime

public struct TextToMeshOptions: Sendable {
    public var guidanceScale: Float
    public var numInferenceSteps: Int
    public var gridSize: Int
    public var queryBatchSize: Int
    public var seed: Int
    /// Upper bound checked before Shap-E loads. Cannot exceed `VolumeGrid.maximumResolution`.
    public var maximumGridSize: Int

    public init(
        guidanceScale: Float = 15,
        numInferenceSteps: Int = 64,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0,
        maximumGridSize: Int = VolumeGrid.maximumResolution
    ) {
        self.guidanceScale = guidanceScale
        self.numInferenceSteps = numInferenceSteps
        self.gridSize = gridSize
        self.queryBatchSize = queryBatchSize
        self.seed = seed
        self.maximumGridSize = maximumGridSize
    }
}

public struct ImageToMeshOptions: Sendable {
    public var gridSize: Int
    public var threshold: Float
    public var queryBatchSize: Int
    /// World-space half-extent of the density grid. `nil` uses the model's renderer radius.
    public var radius: Float?
    public var imageSide: Int
    public var background: Float
    /// Upper bound checked before TripoSR weights load. Cannot exceed `VolumeGrid.maximumResolution`.
    public var maximumGridSize: Int

    public init(
        gridSize: Int = 128,
        threshold: Float = 25,
        queryBatchSize: Int = 8192,
        radius: Float? = nil,
        imageSide: Int = 512,
        background: Float = 0.5,
        maximumGridSize: Int = VolumeGrid.maximumResolution
    ) {
        self.gridSize = gridSize
        self.threshold = threshold
        self.queryBatchSize = queryBatchSize
        self.radius = radius
        self.imageSide = imageSide
        self.background = background
        self.maximumGridSize = maximumGridSize
    }
}

public enum MLX3DRuntime {
    /// Text to mesh through Shap-E.
    ///
    /// `bpeVocabularyURL` defaults to `bpe_simple_vocab_16e6.txt.gz` inside `modelBundleURL`.
    /// Vertex colors are RGB copied from Shap-E's RGBA channel.
    public static func textToMesh(
        prompt: String,
        modelBundleURL: URL,
        bpeVocabularyURL: URL? = nil,
        options: TextToMeshOptions = TextToMeshOptions()
    ) async throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        let manifest = modelBundleURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            throw MLX3DRuntimeError.bundleNotFound(modelBundleURL.path)
        }
        let vocabulary = bpeVocabularyURL
            ?? modelBundleURL.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz")
        guard FileManager.default.fileExists(atPath: vocabulary.path) else {
            throw MLX3DRuntimeError.bpeNotFound(vocabulary.path)
        }

        let runtime = try ShapERuntime(
            modelBundleURL: modelBundleURL,
            bpeVocabularyURL: vocabulary
        )
        let generated = try await runtime.generate(
            prompt: prompt,
            guidanceScale: options.guidanceScale,
            numInferenceSteps: options.numInferenceSteps,
            gridSize: options.gridSize,
            queryBatchSize: options.queryBatchSize,
            seed: options.seed
        )
        let mesh = try mesh(from: generated)
        guard mesh.faceCount > 0 else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return mesh
    }

    /// Image file to mesh through TripoSR.
    ///
    /// The file is decoded, composited over `options.background`, and aspect-fit to
    /// `options.imageSide` (512 by default). Density is sampled on a `gridSize`³ grid
    /// (128 by default) in batches of `queryBatchSize` (8192). Marching cubes uses
    /// `threshold` (25). Vertex colors come from TripoSR's color query.
    public static func imageToMesh(
        imageURL: URL,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions()
    ) throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        try requireWeights(weightsURL)
        let image = try ImageTensor.rgb(
            contentsOf: imageURL,
            side: options.imageSide,
            background: options.background
        )
        return try imageToMesh(image: image, weightsURL: weightsURL, options: options)
    }

    /// Image tensor to mesh through TripoSR.
    ///
    /// `image` is float RGB in `0...1`, either `[height, width, 3]` or `[1, height, width, 3]`.
    /// This overload does not resize. Use ``imageToMesh(imageURL:weightsURL:options:)`` to decode a file.
    public static func imageToMesh(
        image: MLXArray,
        weightsURL: URL,
        options: ImageToMeshOptions = ImageToMeshOptions()
    ) throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
        try requireWeights(weightsURL)
        let model = TripoSRModel()
        MLX.eval(model.parameters())
        try model.loadWeights(from: weightsURL)
        let batched = try batchedImage(image)
        let triplanes = model.forwardTriplanes(image: batched)
        eval(triplanes)
        let mesh = try TripoSRMeshing.extract(model: model, triplanes: triplanes, options: options)
        guard mesh.faceCount > 0 else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return mesh
    }

    static func mesh(from generated: ShapEMeshResult) throws -> Mesh {
        let positions = try generated.verticesArray()
        let faces = try generated.facesArray()
        let colors = try rgbColors(from: generated)
        return Mesh(positions: positions, faces: faces, vertexColors: colors)
    }

    private static func rgbColors(from generated: ShapEMeshResult) throws -> [Float]? {
        guard let rgba = try generated.colorsArray() else {
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

    private static func requireWeights(_ weightsURL: URL) throws {
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
