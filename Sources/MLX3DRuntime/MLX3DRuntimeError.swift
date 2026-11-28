import Foundation

public enum MLX3DRuntimeError: Error, Equatable, LocalizedError, Sendable {
    case invalidGrid(Int)
    case gridSizeOutOfRange(requested: Int, minimum: Int, maximum: Int)
    case fieldSizeMismatch(expected: Int, actual: Int)
    case invalidBatchSize(Int)
    case emptyBBox
    case colorCountMismatch(expected: Int, actual: Int)
    case unsupportedColorShape(shape: [Int])
    case densityCountMismatch(expected: Int, actual: Int)
    case imageUnreadable(String)
    case imageUnsupported(String)
    case weightsNotFound(String)
    case bundleNotFound(String)
    case bpeNotFound(String)
    case emptyMesh
    /// ``MLX3DSession/unload(_:)`` or ``MLX3DSession/unloadAll()`` ran while this
    /// load was still in flight. The finished model was not written back into
    /// the cache. Call ``MLX3DSession/load(_:)`` or generate again to retry.
    case modelUnloaded(backend: MLX3DBackend, bundlePath: String)

    public var errorDescription: String? {
        switch self {
        case .invalidGrid(let resolution):
            return "Grid resolution must be at least 2, got \(resolution)."
        case .gridSizeOutOfRange(let requested, let minimum, let maximum):
            return "Grid size \(requested) is outside \(minimum)...\(maximum). "
                + "The hard cap is VolumeGrid.maximumResolution (\(VolumeGrid.maximumResolution))."
        case .fieldSizeMismatch(let expected, let actual):
            return "Scalar field has \(actual) values, expected \(expected)."
        case .invalidBatchSize(let batchSize):
            return "Query batch size must be positive, got \(batchSize)."
        case .emptyBBox:
            return "Bounding box must have a positive extent on every axis."
        case .colorCountMismatch(let expected, let actual):
            return "Expected \(expected) color values, got \(actual)."
        case .unsupportedColorShape(let shape):
            return "Unsupported color layout \(shape). Accepted layouts: flat [4*V] or [..., 4]."
        case .densityCountMismatch(let expected, let actual):
            return "Density query returned \(actual) values, expected \(expected)."
        case .imageUnreadable(let path):
            return "Could not read image at \(path)."
        case .imageUnsupported(let reason):
            return "Unsupported image: \(reason)."
        case .weightsNotFound(let path):
            return "Weights not found at \(path)."
        case .bundleNotFound(let path):
            return "Shap-E bundle not found at \(path) (expected manifest.json)."
        case .bpeNotFound(let path):
            return "CLIP BPE vocabulary not found at \(path)."
        case .emptyMesh:
            return "Mesh extraction produced no triangles."
        case .modelUnloaded(let backend, let bundlePath):
            return "Unloaded the \(backend.label) model at \(bundlePath) before that load finished. "
                + "The model was not cached. Call load again to retry."
        }
    }
}
