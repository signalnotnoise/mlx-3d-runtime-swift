import Foundation

public enum MLX3DRuntimeError: Error, Equatable, LocalizedError, Sendable {
    case invalidGrid(Int)
    case fieldSizeMismatch(expected: Int, actual: Int)
    case invalidBatchSize(Int)
    case emptyBBox
    case colorCountMismatch(expected: Int, actual: Int)
    case densityCountMismatch(expected: Int, actual: Int)
    case imageUnreadable(String)
    case imageUnsupported(String)
    case weightsNotFound(String)
    case bundleNotFound(String)
    case bpeNotFound(String)
    case emptyMesh

    public var errorDescription: String? {
        switch self {
        case .invalidGrid(let resolution):
            return "Grid resolution must be at least 2, got \(resolution)."
        case .fieldSizeMismatch(let expected, let actual):
            return "Scalar field has \(actual) values, expected \(expected)."
        case .invalidBatchSize(let batchSize):
            return "Query batch size must be positive, got \(batchSize)."
        case .emptyBBox:
            return "Bounding box must have a positive extent on every axis."
        case .colorCountMismatch(let expected, let actual):
            return "Expected \(expected) color values, got \(actual)."
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
        }
    }
}
