import Foundation
import MLX
import ShapERuntime

public enum MarchingCubesExtractor {
    /// Extract the isosurface `field == threshold` with Shap-E's public marching cubes.
    ///
    /// `field` is packed in the same order as ``VolumeGrid/positions(resolution:min:max:)``:
    /// X slowest, Z fastest, shape `resolution`³. Values above `threshold` are the
    /// interior (`field - threshold > 0`), which matches TripoSR's `density - threshold`
    /// field and Shap-E's positive-inside SDF. Vertices are returned in that bbox.
    /// Shap-E's extractor already emits X, Y, Z, so the result is not axis-swapped.
    public static func extract(
        field: [Float],
        resolution: Int,
        threshold: Float = 0,
        bboxMin: SIMD3<Float>,
        bboxMax: SIMD3<Float>,
        vertexColorsAt: (([Float]) throws -> [Float])? = nil
    ) throws -> Mesh {
        guard resolution >= 2 else {
            throw MLX3DRuntimeError.invalidGrid(resolution)
        }
        let expected = resolution * resolution * resolution
        guard field.count == expected else {
            throw MLX3DRuntimeError.fieldSizeMismatch(expected: expected, actual: field.count)
        }
        let extent = bboxMax - bboxMin
        guard extent.x > 0, extent.y > 0, extent.z > 0 else {
            throw MLX3DRuntimeError.emptyBBox
        }

        var sdfValues = field
        for index in sdfValues.indices {
            sdfValues[index] -= threshold
        }
        let sdf = MLXArray(sdfValues).reshaped([resolution, resolution, resolution])
        let bboxMinArray = MLXArray([bboxMin.x, bboxMin.y, bboxMin.z])
        let bboxSizeArray = MLXArray([extent.x, extent.y, extent.z])
        let (vertices, faces) = MarchingCubes.extract(
            sdf: sdf,
            bboxMin: bboxMinArray,
            bboxSize: bboxSizeArray
        )
        let faceIndices = faces.asType(.int32)
        eval(vertices, faceIndices)

        guard vertices.shape[0] > 0, faces.shape[0] > 0 else {
            return Mesh(positions: [], faces: [])
        }

        let positions = vertices.asArray(Float.self)
        let indices = faceIndices.asArray(Int32.self)
        guard positions.count.isMultiple(of: 3), indices.count.isMultiple(of: 3) else {
            throw MLX3DRuntimeError.fieldSizeMismatch(expected: positions.count, actual: indices.count)
        }

        let colors: [Float]?
        if let vertexColorsAt {
            let sampled = try vertexColorsAt(positions)
            guard sampled.count == positions.count else {
                throw MLX3DRuntimeError.colorCountMismatch(
                    expected: positions.count,
                    actual: sampled.count
                )
            }
            colors = sampled
        } else {
            colors = nil
        }

        return Mesh(positions: positions, faces: indices, vertexColors: colors)
    }
}
