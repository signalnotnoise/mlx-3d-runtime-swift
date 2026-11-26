import Foundation
import MLX
import TripoSRRuntime

enum TripoSRMeshing {
    static func extract(
        model: TripoSRModel,
        triplanes: MLXArray,
        options: ImageToMeshOptions
    ) throws -> Mesh {
        guard options.gridSize >= 2 else {
            throw MLX3DRuntimeError.invalidGrid(options.gridSize)
        }
        guard options.queryBatchSize > 0 else {
            throw MLX3DRuntimeError.invalidBatchSize(options.queryBatchSize)
        }
        let radius = options.radius ?? model.renderer.radius
        guard radius > 0 else {
            throw MLX3DRuntimeError.emptyBBox
        }

        eval(triplanes)
        let bboxMin = SIMD3<Float>(-radius, -radius, -radius)
        let bboxMax = SIMD3<Float>(radius, radius, radius)
        let samples = VolumeGrid.positions(resolution: options.gridSize, min: bboxMin, max: bboxMax)
        let density = try queryDensity(
            model: model,
            triplanes: triplanes,
            positions: samples,
            batchSize: options.queryBatchSize
        )
        return try MarchingCubesExtractor.extract(
            field: density,
            resolution: options.gridSize,
            threshold: options.threshold,
            bboxMin: bboxMin,
            bboxMax: bboxMax,
            vertexColorsAt: { positions in
                try queryColors(
                    model: model,
                    triplanes: triplanes,
                    positions: positions,
                    batchSize: options.queryBatchSize
                )
            }
        )
    }

    private static func queryDensity(
        model: TripoSRModel,
        triplanes: MLXArray,
        positions: [Float],
        batchSize: Int
    ) throws -> [Float] {
        let pointCount = positions.count / 3
        var density = [Float]()
        density.reserveCapacity(pointCount)
        var offset = 0
        while offset < pointCount {
            let count = min(batchSize, pointCount - offset)
            let slice = Array(positions[(offset * 3)..<((offset + count) * 3)])
            let query = MLXArray(slice).reshaped([count, 3])
            let (densityArray, _) = model.evaluate(triplanes: triplanes, positions: query)
            eval(densityArray)
            let values = densityArray.asArray(Float.self)
            guard values.count == count else {
                throw MLX3DRuntimeError.densityCountMismatch(expected: count, actual: values.count)
            }
            density.append(contentsOf: values)
            offset += count
        }
        return density
    }

    private static func queryColors(
        model: TripoSRModel,
        triplanes: MLXArray,
        positions: [Float],
        batchSize: Int
    ) throws -> [Float] {
        let pointCount = positions.count / 3
        var colors = [Float]()
        colors.reserveCapacity(positions.count)
        var offset = 0
        while offset < pointCount {
            let count = min(batchSize, pointCount - offset)
            let slice = Array(positions[(offset * 3)..<((offset + count) * 3)])
            let query = MLXArray(slice).reshaped([count, 3])
            let (_, colorArray) = model.evaluate(triplanes: triplanes, positions: query)
            eval(colorArray)
            let values = colorArray.asArray(Float.self)
            guard values.count == count * 3 else {
                throw MLX3DRuntimeError.colorCountMismatch(expected: count * 3, actual: values.count)
            }
            colors.append(contentsOf: values)
            offset += count
        }
        return colors
    }
}
