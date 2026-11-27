import Foundation
import MLX
import TripoSRRuntime

enum TripoSRMeshing {
    static func extract(
        model: TripoSRModel,
        triplanes: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        try VolumeGrid.validate(resolution: options.gridSize, maximum: options.maximumGridSize)
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
        let density = try queryDensity(
            model: model,
            triplanes: triplanes,
            resolution: options.gridSize,
            min: bboxMin,
            max: bboxMax,
            batchSize: options.queryBatchSize,
            progress: progress
        )
        GenerationProgress.emit(progress, phase: "meshing", step: 0, totalSteps: 1)
        var mesh = try MarchingCubesExtractor.extract(
            field: density,
            resolution: options.gridSize,
            threshold: options.threshold,
            bboxMin: bboxMin,
            bboxMax: bboxMax,
            useMetalMarchingCubes: options.useMetalMarchingCubes
        )
        GenerationProgress.emit(progress, phase: "meshing", step: 1, totalSteps: 1)
        if mesh.vertexCount > 0 {
            mesh.vertexColors = try queryColors(
                model: model,
                triplanes: triplanes,
                positions: mesh.positions,
                batchSize: options.queryBatchSize,
                progress: progress
            )
        }
        return mesh
    }

    private static func queryDensity(
        model: TripoSRModel,
        triplanes: MLXArray,
        resolution: Int,
        min bboxMin: SIMD3<Float>,
        max bboxMax: SIMD3<Float>,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) throws -> [Float] {
        let pointCount = resolution * resolution * resolution
        let chunkCount = max(1, (pointCount + batchSize - 1) / batchSize)
        var density = [Float]()
        density.reserveCapacity(pointCount)
        var offset = 0
        var chunkIndex = 0
        while offset < pointCount {
            let count = min(batchSize, pointCount - offset)
            let slice = VolumeGrid.positions(
                resolution: resolution,
                min: bboxMin,
                max: bboxMax,
                points: offset..<(offset + count)
            )
            let query = MLXArray(slice).reshaped([count, 3])
            let (densityArray, _) = model.evaluate(triplanes: triplanes, positions: query)
            eval(densityArray)
            let values = densityArray.asArray(Float.self)
            guard values.count == count else {
                throw MLX3DRuntimeError.densityCountMismatch(expected: count, actual: values.count)
            }
            density.append(contentsOf: values)
            offset += count
            chunkIndex += 1
            GenerationProgress.emit(
                progress,
                phase: "volume",
                step: chunkIndex,
                totalSteps: chunkCount
            )
        }
        return density
    }

    private static func queryColors(
        model: TripoSRModel,
        triplanes: MLXArray,
        positions: [Float],
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) throws -> [Float] {
        let pointCount = positions.count / 3
        let chunkCount = max(1, (pointCount + batchSize - 1) / batchSize)
        var colors = [Float]()
        colors.reserveCapacity(positions.count)
        var offset = 0
        var chunkIndex = 0
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
            chunkIndex += 1
            GenerationProgress.emit(
                progress,
                phase: "colors",
                step: chunkIndex,
                totalSteps: chunkCount
            )
        }
        return colors
    }
}
