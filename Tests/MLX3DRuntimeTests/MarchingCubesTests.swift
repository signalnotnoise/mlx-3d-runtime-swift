import Foundation
import MLX
import Testing

@testable import MLX3DRuntime

struct MarchingCubesTests {
    init() {
        Device.setDefault(device: Device(.cpu))
    }

    @Test func volumeGridIncludesBothCorners() throws {
        let minCorner = SIMD3<Float>(-0.87, -0.87, -0.87)
        let maxCorner = SIMD3<Float>(0.87, 0.87, 0.87)
        let values = try VolumeGrid.positions(resolution: 4, min: minCorner, max: maxCorner)
        #expect(values.count == 4 * 4 * 4 * 3)
        #expect(values[0] == minCorner.x)
        #expect(values[1] == minCorner.y)
        #expect(values[2] == minCorner.z)
        #expect(values[values.count - 3] == maxCorner.x)
        #expect(values[values.count - 2] == maxCorner.y)
        #expect(values[values.count - 1] == maxCorner.z)

        // ix = 1, iy = 0, iz = 0. X is slowest.
        let point = 1 * 16
        let expectedX = minCorner.x + (Float(1) / Float(3)) * (maxCorner.x - minCorner.x)
        #expect(abs(values[point * 3] - expectedX) < 1e-5)
        #expect(values[point * 3 + 1] == minCorner.y)
        #expect(values[point * 3 + 2] == minCorner.z)
    }

    @Test func constantFieldsProduceNoSurface() throws {
        let resolution = 8
        let count = resolution * resolution * resolution
        let bboxMin = SIMD3<Float>(-1, -1, -1)
        let bboxMax = SIMD3<Float>(1, 1, 1)
        for field in [[Float](repeating: -1, count: count), [Float](repeating: 1, count: count)] {
            let mesh = try MarchingCubesExtractor.extract(
                field: field,
                resolution: resolution,
                threshold: 0,
                bboxMin: bboxMin,
                bboxMax: bboxMax
            )
            #expect(mesh.vertexCount == 0)
            #expect(mesh.faceCount == 0)
            #expect(mesh.boundaryEdgeCount == 0)
        }
    }

    @Test func ellipsoidIsClosedWithNoBoundaryEdges() throws {
        let mesh = try ellipsoid(attachColors: true)
        #expect(mesh.vertexCount > 0)
        #expect(mesh.faceCount > 0)
        #expect(mesh.boundaryEdgeCount == 0)
        #expect(mesh.invalidFaceCount == 0)
        #expect(mesh.isClosed)
        let colors = try #require(mesh.vertexColors)
        #expect(colors == mesh.positions)

        let x = maxAbs(mesh, axis: 0)
        let y = maxAbs(mesh, axis: 1)
        let z = maxAbs(mesh, axis: 2)
        let detail = "x extent \(x) y \(y) z \(z) vertices \(mesh.vertexCount)"
        #expect(abs(x - 0.62) < 0.10, Comment(rawValue: detail))
        #expect(abs(y - 0.28) < 0.10, Comment(rawValue: detail))
        #expect(abs(z - 0.45) < 0.10, Comment(rawValue: detail))
    }

    private func ellipsoid(attachColors: Bool) throws -> Mesh {
        let resolution = 32
        let bboxMin = SIMD3<Float>(-1, -1, -1)
        let bboxMax = SIMD3<Float>(1, 1, 1)
        let samples = try VolumeGrid.positions(resolution: resolution, min: bboxMin, max: bboxMax)
        let rx: Float = 0.62
        let ry: Float = 0.28
        let rz: Float = 0.45
        var field = [Float](repeating: 0, count: samples.count / 3)
        for point in 0..<(samples.count / 3) {
            let x = samples[point * 3] / rx
            let y = samples[point * 3 + 1] / ry
            let z = samples[point * 3 + 2] / rz
            field[point] = 1 - (x * x + y * y + z * z).squareRoot()
        }
        return try MarchingCubesExtractor.extract(
            field: field,
            resolution: resolution,
            threshold: 0,
            bboxMin: bboxMin,
            bboxMax: bboxMax,
            vertexColorsAt: attachColors ? { positions in positions } : nil
        )
    }

    private func maxAbs(_ mesh: Mesh, axis: Int) -> Float {
        var value: Float = 0
        for vertex in 0..<mesh.vertexCount {
            value = max(value, abs(mesh.positions[vertex * 3 + axis]))
        }
        return value
    }
}

struct VolumeGridBatchTests {
    @Test func rejectsGridSizesOutsideTheInclusiveCap() {
        #expect(
            throws: MLX3DRuntimeError.gridSizeOutOfRange(
                requested: 1,
                minimum: VolumeGrid.minimumResolution,
                maximum: VolumeGrid.maximumResolution
            )
        ) {
            try VolumeGrid.validate(resolution: 1)
        }
        #expect(
            throws: MLX3DRuntimeError.gridSizeOutOfRange(
                requested: 513,
                minimum: VolumeGrid.minimumResolution,
                maximum: VolumeGrid.maximumResolution
            )
        ) {
            try VolumeGrid.validate(resolution: 513)
        }
        #expect(
            throws: MLX3DRuntimeError.gridSizeOutOfRange(
                requested: 32,
                minimum: VolumeGrid.minimumResolution,
                maximum: 16
            )
        ) {
            try VolumeGrid.validate(resolution: 32, maximum: 16)
        }
    }

    @Test func oversizedImageGridIsRejectedBeforeWeightsLoad() {
        #expect(
            throws: MLX3DRuntimeError.gridSizeOutOfRange(
                requested: 513,
                minimum: VolumeGrid.minimumResolution,
                maximum: VolumeGrid.maximumResolution
            )
        ) {
            try MLX3DRuntime.imageToMesh(
                imageURL: URL(fileURLWithPath: "/nonexistent/mlx3d.png"),
                weightsURL: URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors"),
                options: ImageToMeshOptions(gridSize: 513)
            )
        }
    }

    @Test func lazyBatchesMatchTheFullPointList() throws {
        let resolution = 5
        let minCorner = SIMD3<Float>(-0.5, 0, 0.25)
        let maxCorner = SIMD3<Float>(0.5, 1, 1.25)
        let full = try VolumeGrid.positions(resolution: resolution, min: minCorner, max: maxCorner)
        let batchSize = 3
        let pointCount = resolution * resolution * resolution
        var batched: [Float] = []
        var offset = 0
        while offset < pointCount {
            let count = min(batchSize, pointCount - offset)
            let batch = VolumeGrid.positions(
                resolution: resolution,
                min: minCorner,
                max: maxCorner,
                points: offset..<(offset + count)
            )
            #expect(batch.count == count * 3)
            #expect(batch.count < full.count)
            batched.append(contentsOf: batch)
            offset += count
        }
        #expect(batched == full)
    }
}
