import CoreGraphics
import Foundation
import MLX
import ShapERuntime
import Testing

@testable import MLX3DRuntime

struct GenerationProgressTests {
    init() {
        Device.setDefault(device: Device(.cpu))
    }

    @Test func generationDefaultsKeepThePreviousBehavior() {
        let text = TextToMeshOptions()
        #expect(text.useMetalMarchingCubes == false)
        #expect(text.progressInterval == 0)
        #expect(text.gridSize == 128)

        let tripo = ImageToMeshOptions()
        #expect(tripo.gridSize == 128)
        #expect(tripo.useMetalMarchingCubes == false)
        #expect(tripo.resizeMode == .aspectFit)
        let mesquetteGrid = ImageToMeshOptions(gridSize: 256)
        #expect(mesquetteGrid.gridSize == 256)

        let shapEImage = ShapEImageToMeshOptions()
        #expect(shapEImage.useMetalMarchingCubes == true)
        #expect(shapEImage.progressInterval == 0)
    }

    @Test func nilProgressDoesNotBuildAShapECallback() {
        #expect(MLX3DRuntime.shapEProgress(nil) == nil)
    }

    @Test func nilPartialMeshesAreForwardedWithoutAMesh() {
        let log = ProgressLog()
        let callback = MLX3DRuntime.shapEProgress { log.record($0) }
        callback?("sampling", 2, 8, nil)
        #expect(log.events.count == 1)
        #expect(log.events[0].phase == "sampling")
        #expect(log.events[0].step == 2)
        #expect(log.events[0].totalSteps == 8)
        #expect(log.events[0].partialMeshes == nil)
    }

    @Test func missingTextBundleThrowsAndDoesNotCallProgress() async {
        let log = ProgressLog()
        await #expect(throws: MLX3DRuntimeError.self) {
            _ = try await MLX3DRuntime.textToMesh(
                prompt: "a chair",
                modelBundleURL: URL(fileURLWithPath: "/nonexistent/runtime_bundle"),
                progress: { log.record($0) }
            )
        }
        #expect(log.events.isEmpty)
    }

    @Test func missingShapEImageBundleThrowsAndDoesNotCallProgress() async {
        let log = ProgressLog()
        await #expect(throws: MLX3DRuntimeError.bundleNotFound("/nonexistent/shap-e-bundle")) {
            _ = try await MLX3DRuntime.imageToMeshShapE(
                cgImage: GenerationProgressTests.solidImage(width: 4, height: 4),
                modelBundleURL: URL(fileURLWithPath: "/nonexistent/shap-e-bundle"),
                options: ShapEImageToMeshOptions(gridSize: 8, applyWhiteBackgroundRemoval: false),
                progress: { log.record($0) }
            )
        }
        #expect(log.events.isEmpty)
    }

    @Test func missingTripoWeightsThrowAndDoNotCallProgress() {
        let log = ProgressLog()
        #expect(throws: MLX3DRuntimeError.self) {
            _ = try MLX3DRuntime.imageToMesh(
                imageURL: URL(fileURLWithPath: "/nonexistent/mlx3d.png"),
                weightsURL: URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors"),
                progress: { log.record($0) }
            )
        }
        #expect(log.events.isEmpty)
    }

    @Test func invalidTextBatchThrowsBeforeLoadAndDoesNotCallProgress() async {
        let log = ProgressLog()
        let session = MLX3DSession(
            shapELoader: { _ in OrderedShapE() },
            tripoLoader: { _ in OrderedTripo() }
        )
        await #expect(throws: MLX3DRuntimeError.invalidBatchSize(0)) {
            _ = try await session.textToMeshBatch(
                prompt: "a chair",
                modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/progress-bundle"),
                batchSize: 0,
                progress: { log.record($0) }
            )
        }
        #expect(log.events.isEmpty)
        #expect(await session.loadsStarted(for: .shapE) == 0)
    }

    @Test func nilSessionProgressStaysNil() async throws {
        let seen = ProgressPresence()
        let session = MLX3DSession(
            shapELoader: { _ in PresenceShapE(seen: seen) },
            tripoLoader: { _ in OrderedTripo() }
        )
        let mesh = try await session.textToMesh(
            prompt: "a chair",
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/progress-bundle")
        )
        #expect(mesh == progressTriangle)
        #expect(seen.wasNil == true)
    }

    @Test func sessionForwardsShapEEventsInOrderIncludingOnePartial() async throws {
        let log = ProgressLog()
        let session = MLX3DSession(
            shapELoader: { _ in OrderedShapE() },
            tripoLoader: { _ in OrderedTripo() }
        )
        let meshes = try await session.textToMeshBatch(
            prompt: "a chair",
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/progress-bundle"),
            batchSize: 2,
            options: TextToMeshOptions(progressInterval: 5),
            progress: { log.record($0) }
        )
        #expect(meshes.count == 2)
        #expect(log.events.map(\.phase) == OrderedShapE.phases)
        let partialIndexes = log.events.indices.filter { log.events[$0].partialMeshes != nil }
        #expect(partialIndexes == [1])
        #expect(log.events[1].partialMeshes == [progressTriangle])
    }

    @Test func errorAfterAnEventDoesNotAppendAnotherEvent() async {
        let log = ProgressLog()
        let session = MLX3DSession(
            shapELoader: { _ in ThrowingShapE() },
            tripoLoader: { _ in OrderedTripo() }
        )
        await #expect(throws: MLX3DRuntimeError.emptyMesh) {
            _ = try await session.textToMesh(
                prompt: "a chair",
                modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/progress-bundle"),
                progress: { log.record($0) }
            )
        }
        #expect(log.events.map(\.phase) == ["embedding"])
    }

    @Test func sessionForwardsTripoSREventsInOrderWithoutPartials() async throws {
        let log = ProgressLog()
        let session = MLX3DSession(
            shapELoader: { _ in OrderedShapE() },
            tripoLoader: { _ in OrderedTripo() }
        )
        let mesh = try await session.imageToMesh(
            cgImage: GenerationProgressTests.solidImage(width: 4, height: 4),
            weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/progress-weights.safetensors"),
            options: ImageToMeshOptions(gridSize: 8, imageSide: 4, applyForegroundCutout: false),
            progress: { log.record($0) }
        )
        #expect(mesh == progressTriangle)
        #expect(log.events.map(\.phase) == OrderedTripo.phases)
        #expect(log.events.allSatisfy { $0.partialMeshes == nil })
        let steps = log.events.map(\.step)
        #expect(steps == [0, 1, 0, 1])
    }

    @Test func emptyAndDegeneratePartialsAreDroppedWithoutCrashing() throws {
        let empty = try shapEMesh(vertices: [], vertexShape: [0, 3], faces: [], faceShape: [0, 3])
        let wrongRank = try shapEMesh(
            vertices: [0, 0, 0, 1],
            vertexShape: [2, 2],
            faces: [0, 1, 2],
            faceShape: [1, 3]
        )
        let emptyFaces = try shapEMesh(
            vertices: trianglePositions,
            vertexShape: [3, 3],
            faces: [],
            faceShape: [0, 3]
        )
        let good = try coloredTriangleResult()
        let log = ProgressLog()
        let callback = MLX3DRuntime.shapEProgress { log.record($0) }

        callback?("sampling", 1, 4, [empty, wrongRank, emptyFaces, good])
        callback?("sampling", 2, 4, [empty])

        #expect(log.events.count == 2)
        #expect(log.events[0].phase == "sampling")
        #expect(log.events[0].partialMeshes == [coloredTriangle])
        #expect(log.events[1].partialMeshes == [])
        #expect(throws: MLX3DRuntimeError.emptyMesh) {
            _ = try MLX3DRuntime.meshes(from: [empty])
        }
        #expect(throws: MLX3DRuntimeError.emptyMesh) {
            _ = try MLX3DRuntime.mesh(from: emptyFaces)
        }
        let finished = try MLX3DRuntime.meshes(from: [good])
        #expect(finished == [coloredTriangle])
        let emptyVertices = try empty.mlxVertices()
        #expect(throws: MLX3DRuntimeError.emptyMesh) {
            _ = try ShapEMeshArrays.floats(emptyVertices, components: 3)
        }
    }

    @Test func badColorsAndOutOfRangeFacesAreDropped() throws {
        let badColors = try shapEMesh(
            vertices: trianglePositions,
            vertexShape: [3, 3],
            faces: [0, 1, 2],
            faceShape: [1, 3],
            rgba: [Float](repeating: 0, count: 9),
            rgbaShape: [3, 3]
        )
        let overflow = try shapEMesh(
            vertices: trianglePositions,
            vertexShape: [3, 3],
            faces64: [0, 1, Int64(Int32.max) + 1],
            faceShape: [1, 3]
        )
        let int64 = try shapEMesh(
            vertices: trianglePositions,
            vertexShape: [3, 3],
            faces64: [0, 1, 2],
            faceShape: [1, 3],
            rgba: triangleRGBA,
            rgbaShape: [3, 4]
        )
        let log = ProgressLog()
        let callback = MLX3DRuntime.shapEProgress { log.record($0) }
        callback?("sampling", 3, 8, [badColors, overflow, int64])

        #expect(log.events[0].partialMeshes == [coloredTriangle])
        #expect(throws: MLX3DRuntimeError.colorCountMismatch(expected: 8, actual: 9)) {
            _ = try MLX3DRuntime.mesh(from: badColors)
        }
        #expect(throws: MLX3DRuntimeError.emptyMesh) {
            _ = try MLX3DRuntime.mesh(from: overflow)
        }
    }

    @Test func transposedAndFloat16VerticesDoNotTrap() throws {
        let columns = MLXArray(
            [Float(0), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4]
        )
        let rows = columns.transposed()
        #expect(rows.shape == [4, 3])
        #expect(rows.strides != [3, 1])
        let values = try ShapEMeshArrays.floats(rows, components: 3)
        #expect(values == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

        let half = MLXArray([Float(0), 0, 0, 1, 0, 0, 0, 1, 0], [3, 3]).asType(.float16)
        let cast = try ShapEMeshArrays.floats(half, components: 3)
        #expect(cast == trianglePositions)

        let storedColors = MLXArray([Float](0..<16), [4, 4])
        let stridedColors = storedColors.transposed()
        #expect(stridedColors.strides != [4, 1])
        let colorValues = try ShapEMeshArrays.floats(stridedColors, components: 4)
        #expect(colorValues.count == 16)

        let wide = MLXArray([Float](0..<24), [4, 6])
        let rowSlice = wide[0..., 0..<3]
        #expect(rowSlice.shape == [4, 3])
        #expect(rowSlice.strides != [3, 1])
        let sliced = try ShapEMeshArrays.floats(rowSlice, components: 3)
        #expect(sliced == [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20])
    }

    static func solidImage(width: Int, height: Int) -> CGImage {
        let bytesPerRow = width * 4
        let pixels = [UInt8](repeating: 255, count: height * bytesPerRow)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }
}

private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [MLX3DGenerationEvent] = []

    var events: [MLX3DGenerationEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func record(_ event: MLX3DGenerationEvent) {
        lock.lock()
        storage.append(event)
        lock.unlock()
    }
}

private final class ProgressPresence: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Bool?

    var wasNil: Bool? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }

    func setWasNil(_ value: Bool) {
        lock.lock()
        stored = value
        lock.unlock()
    }
}

private struct OrderedShapE: SessionShapE {
    static let phases = ["embedding", "sampling", "projecting", "resolving"]

    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        for (index, phase) in Self.phases.enumerated() {
            let partials: [Mesh]? = phase == "sampling" ? [progressTriangle] : nil
            progress?(
                MLX3DGenerationEvent(
                    phase: phase,
                    step: index,
                    totalSteps: Self.phases.count,
                    partialMeshes: partials
                )
            )
        }
        return Array(repeating: progressTriangle, count: batchSize)
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        [progressTriangle]
    }
}

private struct PresenceShapE: SessionShapE {
    let seen: ProgressPresence

    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        seen.setWasNil(progress == nil)
        return Array(repeating: progressTriangle, count: batchSize)
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        [progressTriangle]
    }
}

private struct ThrowingShapE: SessionShapE {
    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        progress?(MLX3DGenerationEvent(phase: "embedding", step: 0, totalSteps: 1))
        throw MLX3DRuntimeError.emptyMesh
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        throw MLX3DRuntimeError.emptyMesh
    }
}

private struct OrderedTripo: SessionTripoSR {
    static let phases = ["encoding", "volume", "meshing", "colors"]

    func mesh(
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        let steps = [0, 1, 0, 1]
        for (phase, step) in zip(Self.phases, steps) {
            progress?(
                MLX3DGenerationEvent(
                    phase: phase,
                    step: step,
                    totalSteps: 1,
                    partialMeshes: nil
                )
            )
        }
        return progressTriangle
    }
}

private let trianglePositions: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
private let triangleRGBA: [Float] = [
    0.25, 0.5, 0.75, 1,
    0.25, 0.5, 0.75, 1,
    0.25, 0.5, 0.75, 1,
]
private let coloredTriangle = Mesh(
    positions: trianglePositions,
    faces: [0, 1, 2],
    vertexColors: [
        0.25, 0.5, 0.75,
        0.25, 0.5, 0.75,
        0.25, 0.5, 0.75,
    ]
)

private func coloredTriangleResult() throws -> ShapEMeshResult {
    try shapEMesh(
        vertices: trianglePositions,
        vertexShape: [3, 3],
        faces: [0, 1, 2],
        faceShape: [1, 3],
        rgba: triangleRGBA,
        rgbaShape: [3, 4]
    )
}

private func shapEMesh(
    vertices: [Float],
    vertexShape: [Int],
    faces: [Int32] = [],
    faces64: [Int64]? = nil,
    faceShape: [Int],
    rgba: [Float]? = nil,
    rgbaShape: [Int]? = nil
) throws -> ShapEMeshResult {
    let faceTensor: RuntimeTensor
    if let faces64 {
        faceTensor = try RuntimeTensor(
            name: "faces",
            shape: faceShape,
            dataType: .int64,
            data: faces64.withUnsafeBytes { Data($0) }
        )
    } else {
        faceTensor = try RuntimeTensor(
            name: "faces",
            shape: faceShape,
            dataType: .int32,
            data: faces.withUnsafeBytes { Data($0) }
        )
    }
    var channels: [String: RuntimeTensor]?
    if let rgba, let rgbaShape {
        channels = [
            "RGBA": try RuntimeTensor(
                name: "RGBA",
                shape: rgbaShape,
                dataType: .float32,
                data: rgba.withUnsafeBytes { Data($0) }
            )
        ]
    }
    return ShapEMeshResult(
        meshResult: MeshResult(
            vertices: try RuntimeTensor(
                name: "vertices",
                shape: vertexShape,
                dataType: .float32,
                data: vertices.withUnsafeBytes { Data($0) }
            ),
            faces: faceTensor,
            vertexChannels: channels,
            bbox: BoundingBox(minCorner: [0, 0, 0], maxCorner: [1, 1, 0]),
            timings: [:],
            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "test")
        )
    )
}

private let progressTriangle = Mesh(
    positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
    faces: [0, 1, 2]
)
