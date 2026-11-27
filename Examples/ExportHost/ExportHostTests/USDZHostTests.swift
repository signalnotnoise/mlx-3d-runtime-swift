import Metal
import ModelIO
import RealityKit
import XCTest
import simd

import MLX3DRuntime

/// USDZ checks that must run inside an application. `MDLAsset.canExportFileExtension("usdz")`
/// is false in the SwiftPM `xctest` host and in `MLX3DCLI`.
final class USDZHostTests: XCTestCase {
    func testUSDExporterIsRegisteredInTheApp() {
        XCTAssertTrue(
            MDLAsset.canExportFileExtension("usdz"),
            "Model I/O should register the USD exporter in this app host."
        )
    }

    func testNativeUSDZRoundTripsPositionsIndicesAndColors() throws {
        let url = try directory().appendingPathComponent("native.usdz")
        try MeshExport.writeUSDZ(figure(), to: url)
        let data = try Data(contentsOf: url)
        XCTAssertGreaterThan(data.count, 64)
        XCTAssertEqual(Array(data.prefix(2)), [0x50, 0x4b])
        let mesh = try modelIOMesh(at: url)
        expectClose(try positions(mesh), nativePoints)
        XCTAssertEqual(try indices(mesh), [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3])
        expectClose(try colors(mesh), [
            SIMD3(0.2, 0.2, 0.2),
            SIMD3(1, 0, 0),
            SIMD3(0, 1, 0),
            SIMD3(0, 0, 1),
        ])
    }

    func testYUpUSDZStandsUpAndPreservesHandedness() throws {
        let url = try directory().appendingPathComponent("up.usdz")
        try MeshExport.writeUSDZ(figure(), to: url, options: MeshExportOptions(upAxis: .yUp))
        let saved = try positions(modelIOMesh(at: url))
        let expected = nativePoints.map { SIMD3<Float>($0.x, $0.z, -$0.y) }
        expectClose(saved, expected)
        XCTAssertEqual(try indices(modelIOMesh(at: url)), [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3])

        let basis = triple(
            SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(0, 0, -1),
            SIMD3<Float>(0, 1, 0)
        )
        XCTAssertEqual(basis, 1, accuracy: 1e-5)

        let back = saved[3]
        let face = saved[2]
        let right = saved[1]
        for other in [saved[0], saved[1], saved[2]] {
            XCTAssertGreaterThan(back.y, other.y)
        }
        for other in [saved[0], saved[2], saved[3]] {
            XCTAssertGreaterThan(right.x, other.x)
        }
        for other in [saved[0], saved[1], saved[3]] {
            XCTAssertGreaterThan(face.z, other.z)
        }

        let faces: [Int32] = [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3]
        let faceCount = faces.count / 3
        for faceIndex in 0..<faceCount {
            let i0 = Int(faces[faceIndex * 3])
            let i1 = Int(faces[faceIndex * 3 + 1])
            let i2 = Int(faces[faceIndex * 3 + 2])
            let original = normal(nativePoints[i0], nativePoints[i1], nativePoints[i2])
            let exported = normal(saved[i0], saved[i1], saved[i2])
            let rotated = SIMD3<Float>(original.x, original.z, -original.y)
            XCTAssertEqual(exported.x, rotated.x, accuracy: 1e-4)
            XCTAssertEqual(exported.y, rotated.y, accuracy: 1e-4)
            XCTAssertEqual(exported.z, rotated.z, accuracy: 1e-4)
        }

        let before = triple(
            nativePoints[1] - nativePoints[0],
            nativePoints[2] - nativePoints[0],
            nativePoints[3] - nativePoints[0]
        )
        let after = triple(saved[1] - saved[0], saved[2] - saved[0], saved[3] - saved[0])
        XCTAssertLessThan(before, 0)
        XCTAssertLessThan(after, 0)
        XCTAssertEqual(before, after, accuracy: 1e-4)
    }

    func testYUpUSDZLoadsInRealityKitWhenAvailable() async throws {
        #if canImport(Metal)
        try XCTSkipUnless(MTLCreateSystemDefaultDevice() != nil, "No Metal device.")
        #else
        throw XCTSkip("Metal is not available.")
        #endif
        let url = try directory().appendingPathComponent("readable.usdz")
        try MeshExport.writeUSDZ(figure(), to: url, options: MeshExportOptions(upAxis: .yUp))
        let entity = try await loadEntity(url)
        XCTAssertTrue(entityContainsModel(entity))
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-usdz-host-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private var nativePoints: [SIMD3<Float>] {
        [
            SIMD3(0, 0, 0),
            SIMD3(1, 0, 0.25),
            SIMD3(0, -1, 0.5),
            SIMD3(0, 0, 2),
        ]
    }

    private func figure() -> Mesh {
        Mesh(
            positions: [
                0, 0, 0,
                1, 0, 0.25,
                0, -1, 0.5,
                0, 0, 2,
            ],
            faces: [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3],
            vertexColors: [
                0.2, 0.2, 0.2,
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ]
        )
    }

    private func expectClose(_ actual: [SIMD3<Float>], _ expected: [SIMD3<Float>], accuracy: Float = 1e-4) {
        XCTAssertEqual(actual.count, expected.count)
        for (got, want) in zip(actual, expected) {
            XCTAssertEqual(got.x, want.x, accuracy: accuracy)
            XCTAssertEqual(got.y, want.y, accuracy: accuracy)
            XCTAssertEqual(got.z, want.z, accuracy: accuracy)
        }
    }

    private func normal(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> SIMD3<Float> {
        cross(b - a, c - a)
    }

    private func cross(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        )
    }

    private func triple(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Float {
        a.x * (b.y * c.z - b.z * c.y) + a.y * (b.z * c.x - b.x * c.z) + a.z * (b.x * c.y - b.y * c.x)
    }

    private func modelIOMesh(at url: URL) throws -> MDLMesh {
        let asset = MDLAsset(url: url)
        let objects = asset.childObjects(of: MDLMesh.self)
        return try XCTUnwrap(objects.first as? MDLMesh)
    }

    private func positions(_ mesh: MDLMesh) throws -> [SIMD3<Float>] {
        try channel(mesh, name: MDLVertexAttributePosition)
    }

    private func colors(_ mesh: MDLMesh) throws -> [SIMD3<Float>] {
        try channel(mesh, name: MDLVertexAttributeColor)
    }

    private func channel(_ mesh: MDLMesh, name: String) throws -> [SIMD3<Float>] {
        let descriptor = mesh.vertexDescriptor
        var match: MDLVertexAttribute?
        for index in 0..<descriptor.attributes.count {
            guard let attribute = descriptor.attributes[index] as? MDLVertexAttribute else { continue }
            if attribute.name == name {
                match = attribute
                break
            }
        }
        let attribute = try XCTUnwrap(match)
        let layout = descriptor.layouts[attribute.bufferIndex] as? MDLVertexBufferLayout
        let stride = layout?.stride ?? MemoryLayout<Float>.size * 3
        let buffer = mesh.vertexBuffers[attribute.bufferIndex]
        let map = buffer.map()
        let raw = UnsafeRawPointer(map.bytes)
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(mesh.vertexCount)
        for vertex in 0..<mesh.vertexCount {
            let base = vertex * stride + attribute.offset
            values.append(
                SIMD3(
                    raw.load(fromByteOffset: base, as: Float.self),
                    raw.load(fromByteOffset: base + MemoryLayout<Float>.size, as: Float.self),
                    raw.load(fromByteOffset: base + MemoryLayout<Float>.size * 2, as: Float.self)
                )
            )
        }
        return values
    }

    private func indices(_ mesh: MDLMesh) throws -> [UInt32] {
        let submeshes = try XCTUnwrap(mesh.submeshes)
        let submesh = try XCTUnwrap(submeshes.firstObject as? MDLSubmesh)
        XCTAssertEqual(submesh.geometryType, .triangles)
        let map = submesh.indexBuffer.map()
        let raw = UnsafeRawPointer(map.bytes)
        var values: [UInt32] = []
        values.reserveCapacity(submesh.indexCount)
        switch submesh.indexType {
        case .uInt32:
            for index in 0..<submesh.indexCount {
                values.append(raw.load(fromByteOffset: index * 4, as: UInt32.self))
            }
        case .uInt16:
            for index in 0..<submesh.indexCount {
                values.append(UInt32(raw.load(fromByteOffset: index * 2, as: UInt16.self)))
            }
        case .uInt8:
            for index in 0..<submesh.indexCount {
                values.append(UInt32(raw.load(fromByteOffset: index, as: UInt8.self)))
            }
        case .invalid:
            XCTFail("USDZ submesh index type is invalid")
        @unknown default:
            XCTFail("USDZ submesh index type is unknown")
        }
        return values
    }

    private func loadEntity(_ url: URL) async throws -> Entity {
        #if os(macOS)
        if #available(macOS 15, *) {
            return try await Entity(contentsOf: url)
        }
        #elseif os(iOS)
        if #available(iOS 18, *) {
            return try await Entity(contentsOf: url)
        }
        #endif
        throw XCTSkip("RealityKit URL loading is not available on this OS.")
    }

    private func entityContainsModel(_ entity: Entity) -> Bool {
        if entity.components.has(ModelComponent.self) {
            return true
        }
        for child in entity.children where entityContainsModel(child) {
            return true
        }
        return false
    }
}
