import Metal
import ModelIO
import RealityKit
import XCTest
import simd

import MLX3DRuntime

/// Reads a USDZ produced without `MDLAsset.canExportFileExtension("usdz")`,
/// which is false even in this app. Model I/O and RealityKit still import it.
final class USDZHostTests: XCTestCase {
    func testNativeUSDZRoundTripsPositionsIndicesAndColors() throws {
        let url = try directory().appendingPathComponent("native.usdz")
        try MeshExport.writeUSDZ(figure(), to: url)
        let usda = try usdaText(from: url)
        XCTAssertTrue(usda.contains("upAxis = \"Z\""))
        let saved = try points(usda, label: "point3f[] points = ")
        expectClose(saved, nativePoints)
        XCTAssertEqual(try integers(usda, label: "int[] faceVertexIndices = "), [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3])
        expectClose(try points(usda, label: "color3f[] primvars:displayColor = "), [
            SIMD3(0.2, 0.2, 0.2),
            SIMD3(1, 0, 0),
            SIMD3(0, 1, 0),
            SIMD3(0, 0, 1),
        ])
        let mesh = try modelIOMesh(at: url)
        XCTAssertEqual(mesh.vertexCount, nativePoints.count)
    }

    func testYUpUSDZStandsUpAndPreservesHandedness() throws {
        let url = try directory().appendingPathComponent("up.usdz")
        try MeshExport.writeUSDZ(figure(), to: url, options: MeshExportOptions(upAxis: .yUp))
        let usda = try usdaText(from: url)
        XCTAssertTrue(usda.contains("upAxis = \"Y\""))
        let saved = try points(usda, label: "point3f[] points = ")
        let expected = nativePoints.map { SIMD3<Float>($0.x, $0.z, -$0.y) }
        expectClose(saved, expected)
        XCTAssertEqual(try integers(usda, label: "int[] faceVertexIndices = "), [0, 1, 3, 0, 3, 2, 0, 2, 1, 1, 2, 3])
        let mesh = try modelIOMesh(at: url)
        XCTAssertEqual(mesh.vertexCount, expected.count)

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

    private func usdaText(from url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        XCTAssertEqual(Array(data.prefix(2)), [0x50, 0x4b])
        let nameLength = Int(UInt16(data[26]) | (UInt16(data[27]) << 8))
        let extraLength = Int(UInt16(data[28]) | (UInt16(data[29]) << 8))
        let size = Int(
            UInt32(data[18])
                | (UInt32(data[19]) << 8)
                | (UInt32(data[20]) << 16)
                | (UInt32(data[21]) << 24)
        )
        let start = 30 + nameLength + extraLength
        XCTAssertEqual(start % 64, 0)
        let payload = data.subdata(in: start..<(start + size))
        return try XCTUnwrap(String(data: payload, encoding: .utf8))
    }

    private func points(_ text: String, label: String) throws -> [SIMD3<Float>] {
        let body = try listBody(text, label: label)
        let pattern = #"\(([-0-9.eE]+),\s*([-0-9.eE]+),\s*([-0-9.eE]+)\)"#
        let expression = try NSRegularExpression(pattern: pattern)
        let range = NSRange(body.startIndex..., in: body)
        return expression.matches(in: body, range: range).map { match in
            let x = Float(body[Range(match.range(at: 1), in: body)!]) ?? .nan
            let y = Float(body[Range(match.range(at: 2), in: body)!]) ?? .nan
            let z = Float(body[Range(match.range(at: 3), in: body)!]) ?? .nan
            return SIMD3(x, y, z)
        }
    }

    private func integers(_ text: String, label: String) throws -> [Int] {
        try listBody(text, label: label).split(separator: ",").map {
            Int($0.trimmingCharacters(in: .whitespaces)) ?? -1
        }
    }

    private func listBody(_ text: String, label: String) throws -> String {
        guard let labelRange = text.range(of: label) else {
            XCTFail("USD layer is missing \(label)")
            return ""
        }
        let rest = text[labelRange.upperBound...]
        guard let open = rest.firstIndex(of: "["), let close = rest[open...].firstIndex(of: "]") else {
            XCTFail("USD layer is missing the list for \(label)")
            return ""
        }
        return String(rest[rest.index(after: open)..<close])
    }

    private func expectSamePositions(_ actual: [SIMD3<Float>], _ expected: [SIMD3<Float>]) {
        XCTAssertEqual(actual.count, expected.count)
        var remaining = expected
        for point in actual {
            guard let index = remaining.firstIndex(where: { distance($0, point) < 1e-3 }) else {
                XCTFail("Imported mesh is missing \(point)")
                return
            }
            remaining.remove(at: index)
        }
    }

    private func distance(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> Float {
        let delta = lhs - rhs
        return (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z).squareRoot()
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
