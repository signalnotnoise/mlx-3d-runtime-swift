import Foundation
import ModelIO
import Testing
import simd

#if canImport(Metal)
import Metal
#endif
#if canImport(RealityKit)
import RealityKit
#endif

@testable import MLX3DRuntime

struct MeshExportTests {
    private func triangle(colors: Bool) -> Mesh {
        Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 2],
            vertexColors: colors ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : nil
        )
    }

    /// Asymmetric Z-up figure. Index 0 is the feet, 1 is +X, 2 faces −Y, 3 is the back at +Z.
    private func figure() -> Mesh {
        Mesh(
            positions: [
                0, 0, 0,
                1, 0, 0.25,
                0, -1, 0.5,
                0, 0, 2,
            ],
            faces: [
                0, 1, 3,
                0, 3, 2,
                0, 2, 1,
                1, 2, 3,
            ],
            vertexColors: [
                0.2, 0.2, 0.2,
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ]
        )
    }

    private var nativePoints: [SIMD3<Float>] {
        [
            SIMD3(0, 0, 0),
            SIMD3(1, 0, 0.25),
            SIMD3(0, -1, 0.5),
            SIMD3(0, 0, 2),
        ]
    }

    /// `(x, y, z) -> (x, z, -y)`. Back (+Z) becomes +Y. Facing (−Y) becomes +Z. +X stays +X.
    private func yUp(_ point: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(point.x, point.z, -point.y)
    }

    @Test func objWritesPositionsAndOneBasedFaces() throws {
        let text = try MeshExport.objText(triangle(colors: false))
        #expect(text == """
        # mlx-3d-runtime-swift vertices 3 faces 1
        v 0.000000 0.000000 0.000000
        v 1.000000 0.000000 0.000000
        v 0.000000 1.000000 0.000000
        f 1 2 3

        """)
    }

    @Test func objWritesVertexColors() throws {
        let text = try MeshExport.objText(triangle(colors: true))
        #expect(text == """
        # mlx-3d-runtime-swift vertices 3 faces 1
        v 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000
        v 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000
        v 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000
        f 1 2 3

        """)
    }

    @Test func plyDocumentIsUnchanged() throws {
        let colored = try MeshExport.plyText(triangle(colors: true))
        #expect(colored == """
        ply
        format ascii 1.0
        comment mlx-3d-runtime-swift
        element vertex 3
        property float x
        property float y
        property float z
        property float red
        property float green
        property float blue
        element face 1
        property list uchar int vertex_indices
        end_header
        0.000000 0.000000 0.000000 1.000000 0.000000 0.000000
        1.000000 0.000000 0.000000 0.000000 1.000000 0.000000
        0.000000 1.000000 0.000000 0.000000 0.000000 1.000000
        3 0 1 2

        """)
        let plain = try MeshExport.plyText(triangle(colors: false))
        #expect(plain.contains("format ascii 1.0\n"))
        #expect(plain.contains("element vertex 3\n"))
        #expect(plain.contains("element face 1\n"))
        #expect(!plain.contains("property float red"))
        #expect(plain.contains("0.000000 0.000000 0.000000\n"))
        #expect(plain.contains("3 0 1 2\n"))
    }

    @Test func objMaterialWritesReferenceAndAverageAlbedo() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let obj = directory.appendingPathComponent("figure.obj")
        try MeshExport.writeOBJ(figure(), to: obj)

        let mtl = MeshExport.materialURL(forOBJ: obj)
        let objText = try String(contentsOf: obj, encoding: .utf8)
        let mtlText = try String(contentsOf: mtl, encoding: .utf8)
        #expect(mtl.lastPathComponent == "figure.mtl")
        #expect(objText.contains("mtllib figure.mtl\n"))
        #expect(objText.contains("usemtl \(MeshExport.materialName)\n"))
        #expect(objText.contains("o figure\n"))
        #expect(mtlText.contains("newmtl \(MeshExport.materialName)\n"))
        #expect(mtlText.contains("Ka 0.100000 0.100000 0.100000\n"))
        #expect(mtlText.contains("Ks 0.050000 0.050000 0.050000\n"))
        #expect(mtlText.contains("Ns 20\n"))
        #expect(mtlText.contains("d 1.000000\n"))
        #expect(mtlText.contains("illum 2"))
        #expect(mtlText.contains("Kd \(formatted(averageAlbedo(of: figure())))\n"))
        for line in objText.split(separator: "\n") where line.hasPrefix("v ") {
            #expect(line.split(separator: " ").count == 4)
        }
        #expect(objFaces(objText) == [[1, 2, 4], [1, 4, 3], [1, 3, 2], [2, 3, 4]])
    }

    @Test func objMaterialWithoutColorsUsesGrayAlbedo() throws {
        let text = try MeshExport.materialText(for: triangle(colors: false))
        #expect(text.contains("newmtl VertexColorAverage\n"))
        #expect(text.contains("Kd 0.750000 0.750000 0.750000\n"))
        let wavefront = try MeshExport.wavefrontText(
            triangle(colors: false),
            mtlFileName: "tri.mtl",
            objectName: "tri"
        )
        #expect(wavefront.contains("mtllib tri.mtl\n"))
        #expect(wavefront.contains("usemtl VertexColorAverage\n"))
        #expect(!wavefront.contains("v 0.000000 0.000000 0.000000 1"))
    }

    @Test func vertexRGBFileStaysBackwardCompatible() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let obj = directory.appendingPathComponent("legacy.obj")
        let mesh = triangle(colors: true)
        let options = MeshExportOptions(objColors: .vertexRGB)
        try MeshExport.writeOBJ(mesh, to: obj, options: options)
        let written = try String(contentsOf: obj, encoding: .utf8)
        #expect(written == (try MeshExport.objText(mesh)))
        #expect(!FileManager.default.fileExists(atPath: MeshExport.materialURL(forOBJ: obj).path))
    }

    @Test func writeDispatchesOnExtension() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let mesh = triangle(colors: false)
        let obj = directory.appendingPathComponent("mesh.obj")
        let ply = directory.appendingPathComponent("mesh.ply")
        try MeshExport.write(mesh, to: obj)
        try MeshExport.write(mesh, to: ply)
        let objText = try String(contentsOf: obj, encoding: .utf8)
        let plyText = try String(contentsOf: ply, encoding: .utf8)
        let mtlText = try String(contentsOf: MeshExport.materialURL(forOBJ: obj), encoding: .utf8)
        #expect(objText.contains("mtllib mesh.mtl\n"))
        #expect(objText.contains("f 1 2 3\n"))
        #expect(mtlText.contains("Kd 0.750000 0.750000 0.750000\n"))
        #expect(plyText.contains("3 0 1 2\n"))
        #expect(plyText == (try MeshExport.plyText(mesh)))
    }

    @Test func nativeExportDoesNotConvertOrMutate() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        var mesh = figure()
        let original = mesh.positions
        let obj = directory.appendingPathComponent("native.obj")
        let ply = directory.appendingPathComponent("native.ply")
        try MeshExport.writeOBJ(mesh, to: obj)
        try MeshExport.writePLY(mesh, to: ply)
        #expect(mesh.positions == original)

        let objPoints = points(objVertices(try String(contentsOf: obj, encoding: .utf8)))
        let plyPoints = points(plyVertices(try String(contentsOf: ply, encoding: .utf8)))
        expectClose(objPoints, nativePoints)
        expectClose(plyPoints, nativePoints)
        #expect(objFaces(try String(contentsOf: obj, encoding: .utf8)) == [[1, 2, 4], [1, 4, 3], [1, 3, 2], [2, 3, 4]])
        #expect(plyFaces(try String(contentsOf: ply, encoding: .utf8)) == [[0, 1, 3], [0, 3, 2], [0, 2, 1], [1, 2, 3]])
        #expect(mesh.positions == original)
    }

    @Test func yUpSavedMeshStandsUpAndPreservesHandedness() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let mesh = figure()
        let options = MeshExportOptions(upAxis: .yUp)
        let expected = nativePoints.map(yUp)
        let basis = triple(
            yUp(SIMD3(1, 0, 0)),
            yUp(SIMD3(0, 1, 0)),
            yUp(SIMD3(0, 0, 1))
        )
        #expect(abs(basis - 1) < 1e-5)

        let obj = directory.appendingPathComponent("up.obj")
        let ply = directory.appendingPathComponent("up.ply")
        try MeshExport.writeOBJ(mesh, to: obj, options: options)
        try MeshExport.writePLY(mesh, to: ply, options: options)

        let objText = try String(contentsOf: obj, encoding: .utf8)
        let plyText = try String(contentsOf: ply, encoding: .utf8)
        let savedOBJ = points(objVertices(objText))
        let savedPLY = points(plyVertices(plyText))
        expectClose(savedOBJ, expected)
        expectClose(savedPLY, expected)
        #expect(objFaces(objText) == [[1, 2, 4], [1, 4, 3], [1, 3, 2], [2, 3, 4]])
        #expect(plyFaces(plyText) == [[0, 1, 3], [0, 3, 2], [0, 2, 1], [1, 2, 3]])

        for saved in [savedOBJ, savedPLY] {
            expectUprightFacing(saved)
            expectNormalsMatchRotation(saved)
            let before = triple(
                nativePoints[1] - nativePoints[0],
                nativePoints[2] - nativePoints[0],
                nativePoints[3] - nativePoints[0]
            )
            let after = triple(
                saved[1] - saved[0],
                saved[2] - saved[0],
                saved[3] - saved[0]
            )
            #expect(before < 0)
            #expect(after < 0)
            #expect(abs(before - after) < 1e-4)
        }

        let legacy = try MeshExport.objText(mesh, options: options)
        #expect(legacy.contains("v 0.000000 2.000000 0.000000 0.000000 0.000000 1.000000\n"))
        let uprightPLY = try MeshExport.plyText(mesh, options: options)
        #expect(uprightPLY.contains("property float x\nproperty float y\nproperty float z\n"))
        #expect(uprightPLY.contains("0.000000 2.000000 0.000000 0.000000 0.000000 1.000000\n"))
        let nativeMTL = try MeshExport.materialText(for: mesh)
        let yUpOBJ = try String(contentsOf: MeshExport.materialURL(forOBJ: obj), encoding: .utf8)
        #expect(yUpOBJ == nativeMTL)
    }

    @Test func usdzWritesA64ByteAlignedZip() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("packed.usdz")
        try MeshExport.write(triangle(colors: true), to: url)
        let data = try Data(contentsOf: url)
        #expect(data.prefix(2) == Data([0x50, 0x4b]))
        var offset = 0
        var payloads = 0
        while offset + 30 <= data.count {
            let signature = littleUInt32(data, offset)
            if signature != 0x04034b50 { break }
            let nameLength = Int(littleUInt16(data, offset + 26))
            let extraLength = Int(littleUInt16(data, offset + 28))
            let size = Int(littleUInt32(data, offset + 18))
            let start = offset + 30 + nameLength + extraLength
            #expect(start.isMultiple(of: 64))
            payloads += 1
            offset = start + size
        }
        #expect(payloads >= 1)
    }

    /// Model I/O does not export the `usdz` extension. These reads run in
    /// `Examples/ExportHost`, where the file is imported again.
    @Test(
        .enabled(
            if: MDLAsset.canExportFileExtension("usdz"),
            "Skipped in the SwiftPM test process: Model I/O does not register the USD exporter there. Examples/ExportHost verifies USDZ inside an app."
        )
    )
    func usdzKeepsPerVertexColorAndIsAZip() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("color.usdz")
        try MeshExport.writeUSDZ(figure(), to: url)
        let data = try Data(contentsOf: url)
        #expect(data.count > 64)
        #expect(data.prefix(2) == Data([0x50, 0x4b]))
        let mesh = try modelIOMesh(at: url)
        let colors = try modelIOColors(mesh)
        #expect(colors.count == 4)
        expectClose(colors, [
            SIMD3(0.2, 0.2, 0.2),
            SIMD3(1, 0, 0),
            SIMD3(0, 1, 0),
            SIMD3(0, 0, 1),
        ])
    }

    #if canImport(RealityKit)
    @Test(
        .enabled(
            if: MDLAsset.canExportFileExtension("usdz"),
            "Skipped in the SwiftPM test process: Model I/O does not register the USD exporter there. Examples/ExportHost verifies USDZ inside an app."
        )
    )
    @MainActor
    func usdzLoadsInRealityKitWhenAvailable() async throws {
        #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        #endif
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("readable.usdz")
        try MeshExport.writeUSDZ(figure(), to: url, options: MeshExportOptions(upAxis: .yUp))
        let entity = try await loadEntity(url)
        #expect(entityContainsModel(entity))
    }
    #endif

    @Test func exportRejectsEmptyMesh() {
        let mesh = Mesh(positions: [], faces: [])
        #expect(throws: MeshExportError.emptyMesh) {
            try MeshExport.objText(mesh)
        }
        #expect(throws: MeshExportError.emptyMesh) {
            try MeshExport.materialText(for: mesh)
        }
    }

    @Test func exportRejectsFacePastTheVertexList() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 5]
        )
        #expect(throws: MeshExportError.faceVertexMismatch) {
            try MeshExport.plyText(mesh)
        }
    }

    @Test func exportRejectsNegativeFaceIndex() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, -1]
        )
        #expect(throws: MeshExportError.faceVertexMismatch) {
            try MeshExport.wavefrontText(mesh, mtlFileName: "m.mtl", objectName: "m")
        }
    }

    @Test func exportRejectsNonTripleFaces() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1]
        )
        #expect(throws: MeshExportError.self) {
            try MeshExport.plyText(mesh)
        }
    }

    @Test func exportRejectsColorCountMismatch() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 2],
            vertexColors: [1, 0, 0]
        )
        #expect(throws: MeshExportError.self) {
            try MeshExport.objText(mesh)
        }
    }

    @Test func exportRejectsUnknownExtension() {
        let mesh = triangle(colors: false)
        #expect(throws: MeshExportError.self) {
            try MeshExport.write(mesh, to: URL(fileURLWithPath: "/tmp/mesh.stl"))
        }
    }

    private func littleUInt16(_ data: Data, _ offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    private func littleUInt32(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(littleUInt16(data, offset)) | (UInt32(littleUInt16(data, offset + 2)) << 16)
    }

    private func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-export-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func expectUprightFacing(
        _ saved: [SIMD3<Float>],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let back = saved[3]
        let face = saved[2]
        let right = saved[1]
        for other in [saved[0], saved[1], saved[2]] {
            #expect(back.y > other.y, sourceLocation: sourceLocation)
        }
        for other in [saved[0], saved[2], saved[3]] {
            #expect(right.x > other.x, sourceLocation: sourceLocation)
        }
        for other in [saved[0], saved[1], saved[3]] {
            #expect(face.z > other.z, sourceLocation: sourceLocation)
        }
    }

    private func expectNormalsMatchRotation(
        _ saved: [SIMD3<Float>],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let faces = figure().faces
        let faceCount = faces.count / 3
        for face in 0..<faceCount {
            let i0 = Int(faces[face * 3])
            let i1 = Int(faces[face * 3 + 1])
            let i2 = Int(faces[face * 3 + 2])
            let original = normal(nativePoints[i0], nativePoints[i1], nativePoints[i2])
            let exported = normal(saved[i0], saved[i1], saved[i2])
            let rotated = yUp(original)
            #expect(abs(exported.x - rotated.x) < 1e-4, sourceLocation: sourceLocation)
            #expect(abs(exported.y - rotated.y) < 1e-4, sourceLocation: sourceLocation)
            #expect(abs(exported.z - rotated.z) < 1e-4, sourceLocation: sourceLocation)
        }
    }

    private func expectClose(
        _ actual: [SIMD3<Float>],
        _ expected: [SIMD3<Float>],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.count, sourceLocation: sourceLocation)
        for (got, want) in zip(actual, expected) {
            #expect(abs(got.x - want.x) < 1e-5, sourceLocation: sourceLocation)
            #expect(abs(got.y - want.y) < 1e-5, sourceLocation: sourceLocation)
            #expect(abs(got.z - want.z) < 1e-5, sourceLocation: sourceLocation)
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
        dot(a, cross(b, c))
    }

    private func dot(_ lhs: SIMD3<Float>, _ rhs: SIMD3<Float>) -> Float {
        lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z
    }

    private func points(_ rows: [[Float]]) -> [SIMD3<Float>] {
        rows.map { SIMD3($0[0], $0[1], $0[2]) }
    }

    private func formatted(_ color: SIMD3<Float>) -> String {
        "\(format(color.x)) \(format(color.y)) \(format(color.z))"
    }

    private func format(_ value: Float) -> String {
        let finite: Float = value.isFinite ? value : 0
        let normalized: Float = finite == 0 ? 0 : finite
        return String(
            format: "%.6f",
            locale: Locale(identifier: "en_US_POSIX"),
            Double(normalized)
        )
    }

    private func averageAlbedo(of mesh: Mesh) -> SIMD3<Float> {
        let colors = mesh.vertexColors ?? []
        let count = mesh.vertexCount
        var red: Float = 0
        var green: Float = 0
        var blue: Float = 0
        for vertex in 0..<count {
            red += colors[vertex * 3]
            green += colors[vertex * 3 + 1]
            blue += colors[vertex * 3 + 2]
        }
        let scale = Float(count)
        return SIMD3(red / scale, green / scale, blue / scale)
    }

    private func objVertices(_ text: String) -> [[Float]] {
        text.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ")
            guard parts.first == "v" else { return nil }
            return parts.dropFirst().compactMap { Float(String($0)) }
        }
    }

    private func objFaces(_ text: String) -> [[Int]] {
        text.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ")
            guard parts.first == "f" else { return nil }
            return parts.dropFirst().compactMap { Int($0) }
        }
    }

    private func plyVertices(_ text: String) -> [[Float]] {
        let lines = text.split(separator: "\n").map(String.init)
        guard let headerEnd = lines.firstIndex(of: "end_header") else { return [] }
        var rows: [[Float]] = []
        for line in lines.dropFirst(headerEnd + 1) {
            let parts = line.split(separator: " ")
            if parts.first == "3" { break }
            let values = parts.compactMap { Float(String($0)) }
            guard values.count >= 3 else { continue }
            rows.append(Array(values.prefix(3)))
        }
        return rows
    }

    private func plyFaces(_ text: String) -> [[Int]] {
        text.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ")
            guard parts.first == "3" else { return nil }
            return parts.dropFirst().compactMap { Int($0) }
        }
    }

    private func modelIOMesh(at url: URL) throws -> MDLMesh {
        let asset = MDLAsset(url: url)
        let objects = asset.childObjects(of: MDLMesh.self)
        return try #require(objects.first as? MDLMesh)
    }

    private func modelIOPositions(_ mesh: MDLMesh) throws -> [SIMD3<Float>] {
        try channel(mesh, name: MDLVertexAttributePosition)
    }

    private func modelIOColors(_ mesh: MDLMesh) throws -> [SIMD3<Float>] {
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
        let attribute = try #require(match)
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

    private func modelIOIndices(_ mesh: MDLMesh) throws -> [UInt32] {
        let submeshes = try #require(mesh.submeshes)
        let submesh = try #require(submeshes.firstObject as? MDLSubmesh)
        #expect(submesh.geometryType == .triangles)
        let map = submesh.indexBuffer.map()
        let raw = UnsafeRawPointer(map.bytes)
        var indices: [UInt32] = []
        indices.reserveCapacity(submesh.indexCount)
        switch submesh.indexType {
        case .uInt32:
            for index in 0..<submesh.indexCount {
                indices.append(raw.load(fromByteOffset: index * 4, as: UInt32.self))
            }
        case .uInt16:
            for index in 0..<submesh.indexCount {
                indices.append(UInt32(raw.load(fromByteOffset: index * 2, as: UInt16.self)))
            }
        case .uInt8:
            for index in 0..<submesh.indexCount {
                indices.append(UInt32(raw.load(fromByteOffset: index, as: UInt8.self)))
            }
        case .invalid:
            Issue.record("USDZ submesh index type is invalid")
        @unknown default:
            Issue.record("USDZ submesh index type is unknown")
        }
        return indices
    }

    #if canImport(RealityKit)
    @MainActor
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
        throw MeshExportError.exportFailed("RealityKit URL loading is not available on this OS.")
    }

    @MainActor
    private func entityContainsModel(_ entity: Entity) -> Bool {
        if entity.components.has(ModelComponent.self) {
            return true
        }
        for child in entity.children where entityContainsModel(child) {
            return true
        }
        return false
    }
    #endif
}
