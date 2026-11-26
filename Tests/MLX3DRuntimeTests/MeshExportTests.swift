import Foundation
import Testing

@testable import MLX3DRuntime

struct MeshExportTests {
    private func triangle(colors: Bool) -> Mesh {
        Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 2],
            vertexColors: colors ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : nil
        )
    }

    @Test func objWritesPositionsAndOneBasedFaces() throws {
        let text = try MeshExport.objText(triangle(colors: false))
        #expect(text.contains("v 0.000000 0.000000 0.000000\n"))
        #expect(text.contains("v 1.000000 0.000000 0.000000\n"))
        #expect(text.contains("v 0.000000 1.000000 0.000000\n"))
        #expect(text.contains("f 1 2 3\n"))
        #expect(!text.contains("v 0.000000 0.000000 0.000000 1.000000"))
    }

    @Test func objWritesVertexColors() throws {
        let text = try MeshExport.objText(triangle(colors: true))
        #expect(text.contains("v 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000\n"))
        #expect(text.contains("v 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000\n"))
        #expect(text.contains("f 1 2 3\n"))
    }

    @Test func plyWritesHeaderVerticesAndZeroBasedFaces() throws {
        let text = try MeshExport.plyText(triangle(colors: true))
        #expect(text.contains("format ascii 1.0\n"))
        #expect(text.contains("element vertex 3\n"))
        #expect(text.contains("element face 1\n"))
        #expect(text.contains("property float red\n"))
        #expect(text.contains("0.000000 0.000000 0.000000 1.000000 0.000000 0.000000\n"))
        #expect(text.contains("3 0 1 2\n"))
    }

    @Test func plyOmitsColorsWhenAbsent() throws {
        let text = try MeshExport.plyText(triangle(colors: false))
        #expect(!text.contains("property float red"))
        #expect(text.contains("0.000000 0.000000 0.000000\n"))
    }

    @Test func writeDispatchesOnExtension() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-export-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let mesh = triangle(colors: false)
        let obj = directory.appendingPathComponent("mesh.obj")
        let ply = directory.appendingPathComponent("mesh.ply")
        try MeshExport.write(mesh, to: obj)
        try MeshExport.write(mesh, to: ply)
        let objText = try String(contentsOf: obj, encoding: .utf8)
        let plyText = try String(contentsOf: ply, encoding: .utf8)
        #expect(objText.contains("f 1 2 3\n"))
        #expect(plyText.contains("3 0 1 2\n"))
    }

    @Test func exportRejectsEmptyMesh() {
        let mesh = Mesh(positions: [], faces: [])
        #expect(throws: MeshExportError.emptyMesh) {
            try MeshExport.objText(mesh)
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
}
