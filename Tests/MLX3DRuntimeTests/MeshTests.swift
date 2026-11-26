import Foundation
import Testing

@testable import MLX3DRuntime

struct MeshTests {
    @Test func singleTriangleIsOpen() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 2]
        )
        #expect(mesh.vertexCount == 3)
        #expect(mesh.faceCount == 1)
        #expect(mesh.boundaryEdgeCount == 3)
        #expect(mesh.invalidFaceCount == 0)
        #expect(!mesh.isClosed)
    }

    @Test func tetrahedronIsClosed() {
        let mesh = Mesh(
            positions: [
                0, 0, 0,
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
            faces: [
                0, 2, 1,
                0, 1, 3,
                0, 3, 2,
                1, 2, 3,
            ]
        )
        #expect(mesh.boundaryEdgeCount == 0)
        #expect(mesh.invalidFaceCount == 0)
        #expect(mesh.isClosed)
    }

    @Test func degenerateFaceIsInvalid() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 0, 1]
        )
        #expect(mesh.invalidFaceCount == 1)
        #expect(!mesh.isClosed)
    }
}
