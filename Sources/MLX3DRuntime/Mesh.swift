import Foundation

/// Triangle mesh shared by the text-to-3D and image-to-3D paths.
///
/// `positions` and `vertexColors` are packed `x, y, z` / `r, g, b`.
/// `faces` are packed triangle indices into `positions`.
public struct Mesh: Equatable, Sendable {
    public var positions: [Float]
    public var faces: [Int32]
    public var vertexColors: [Float]?

    public init(positions: [Float], faces: [Int32], vertexColors: [Float]? = nil) {
        self.positions = positions
        self.faces = faces
        self.vertexColors = vertexColors
    }

    public var vertexCount: Int {
        positions.count / 3
    }

    public var faceCount: Int {
        faces.count / 3
    }

    /// Edges that belong to exactly one non-degenerate face.
    public var boundaryEdgeCount: Int {
        topology().boundaryEdgeCount
    }

    /// Faces with an index outside the vertex list, or with a repeated index.
    public var invalidFaceCount: Int {
        topology().invalidFaceCount
    }

    /// A mesh with at least one face and no open boundary.
    public var isClosed: Bool {
        let stats = topology()
        return stats.faceCount > 0 && stats.boundaryEdgeCount == 0 && stats.invalidFaceCount == 0
    }

    private func topology() -> (boundaryEdgeCount: Int, invalidFaceCount: Int, faceCount: Int) {
        guard positions.count.isMultiple(of: 3), faces.count.isMultiple(of: 3) else {
            return (0, max(faces.count / 3, 1), faces.count / 3)
        }
        let vertices = vertexCount
        let faceCount = self.faceCount
        var edgeUse: [EdgeKey: Int] = [:]
        edgeUse.reserveCapacity(faceCount * 2)
        var invalid = 0

        for face in 0..<faceCount {
            let i0 = faces[face * 3]
            let i1 = faces[face * 3 + 1]
            let i2 = faces[face * 3 + 2]
            let usable =
                i0 >= 0 && i1 >= 0 && i2 >= 0
                && Int(i0) < vertices && Int(i1) < vertices && Int(i2) < vertices
                && i0 != i1 && i1 != i2 && i0 != i2
            if !usable {
                invalid += 1
                continue
            }
            edgeUse[EdgeKey(i0, i1), default: 0] += 1
            edgeUse[EdgeKey(i1, i2), default: 0] += 1
            edgeUse[EdgeKey(i2, i0), default: 0] += 1
        }

        var boundary = 0
        for count in edgeUse.values where count == 1 {
            boundary += 1
        }
        return (boundary, invalid, faceCount)
    }
}

private struct EdgeKey: Hashable {
    let a: Int32
    let b: Int32

    init(_ i: Int32, _ j: Int32) {
        if i <= j {
            a = i
            b = j
        } else {
            a = j
            b = i
        }
    }
}
