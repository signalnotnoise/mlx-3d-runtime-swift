import Foundation

public enum MeshExportError: Error, Equatable, LocalizedError, Sendable {
    case emptyMesh
    case inconsistentCounts(String)
    case faceVertexMismatch
    case unsupportedFormat(String)

    public var errorDescription: String? {
        switch self {
        case .emptyMesh:
            return "Cannot export an empty mesh."
        case .inconsistentCounts(let reason):
            return reason
        case .faceVertexMismatch:
            return "Face index out of range for the vertex list."
        case .unsupportedFormat(let format):
            return "Unsupported mesh format '\(format)'. Use obj or ply."
        }
    }
}

public enum MeshExport {
    /// Write a Wavefront OBJ file. Faces are 1-indexed.
    /// Vertex colors are extra `v` components: `v x y z r g b`.
    public static func writeOBJ(_ mesh: Mesh, to url: URL, comment: String? = nil) throws {
        let text = try objText(mesh, comment: comment)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Write an ASCII PLY file. Colors are float `red` `green` `blue` in `0...1` when present.
    public static func writePLY(_ mesh: Mesh, to url: URL) throws {
        let text = try plyText(mesh)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Pick OBJ or PLY from the file extension.
    public static func write(_ mesh: Mesh, to url: URL) throws {
        switch url.pathExtension.lowercased() {
        case "obj":
            try writeOBJ(mesh, to: url)
        case "ply":
            try writePLY(mesh, to: url)
        default:
            throw MeshExportError.unsupportedFormat(url.pathExtension)
        }
    }

    public static func objText(_ mesh: Mesh, comment: String? = nil) throws -> String {
        let checked = try validated(mesh)
        var text = ""
        text.reserveCapacity(checked.vertexCount * 48 + checked.faceCount * 24)
        if let comment {
            for line in comment.split(separator: "\n", omittingEmptySubsequences: false) {
                text += "# \(line)\n"
            }
        }
        text += "# mlx-3d-runtime-swift vertices \(checked.vertexCount) faces \(checked.faceCount)\n"
        for vertex in 0..<checked.vertexCount {
            let x = formatComponent(checked.positions[vertex * 3])
            let y = formatComponent(checked.positions[vertex * 3 + 1])
            let z = formatComponent(checked.positions[vertex * 3 + 2])
            if let colors = checked.colors {
                let r = formatComponent(colors[vertex * 3])
                let g = formatComponent(colors[vertex * 3 + 1])
                let b = formatComponent(colors[vertex * 3 + 2])
                text += "v \(x) \(y) \(z) \(r) \(g) \(b)\n"
            } else {
                text += "v \(x) \(y) \(z)\n"
            }
        }
        for face in 0..<checked.faceCount {
            let i0 = Int(checked.faces[face * 3]) + 1
            let i1 = Int(checked.faces[face * 3 + 1]) + 1
            let i2 = Int(checked.faces[face * 3 + 2]) + 1
            text += "f \(i0) \(i1) \(i2)\n"
        }
        return text
    }

    public static func plyText(_ mesh: Mesh) throws -> String {
        let checked = try validated(mesh)
        var text = ""
        text.reserveCapacity(checked.vertexCount * 48 + checked.faceCount * 24 + 256)
        text += "ply\n"
        text += "format ascii 1.0\n"
        text += "comment mlx-3d-runtime-swift\n"
        text += "element vertex \(checked.vertexCount)\n"
        text += "property float x\n"
        text += "property float y\n"
        text += "property float z\n"
        if checked.colors != nil {
            text += "property float red\n"
            text += "property float green\n"
            text += "property float blue\n"
        }
        text += "element face \(checked.faceCount)\n"
        text += "property list uchar int vertex_indices\n"
        text += "end_header\n"
        for vertex in 0..<checked.vertexCount {
            let x = formatComponent(checked.positions[vertex * 3])
            let y = formatComponent(checked.positions[vertex * 3 + 1])
            let z = formatComponent(checked.positions[vertex * 3 + 2])
            if let colors = checked.colors {
                let r = formatComponent(colors[vertex * 3])
                let g = formatComponent(colors[vertex * 3 + 1])
                let b = formatComponent(colors[vertex * 3 + 2])
                text += "\(x) \(y) \(z) \(r) \(g) \(b)\n"
            } else {
                text += "\(x) \(y) \(z)\n"
            }
        }
        for face in 0..<checked.faceCount {
            let i0 = checked.faces[face * 3]
            let i1 = checked.faces[face * 3 + 1]
            let i2 = checked.faces[face * 3 + 2]
            text += "3 \(i0) \(i1) \(i2)\n"
        }
        return text
    }

    private struct CheckedMesh {
        var positions: [Float]
        var faces: [Int32]
        var colors: [Float]?
        var vertexCount: Int
        var faceCount: Int
    }

    private static func validated(_ mesh: Mesh) throws -> CheckedMesh {
        guard mesh.positions.count.isMultiple(of: 3) else {
            throw MeshExportError.inconsistentCounts("Position count must be a multiple of 3.")
        }
        guard mesh.faces.count.isMultiple(of: 3) else {
            throw MeshExportError.inconsistentCounts("Face index count must be a multiple of 3.")
        }
        let vertexCount = mesh.positions.count / 3
        let faceCount = mesh.faces.count / 3
        guard vertexCount > 0, faceCount > 0 else {
            throw MeshExportError.emptyMesh
        }
        if let colors = mesh.vertexColors {
            guard colors.count == vertexCount * 3 else {
                throw MeshExportError.inconsistentCounts(
                    "Vertex colors must contain \(vertexCount * 3) floats, got \(colors.count)."
                )
            }
        }
        for index in mesh.faces {
            guard index >= 0, Int(index) < vertexCount else {
                throw MeshExportError.faceVertexMismatch
            }
        }
        return CheckedMesh(
            positions: mesh.positions,
            faces: mesh.faces,
            colors: mesh.vertexColors,
            vertexCount: vertexCount,
            faceCount: faceCount
        )
    }

    private static func formatComponent(_ value: Float) -> String {
        let finite: Float = value.isFinite ? value : 0
        let normalized: Float = finite == 0 ? 0 : finite
        return String(
            format: "%.6f",
            locale: Locale(identifier: "en_US_POSIX"),
            Double(normalized)
        )
    }
}
