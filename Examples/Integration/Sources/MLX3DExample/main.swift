import MLX3DRuntime
import ShapERuntime
import TripoSRRuntime

/// Links MLX3DRuntime, ShapERuntime, and TripoSRRuntime in one executable.
/// It does not load weights.
@main
struct MLX3DExample {
    static func main() {
        let mesh = Mesh(
            positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            faces: [0, 1, 2]
        )
        print("MLX3DRuntime vertices \(mesh.vertexCount) faces \(mesh.faceCount)")
        print("ShapERuntime \(String(reflecting: MarchingCubes.self))")
        print("TripoSRRuntime \(String(reflecting: TripoSRModel.self))")
    }
}
