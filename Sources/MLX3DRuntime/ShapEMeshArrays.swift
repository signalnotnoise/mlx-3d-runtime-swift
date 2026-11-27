import MLX
import ShapERuntime

/// Reads Shap-E vertex, face, and color buffers without calling ``MLXArray/asArray(_:)``
/// on an array that can trap.
///
/// mlx-swift 0.31.4 `MLXArray.asArray` (`MLXArray+Bytes.swift` lines 130-131) builds
/// `UnsafeRawBufferPointer(start: mlx_array_data_uint8, count: physicalSize * itemSize)`.
/// The pointer initializer traps, as a Swift fatal error, with
/// "UnsafeRawBufferPointer has a nil start and nonzero count" when `size` is 0
/// (the data pointer is nil) and `physicalSize` is still nonzero.
/// `physicalSize` is `max(abs(dimension * stride))`, so an empty `[0, 3]` partial
/// whose strides are `[3, 1]` has physical size 3. A transposed or otherwise
/// strided array can also trap in that contiguous copy when the physical buffer
/// is larger than the logical size. `try?` does not catch either trap.
enum ShapEMeshArrays {
    static func positions(from generated: ShapEMeshResult) throws -> [Float] {
        try floats(try generated.mlxVertices(), components: 3)
    }

    static func faces(from generated: ShapEMeshResult) throws -> [Int32] {
        try rejectUnrepresentableInt64Faces(generated.meshResult.faces)
        return try ints(try generated.mlxFaces(), components: 3)
    }

    /// RGBA floats, or nil when the mesh has no RGBA channel.
    /// A zero-size channel returns an empty buffer and does not call `asArray`.
    static func rgba(from generated: ShapEMeshResult) throws -> [Float]? {
        guard let colors = try generated.mlxTextureChannel("RGBA") else {
            return nil
        }
        if colors.size == 0 {
            return []
        }
        guard hasPackedLayout(colors, components: 4) else {
            throw MLX3DRuntimeError.colorCountMismatch(
                expected: (colors.size / 4) * 4,
                actual: colors.size
            )
        }
        return try readFloats(colors)
    }

    static func floats(_ array: MLXArray, components: Int) throws -> [Float] {
        guard hasPackedLayout(array, components: components) else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return try readFloats(array)
    }

    private static func ints(_ array: MLXArray, components: Int) throws -> [Int32] {
        guard hasPackedLayout(array, components: components) else {
            throw MLX3DRuntimeError.emptyMesh
        }
        let typed = array.dtype == .int32 ? array : array.asType(.int32)
        return try read(typed, Int32.self)
    }

    private static func rejectUnrepresentableInt64Faces(_ faces: RuntimeTensor) throws {
        guard faces.dataType == .int64 else { return }
        let wide = try faces.int64Values()
        guard wide.allSatisfy({ Int32(exactly: $0) != nil }) else {
            throw MLX3DRuntimeError.emptyMesh
        }
    }

    /// `size > 0`, rank at least 1, and the last dimension equal to `components`.
    private static func hasPackedLayout(_ array: MLXArray, components: Int) -> Bool {
        array.size > 0
            && array.ndim >= 1
            && array.shape.last == components
            && array.size.isMultiple(of: components)
    }

    private static func readFloats(_ array: MLXArray) throws -> [Float] {
        let typed = array.dtype == .float32 ? array : array.asType(.float32)
        return try read(typed, Float.self)
    }

    private static func read<T: HasDType>(_ array: MLXArray, _ type: T.Type) throws -> [T] {
        let packed = array.contiguous()
        guard packed.size > 0 else {
            throw MLX3DRuntimeError.emptyMesh
        }
        return packed.asArray(type)
    }
}
