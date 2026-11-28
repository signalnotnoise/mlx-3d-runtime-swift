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
    struct PartialArrays {
        var vertices: MLXArray
        var faces: MLXArray
        var rgba: MLXArray?
    }

    static func mesh(from generated: ShapEMeshResult) throws -> Mesh {
        try prepare(generated.meshResult.faces)
        return try mesh(
            vertices: try generated.mlxVertices(),
            faces: try generated.mlxFaces(),
            rgba: try generated.mlxTextureChannel("RGBA")
        )
    }

    /// Same conversion `mesh(from:)` uses, fed by arrays instead of a `ShapEMeshResult`.
    static func mesh(vertices: MLXArray, faces: MLXArray, rgba: MLXArray?) throws -> Mesh {
        let positions = try floats(vertices, components: 3)
        let indices = try ints(faces, components: 3)
        let colors = try vertexColors(from: rgba, vertexCount: positions.count / 3)
        return Mesh(positions: positions, faces: indices, vertexColors: colors)
    }

    static func kept(_ partials: [PartialArrays]) -> [Mesh] {
        partials.compactMap { partial in
            try? mesh(vertices: partial.vertices, faces: partial.faces, rgba: partial.rgba)
        }
    }

    static func kept(from partials: [ShapEMeshResult]) -> [Mesh] {
        partials.compactMap { partial in
            try? MLX3DRuntime.mesh(from: partial)
        }
    }

    static func prepare(_ faces: RuntimeTensor) throws {
        guard faces.dataType == .int64 else { return }
        _ = try int32s(fromWide: try faces.int64Values())
    }

    /// RGBA floats, or nil when the mesh has no RGBA channel.
    /// A zero-size channel returns an empty buffer and does not call `asArray`.
    /// The count is validated against the mesh's actual geometry vertex count.
    static func rgba(from generated: ShapEMeshResult) throws -> [Float]? {
        guard let colors = try generated.mlxTextureChannel("RGBA") else {
            return nil
        }
        return try validatedRGBAValues(colors, vertexCount: generated.meshResult.vertices.elementCount / 3)
    }

    private static func vertexColors(from rgba: MLXArray?, vertexCount: Int) throws -> [Float]? {
        guard let values = try validatedRGBAValues(rgba, vertexCount: vertexCount) else { return nil }
        var rgb = [Float]()
        rgb.reserveCapacity(vertexCount * 3)
        for vertex in 0..<vertexCount {
            rgb.append(values[vertex * 4])
            rgb.append(values[vertex * 4 + 1])
            rgb.append(values[vertex * 4 + 2])
        }
        return rgb
    }

    /// Shared RGBA validation for both color entry points.
    ///
    /// Shap-E records the RGBA channel flattened (`[4 * V]`), so — unlike
    /// positions/faces, which keep the strict packed-layout rule — colors
    /// accept either the flat form or any shape ending in 4. The scalar
    /// count must then match the *geometry* vertex count exactly.
    /// Nil stays nil and zero-size stays an empty buffer, without `asArray`.
    static func validatedRGBAValues(_ array: MLXArray?, vertexCount: Int) throws -> [Float]? {
        guard let array else { return nil }
        if array.size == 0 { return [] }
        let isFlat = array.ndim == 1 && array.size.isMultiple(of: 4)
        let isShaped = array.ndim >= 2 && array.shape.last == 4 && array.size.isMultiple(of: 4)
        guard isFlat || isShaped else {
            throw MLX3DRuntimeError.unsupportedColorShape(shape: array.shape)
        }
        let values = try readFloats(array)
        guard values.count == vertexCount * 4 else {
            throw MLX3DRuntimeError.colorCountMismatch(expected: vertexCount * 4, actual: values.count)
        }
        return values
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
        if array.dtype == .int64 {
            let packed = array.contiguous()
            guard packed.size > 0 else { throw MLX3DRuntimeError.emptyMesh }
            return try int32s(fromWide: packed.asArray(Int64.self))
        }
        let typed = array.dtype == .int32 ? array : array.asType(.int32)
        return try read(typed, Int32.self)
    }

    private static func int32s(fromWide values: [Int64]) throws -> [Int32] {
        try values.map { value in
            guard let converted = Int32(exactly: value) else {
                throw MLX3DRuntimeError.emptyMesh
            }
            return converted
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
