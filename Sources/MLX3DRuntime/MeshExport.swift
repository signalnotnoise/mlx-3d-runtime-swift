// Copyright (c) 2026 Ismael Alex Otero
//
// OBJ+MTL and USDZ export is adapted from Mesquette/MeshExporter.swift
// in signalnotnoise/Mesquette at 971d381e32fba2e41feb979d4240b9b79ae01209.
// That repository is MIT Licensed, Copyright (c) 2026 signalnotnoise.
// This file keeps that notice. Share sheets, ZIP packaging, file naming,
// and scene placement stay in the app.
//
// Adaptation: `Mesh` is packed RGB, not a Shap-E `MeshResult` RGBA tensor.
// Face checks use this package's existing errors. A `.vertexRGB` option
// keeps the historical `v x y z r g b` writer. Up-axis conversion is
// explicit and export-only.

import Foundation
import ModelIO
#if canImport(SceneKit)
import SceneKit
#endif

public enum MeshExportError: Error, Equatable, LocalizedError, Sendable {
    case emptyMesh
    case inconsistentCounts(String)
    case faceVertexMismatch
    case unsupportedFormat(String)
    case exportFailed(String)

    public var errorDescription: String? {
        switch self {
        case .emptyMesh:
            return "Cannot export an empty mesh."
        case .inconsistentCounts(let reason):
            return reason
        case .faceVertexMismatch:
            return "Face index out of range for the vertex list."
        case .unsupportedFormat(let format):
            return "Unsupported mesh format '\(format)'. Use obj, ply, or usdz."
        case .exportFailed(let message):
            return "Export failed: \(message)"
        }
    }
}

/// Up axis written into an export. The in-memory `Mesh` is never modified.
///
/// Shap-E and TripoSR meshes in this package are Z-up. The body runs from
/// the legs to the back along Z, and the back is toward +Z. `.native` writes
/// those coordinates. `.yUp` is the only conversion, and only at export:
/// `(x, y, z)` becomes `(x, z, -y)`. That is a right-handed rotation by -90
/// degrees about X (determinant +1), so the back stands up on +Y and face
/// winding is unchanged.
public enum MeshExportUpAxis: Sendable, Equatable {
    case native
    case yUp
}

/// How OBJ carries color.
///
/// `.materialAverage` matches Mesquette: positions only on `v`, and the
/// average albedo in the companion MTL `Kd`. `.vertexRGB` is the historical
/// `v x y z r g b` extension and does not write an MTL. The extension is not
/// the default because strict OBJ readers reject the extra components.
public enum MeshOBJColors: Sendable, Equatable {
    case materialAverage
    case vertexRGB
}

public struct MeshExportOptions: Sendable, Equatable {
    public var upAxis: MeshExportUpAxis
    /// Read by OBJ export. PLY and USDZ ignore it.
    public var objColors: MeshOBJColors

    public init(
        upAxis: MeshExportUpAxis = .native,
        objColors: MeshOBJColors = .materialAverage
    ) {
        self.upAxis = upAxis
        self.objColors = objColors
    }
}

public enum MeshExport {
    /// Mesquette material name. OBJ `usemtl` and MTL `newmtl` both use it.
    public static let materialName = "VertexColorAverage"

    /// Write Wavefront OBJ. The default is OBJ + companion MTL.
    /// Pass `objColors: .vertexRGB` for the historical vertex-color file.
    public static func writeOBJ(
        _ mesh: Mesh,
        to url: URL,
        comment: String? = nil,
        options: MeshExportOptions = MeshExportOptions()
    ) throws {
        switch options.objColors {
        case .vertexRGB:
            let text = try objText(mesh, comment: comment, options: options)
            try text.write(to: url, atomically: true, encoding: .utf8)
        case .materialAverage:
            let mtlURL = materialURL(forOBJ: url)
            let text = try wavefrontText(
                mesh,
                mtlFileName: mtlURL.lastPathComponent,
                objectName: url.deletingPathExtension().lastPathComponent,
                comment: comment,
                options: options
            )
            try text.write(to: url, atomically: true, encoding: .utf8)
            try materialText(for: mesh).write(to: mtlURL, atomically: true, encoding: .utf8)
        }
    }

    /// Companion `.mtl` written next to an OBJ. Does not check that the file exists.
    public static func materialURL(forOBJ url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("mtl")
    }

    /// Write an ASCII PLY file. Colors are float `red` `green` `blue` in `0...1` when present.
    public static func writePLY(
        _ mesh: Mesh,
        to url: URL,
        options: MeshExportOptions = MeshExportOptions()
    ) throws {
        let text = try plyText(mesh, options: options)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Write USDZ. Model I/O does not export the `usdz` extension. A crate or
    /// USDA layer is packed into a 64-byte-aligned, uncompressed zip instead.
    public static func writeUSDZ(
        _ mesh: Mesh,
        to url: URL,
        options: MeshExportOptions = MeshExportOptions()
    ) throws {
        let checked = try validated(mesh, upAxis: options.upAxis)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        // A Model I/O crate imports with reordered vertices. USDA keeps the
        // written order, indices, colors, and the chosen up axis.
        try writeUSDZUSDA(checked, upAxis: options.upAxis, to: url)
    }

    /// Pick OBJ, PLY, or USDZ from the file extension.
    public static func write(
        _ mesh: Mesh,
        to url: URL,
        options: MeshExportOptions = MeshExportOptions()
    ) throws {
        switch url.pathExtension.lowercased() {
        case "obj":
            try writeOBJ(mesh, to: url, options: options)
        case "ply":
            try writePLY(mesh, to: url, options: options)
        case "usdz":
            try writeUSDZ(mesh, to: url, options: options)
        default:
            throw MeshExportError.unsupportedFormat(url.pathExtension)
        }
    }

    /// Historical OBJ text. Vertex colors are extra `v` components: `v x y z r g b`.
    /// This does not write an MTL. `options.objColors` is ignored. `upAxis` is applied.
    public static func objText(
        _ mesh: Mesh,
        comment: String? = nil,
        options: MeshExportOptions = MeshExportOptions()
    ) throws -> String {
        let checked = try validated(mesh, upAxis: options.upAxis)
        var text = ""
        text.reserveCapacity(checked.vertexCount * 48 + checked.faceCount * 24)
        appendComment(comment, to: &text)
        text += "# mlx-3d-runtime-swift vertices \(checked.vertexCount) faces \(checked.faceCount)\n"
        for vertex in 0..<checked.vertexCount {
            let (x, y, z) = component(checked.positions, vertex: vertex)
            if let colors = checked.colors {
                let r = formatComponent(colors[vertex * 3])
                let g = formatComponent(colors[vertex * 3 + 1])
                let b = formatComponent(colors[vertex * 3 + 2])
                text += "v \(x) \(y) \(z) \(r) \(g) \(b)\n"
            } else {
                text += "v \(x) \(y) \(z)\n"
            }
        }
        appendFaces(checked, oneBased: true, to: &text)
        return text
    }

    /// OBJ text that references `mtlFileName` and uses `materialName`. Positions only.
    public static func wavefrontText(
        _ mesh: Mesh,
        mtlFileName: String,
        objectName: String,
        comment: String? = nil,
        options: MeshExportOptions = MeshExportOptions()
    ) throws -> String {
        let checked = try validated(mesh, upAxis: options.upAxis)
        var text = ""
        text.reserveCapacity(checked.vertexCount * 32 + checked.faceCount * 24)
        appendComment(comment, to: &text)
        text += "# mlx-3d-runtime-swift vertices \(checked.vertexCount) faces \(checked.faceCount)\n"
        text += "mtllib \(mtlFileName)\n"
        text += "o \(sanitize(objectName))\n"
        text += "usemtl \(materialName)\n"
        for vertex in 0..<checked.vertexCount {
            let (x, y, z) = component(checked.positions, vertex: vertex)
            text += "v \(x) \(y) \(z)\n"
        }
        appendFaces(checked, oneBased: true, to: &text)
        return text
    }

    /// Companion MTL. `Kd` is the average vertex albedo, or `0.75 0.75 0.75` when the mesh has no colors.
    public static func materialText(for mesh: Mesh) throws -> String {
        let checked = try validated(mesh, upAxis: .native)
        let albedo = averageAlbedo(checked.colors, vertexCount: checked.vertexCount)
        return """
        # mlx-3d-runtime-swift MTL
        newmtl \(materialName)
        Ka \(formatComponent(0.1)) \(formatComponent(0.1)) \(formatComponent(0.1))
        Kd \(formatComponent(albedo.0)) \(formatComponent(albedo.1)) \(formatComponent(albedo.2))
        Ks \(formatComponent(0.05)) \(formatComponent(0.05)) \(formatComponent(0.05))
        Ns 20
        d \(formatComponent(1))
        illum 2
        """
    }

    public static func plyText(
        _ mesh: Mesh,
        options: MeshExportOptions = MeshExportOptions()
    ) throws -> String {
        let checked = try validated(mesh, upAxis: options.upAxis)
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
            let (x, y, z) = component(checked.positions, vertex: vertex)
            if let colors = checked.colors {
                let r = formatComponent(colors[vertex * 3])
                let g = formatComponent(colors[vertex * 3 + 1])
                let b = formatComponent(colors[vertex * 3 + 2])
                text += "\(x) \(y) \(z) \(r) \(g) \(b)\n"
            } else {
                text += "\(x) \(y) \(z)\n"
            }
        }
        appendFaces(checked, oneBased: false, to: &text)
        return text
    }

    private struct CheckedMesh {
        var positions: [Float]
        var faces: [UInt32]
        var colors: [Float]?
        var vertexCount: Int
        var faceCount: Int
    }

    /// Rejects empty meshes, non-triples, negative indices, and indices past the vertex list.
    private static func validated(_ mesh: Mesh, upAxis: MeshExportUpAxis) throws -> CheckedMesh {
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
        let indices = try validatedFaceIndices(mesh.faces, vertexCount: vertexCount)
        return CheckedMesh(
            positions: applying(upAxis, to: mesh.positions),
            faces: indices,
            colors: mesh.vertexColors,
            vertexCount: vertexCount,
            faceCount: faceCount
        )
    }

    /// Converts signed face indices to `UInt32` without trapping.
    private static func validatedFaceIndices(_ raw: [Int32], vertexCount: Int) throws -> [UInt32] {
        guard vertexCount >= 0, vertexCount <= Int(UInt32.max) else {
            throw MeshExportError.faceVertexMismatch
        }
        let vertexLimit = UInt32(vertexCount)
        var indices = [UInt32]()
        indices.reserveCapacity(raw.count)
        for value in raw {
            guard let index = UInt32(exactly: value), index < vertexLimit else {
                throw MeshExportError.faceVertexMismatch
            }
            indices.append(index)
        }
        return indices
    }

    /// Export-only. `.native` returns the same storage. `.yUp` allocates a new buffer.
    private static func applying(_ upAxis: MeshExportUpAxis, to positions: [Float]) -> [Float] {
        switch upAxis {
        case .native:
            return positions
        case .yUp:
            var converted = [Float]()
            converted.reserveCapacity(positions.count)
            var offset = 0
            while offset < positions.count {
                let x = positions[offset]
                let y = positions[offset + 1]
                let z = positions[offset + 2]
                converted.append(x)
                converted.append(z)
                converted.append(-y)
                offset += 3
            }
            return converted
        }
    }

    private static func makeMDLAsset(_ geometry: CheckedMesh) throws -> MDLAsset {
        let allocator = MDLMeshBufferDataAllocator()
        let positionBuffer = allocator.newBuffer(with: data(floats: geometry.positions), type: .vertex)

        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(
            name: MDLVertexAttributePosition,
            format: .float3,
            offset: 0,
            bufferIndex: 0
        )
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: MemoryLayout<Float>.size * 3)

        var vertexBuffers: [MDLMeshBuffer] = [positionBuffer]
        if let colors = geometry.colors {
            let colorBuffer = allocator.newBuffer(with: data(floats: colors), type: .vertex)
            descriptor.attributes[1] = MDLVertexAttribute(
                name: MDLVertexAttributeColor,
                format: .float3,
                offset: 0,
                bufferIndex: 1
            )
            descriptor.layouts[1] = MDLVertexBufferLayout(stride: MemoryLayout<Float>.size * 3)
            vertexBuffers.append(colorBuffer)
        }

        let indexBuffer = allocator.newBuffer(with: data(indices: geometry.faces), type: .index)
        let submesh = MDLSubmesh(
            indexBuffer: indexBuffer,
            indexCount: geometry.faces.count,
            indexType: .uInt32,
            geometryType: .triangles,
            material: nil
        )
        let mdlMesh = MDLMesh(
            vertexBuffers: vertexBuffers,
            vertexCount: geometry.vertexCount,
            descriptor: descriptor,
            submeshes: [submesh]
        )
        let asset = MDLAsset(bufferAllocator: allocator)
        asset.add(mdlMesh)
        return asset
    }

    private static func averageAlbedo(_ colors: [Float]?, vertexCount: Int) -> (Float, Float, Float) {
        guard let colors, vertexCount > 0, colors.count >= vertexCount * 3 else {
            return (0.75, 0.75, 0.75)
        }
        var red: Float = 0
        var green: Float = 0
        var blue: Float = 0
        for vertex in 0..<vertexCount {
            let base = vertex * 3
            red += colors[base]
            green += colors[base + 1]
            blue += colors[base + 2]
        }
        let count = Float(vertexCount)
        return (red / count, green / count, blue / count)
    }

    private static func sanitize(_ name: String) -> String {
        let cleaned = name.replacingOccurrences(of: " ", with: "_")
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        let mapped = String(cleaned.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" })
        return mapped.isEmpty ? "mesh" : mapped
    }

    private static func appendComment(_ comment: String?, to text: inout String) {
        guard let comment else { return }
        for line in comment.split(separator: "\n", omittingEmptySubsequences: false) {
            text += "# \(line)\n"
        }
    }

    private static func appendFaces(_ mesh: CheckedMesh, oneBased: Bool, to text: inout String) {
        for face in 0..<mesh.faceCount {
            let i0 = mesh.faces[face * 3]
            let i1 = mesh.faces[face * 3 + 1]
            let i2 = mesh.faces[face * 3 + 2]
            if oneBased {
                text += "f \(i0 + 1) \(i1 + 1) \(i2 + 1)\n"
            } else {
                text += "3 \(i0) \(i1) \(i2)\n"
            }
        }
    }

    private static func component(_ positions: [Float], vertex: Int) -> (String, String, String) {
        let base = vertex * 3
        return (
            formatComponent(positions[base]),
            formatComponent(positions[base + 1]),
            formatComponent(positions[base + 2])
        )
    }

    private static func data(floats: [Float]) -> Data {
        floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private static func data(indices: [UInt32]) -> Data {
        indices.withUnsafeBufferPointer { Data(buffer: $0) }
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

    /// Model I/O exports `usdc`/`usda`, not `usdz`. Zip the crate on 64-byte boundaries.
    private static func writeUSDZCrate(_ geometry: CheckedMesh, to url: URL) throws -> Bool {
        let ext: String
        if MDLAsset.canExportFileExtension("usdc") {
            ext = "usdc"
        } else if MDLAsset.canExportFileExtension("usda") {
            ext = "usda"
        } else {
            return false
        }
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-\(UUID().uuidString)")
            .appendingPathExtension(ext)
        defer { try? FileManager.default.removeItem(at: temp) }
        do {
            try makeMDLAsset(geometry).export(to: temp)
        } catch {
            return false
        }
        let payload = try Data(contentsOf: temp)
        guard !payload.isEmpty else { return false }
        try USDZArchive.pack([("mesh.\(ext)", payload)]).write(to: url, options: .atomic)
        return true
    }

    /// SceneKit packages USDZ directly. `canExportFileExtension("usdz")` stays false.
    private static func writeUSDZSceneKit(_ geometry: CheckedMesh, to url: URL) -> Bool {
        #if canImport(SceneKit)
        guard Bundle.main.bundleURL.pathExtension == "app" else { return false }
        let scene = SCNScene()
        scene.rootNode.addChildNode(SCNNode(geometry: sceneGeometry(geometry)))
        guard scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            try? FileManager.default.removeItem(at: url)
            return false
        }
        guard let data = try? Data(contentsOf: url), data.starts(with: Data([0x50, 0x4b])) else {
            try? FileManager.default.removeItem(at: url)
            return false
        }
        return true
        #else
        return false
        #endif
    }

    private static func writeUSDZUSDA(
        _ geometry: CheckedMesh,
        upAxis: MeshExportUpAxis,
        to url: URL
    ) throws {
        let text = usdaText(geometry, upAxis: upAxis)
        try USDZArchive.pack([("mesh.usda", Data(text.utf8))]).write(to: url, options: .atomic)
    }

    #if canImport(SceneKit)
    private static func sceneGeometry(_ geometry: CheckedMesh) -> SCNGeometry {
        let stride = MemoryLayout<Float>.size * 3
        var sources = [
            SCNGeometrySource(
                data: data(floats: geometry.positions),
                semantic: .vertex,
                vectorCount: geometry.vertexCount,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: stride
            )
        ]
        if let colors = geometry.colors {
            sources.append(
                SCNGeometrySource(
                    data: data(floats: colors),
                    semantic: .color,
                    vectorCount: geometry.vertexCount,
                    usesFloatComponents: true,
                    componentsPerVector: 3,
                    bytesPerComponent: MemoryLayout<Float>.size,
                    dataOffset: 0,
                    dataStride: stride
                )
            )
        }
        let element = SCNGeometryElement(
            data: data(indices: geometry.faces),
            primitiveType: .triangles,
            primitiveCount: geometry.faceCount,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )
        let mesh = SCNGeometry(sources: sources, elements: [element])
        mesh.subdivisionLevel = 0
        return mesh
    }
    #endif

    private static func usdaText(_ geometry: CheckedMesh, upAxis: MeshExportUpAxis) -> String {
        let axis = upAxis == .yUp ? "Y" : "Z"
        var counts = [String]()
        counts.reserveCapacity(geometry.faceCount)
        for _ in 0..<geometry.faceCount {
            counts.append("3")
        }
        var indices = [String]()
        indices.reserveCapacity(geometry.faces.count)
        for index in geometry.faces {
            indices.append(String(index))
        }
        var points = [String]()
        points.reserveCapacity(geometry.vertexCount)
        for vertex in 0..<geometry.vertexCount {
            let base = vertex * 3
            points.append(
                "(\(formatComponent(geometry.positions[base])), \(formatComponent(geometry.positions[base + 1])), \(formatComponent(geometry.positions[base + 2])))"
            )
        }
        var colors = ""
        if let values = geometry.colors {
            var tuples = [String]()
            tuples.reserveCapacity(geometry.vertexCount)
            for vertex in 0..<geometry.vertexCount {
                let base = vertex * 3
                tuples.append(
                    "(\(formatComponent(values[base])), \(formatComponent(values[base + 1])), \(formatComponent(values[base + 2])))"
                )
            }
            colors = """
                color3f[] primvars:displayColor = [\(tuples.joined(separator: ", "))] (
                    interpolation = "vertex"
                )

            """
        }
        return """
        #usda 1.0
        (
            defaultPrim = "Root"
            metersPerUnit = 1
            upAxis = "\(axis)"
        )

        def Xform "Root"
        {
            def Mesh "Mesh"
            {
                uniform token subdivisionScheme = "none"
                int[] faceVertexCounts = [\(counts.joined(separator: ", "))]
                int[] faceVertexIndices = [\(indices.joined(separator: ", "))]
                point3f[] points = [\(points.joined(separator: ", "))]
        \(colors)    }
        }

        """
    }
}

/// Uncompressed zip with each file payload starting on a 64-byte boundary.
private enum USDZArchive {
    static func pack(_ entries: [(String, Data)]) -> Data {
        var local = Data()
        var central = Data()
        for (name, payload) in entries {
            let filename = Data(name.utf8)
            let crc = crc32(payload)
            let offset = local.count
            let padding = extraPadding(offset: offset, nameLength: filename.count)
            let extra = Data(count: padding)
            let size = UInt32(payload.count)
            appendLocalHeader(&local, name: filename, extra: extra, crc: crc, size: size)
            local.append(payload)
            appendCentralHeader(
                &central,
                name: filename,
                extra: extra,
                crc: crc,
                size: size,
                localOffset: UInt32(offset)
            )
        }
        let centralOffset = UInt32(local.count)
        let centralSize = UInt32(central.count)
        local.append(central)
        appendEnd(&local, count: UInt16(entries.count), centralSize: centralSize, centralOffset: centralOffset)
        return local
    }

    private static func extraPadding(offset: Int, nameLength: Int) -> Int {
        let remainder = (offset + 30 + nameLength) % 64
        if remainder == 0 { return 0 }
        return 64 - remainder
    }

    private static func appendLocalHeader(
        _ data: inout Data,
        name: Data,
        extra: Data,
        crc: UInt32,
        size: UInt32
    ) {
        appendUInt32(&data, 0x04034b50)
        appendUInt16(&data, 20)
        appendUInt16(&data, 1 << 11)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt32(&data, crc)
        appendUInt32(&data, size)
        appendUInt32(&data, size)
        appendUInt16(&data, UInt16(name.count))
        appendUInt16(&data, UInt16(extra.count))
        data.append(name)
        data.append(extra)
    }

    private static func appendCentralHeader(
        _ data: inout Data,
        name: Data,
        extra: Data,
        crc: UInt32,
        size: UInt32,
        localOffset: UInt32
    ) {
        appendUInt32(&data, 0x02014b50)
        appendUInt16(&data, 20)
        appendUInt16(&data, 20)
        appendUInt16(&data, 1 << 11)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt32(&data, crc)
        appendUInt32(&data, size)
        appendUInt32(&data, size)
        appendUInt16(&data, UInt16(name.count))
        appendUInt16(&data, UInt16(extra.count))
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt32(&data, 0)
        appendUInt32(&data, localOffset)
        data.append(name)
        data.append(extra)
    }

    private static func appendEnd(
        _ data: inout Data,
        count: UInt16,
        centralSize: UInt32,
        centralOffset: UInt32
    ) {
        appendUInt32(&data, 0x06054b50)
        appendUInt16(&data, 0)
        appendUInt16(&data, 0)
        appendUInt16(&data, count)
        appendUInt16(&data, count)
        appendUInt32(&data, centralSize)
        appendUInt32(&data, centralOffset)
        appendUInt16(&data, 0)
    }

    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >>= 1
                }
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}
