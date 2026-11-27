import CoreGraphics
import Foundation
import ImageIO
import MLX

/// How a non-square image is drawn into the square TripoSR tensor.
public enum ImageResizeMode: Sendable, Equatable {
    /// Letterbox the source inside the square. This is the default.
    case aspectFit
    /// Stretch the source to the square. Mesquette's previous TripoSR facade did this.
    case stretch
}

public enum ImageTensor {
    /// RGB float tensor `[1, side, side, 3]` in `0...1`.
    ///
    /// Alpha is drawn over a flat `background` (TripoSR's published chair run used 0.5).
    /// ``ImageResizeMode/aspectFit`` letterboxes the source. ``ImageResizeMode/stretch``
    /// fills the square. Row 0 is the top of the image.
    public static func rgb(
        contentsOf url: URL,
        side: Int = 512,
        background: Float = 0.5,
        resizeMode: ImageResizeMode = .aspectFit
    ) throws -> MLXArray {
        guard side >= 1 else {
            throw MLX3DRuntimeError.imageUnsupported("side must be at least 1, got \(side).")
        }
        let pixels = try rasterize(url: url, side: side, background: background, resizeMode: resizeMode)
        return MLXArray(pixels).reshaped([1, side, side, 3])
    }

    /// Same compositing as ``rgb(contentsOf:side:background:resizeMode:)`` for an in-memory image.
    public static func rgb(
        cgImage: CGImage,
        side: Int = 512,
        background: Float = 0.5,
        resizeMode: ImageResizeMode = .aspectFit
    ) throws -> MLXArray {
        guard side >= 1 else {
            throw MLX3DRuntimeError.imageUnsupported("side must be at least 1, got \(side).")
        }
        let pixels = try rasterize(image: cgImage, side: side, background: background, resizeMode: resizeMode)
        return MLXArray(pixels).reshaped([1, side, side, 3])
    }

    private static func rasterize(
        url: URL,
        side: Int,
        background: Float,
        resizeMode: ImageResizeMode
    ) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLX3DRuntimeError.imageUnreadable(url.path)
        }
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw MLX3DRuntimeError.imageUnreadable(url.path)
        }
        return try rasterize(image: image, side: side, background: background, resizeMode: resizeMode)
    }

    private static func rasterize(
        image: CGImage,
        side: Int,
        background: Float,
        resizeMode: ImageResizeMode
    ) throws -> [Float] {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard
            let context = CGContext(
                data: nil,
                width: side,
                height: side,
                bitsPerComponent: 8,
                bytesPerRow: side * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            )
        else {
            throw MLX3DRuntimeError.imageUnsupported("Could not allocate a \(side)×\(side) bitmap.")
        }

        let backgroundByte = UInt8((min(max(background, 0), 1) * 255).rounded())
        context.setFillColor(
            red: CGFloat(backgroundByte) / 255,
            green: CGFloat(backgroundByte) / 255,
            blue: CGFloat(backgroundByte) / 255,
            alpha: 1
        )
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))

        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw MLX3DRuntimeError.imageUnsupported("Image has a zero dimension.")
        }
        let drawRect: CGRect
        switch resizeMode {
        case .stretch:
            drawRect = CGRect(x: 0, y: 0, width: side, height: side)
        case .aspectFit:
            let scale = min(CGFloat(side) / sourceWidth, CGFloat(side) / sourceHeight)
            let drawWidth = sourceWidth * scale
            let drawHeight = sourceHeight * scale
            let originX = (CGFloat(side) - drawWidth) / 2
            let originY = (CGFloat(side) - drawHeight) / 2
            drawRect = CGRect(x: originX, y: originY, width: drawWidth, height: drawHeight)
        }
        context.interpolationQuality = .high
        context.draw(image, in: drawRect)

        guard let data = context.data else {
            throw MLX3DRuntimeError.imageUnsupported("Bitmap context has no pixels.")
        }
        let bytes = data.bindMemory(to: UInt8.self, capacity: side * side * 4)
        var rgb = [Float](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            let sourceY = side - 1 - y
            for x in 0..<side {
                let sourceIndex = (sourceY * side + x) * 4
                let destination = (y * side + x) * 3
                rgb[destination] = Float(bytes[sourceIndex]) / 255
                rgb[destination + 1] = Float(bytes[sourceIndex + 1]) / 255
                rgb[destination + 2] = Float(bytes[sourceIndex + 2]) / 255
            }
        }
        return rgb
    }
}
