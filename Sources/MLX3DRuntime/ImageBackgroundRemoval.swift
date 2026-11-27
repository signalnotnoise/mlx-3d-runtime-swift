// Source: signalnotnoise/Mesquette commit 971d381e32fba2e41feb979d4240b9b79ae01209
// Path: Mesquette/ImageBackgroundRemoval.swift
// The body below is copied from that commit, including the warning
// that removeBackground's white composite must not feed TripoSR.
// Adaptation: the UIImage helper maps UIImage.Orientation with an explicit
// switch. UIKit and CGImagePropertyOrientation do not share raw values.

import Foundation
import os
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics

#if canImport(UIKit)
import UIKit
#endif
#if canImport(AppKit)
import AppKit
#endif

public struct ImageBackgroundRemoval {
    nonisolated private static let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.visionarydevops.Mesquette", category: "image")

    
    
    
    
    // MARK: - Public APIs

    /// Vision segmentation + CoreImage compositing are thread-safe but
    /// synchronous and slow; public entry points are `@concurrent` so a
    /// caller on the main actor never blocks on `handler.perform`.
    /// Generates a mask for the foreground object and composites it over a white background.
    /// Returns a new CGImage with the background stripped out.
    /// Platform-agnostic entry that works with a CGImage and orientation.
    @concurrent
    public static func removeBackground(from cgImage: CGImage, orientation: CGImagePropertyOrientation = .up) async throws -> CGImage? {
        try removeBackgroundSync(from: cgImage, orientation: orientation)
    }

    nonisolated private static func removeBackgroundSync(from cgImage: CGImage, orientation: CGImagePropertyOrientation) throws -> CGImage? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        try handler.perform([request])
        
        guard let result = request.results?.first else {
            logger.error("failed to generate foreground mask")
            return nil
        }
        
        // Generate a mask buffer from the segmentation request
        let mask = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        let maskImage = CIImage(cvPixelBuffer: mask)
        let originalImage = CIImage(cgImage: cgImage)
        
        // Blend the original image using the mask
        let filter = CIFilter.blendWithMask()
        filter.inputImage = originalImage
        filter.maskImage = maskImage
        
        // Shap-E expects a solid white background (or transparent, but white works best generally)
        filter.backgroundImage = CIImage(color: .white).cropped(to: originalImage.extent)
        
        guard let outputImage = filter.outputImage else { return nil }
        
        // Create CGImage from the CoreImage output
        let context = CIContext()
        return context.createCGImage(outputImage, from: outputImage.extent)
    }

    // MARK: - TripoSR foreground preparation

    /// Transparent foreground cutout, cropped to the mask bounding box and
    /// padded to a square canvas so the foreground fills `ratio` of the
    /// frame — mirrors upstream TripoSR `resize_foreground` (default 0.85).
    /// Returns an alpha-preserving image; the TripoSR runtime composites it
    /// over gray itself. (The white-composited `removeBackground` above is
    /// Shap-E tuned and must not feed TripoSR: the white becomes geometry.)
    /// Runs off the caller's actor (see `removeBackground`).
    @concurrent
    public static func foregroundCutout(
        from cgImage: CGImage,
        ratio: CGFloat = 0.85,
        orientation: CGImagePropertyOrientation = .up
    ) async throws -> CGImage? {
        try foregroundCutoutSync(from: cgImage, ratio: ratio, orientation: orientation)
    }

    nonisolated private static func foregroundCutoutSync(
        from cgImage: CGImage,
        ratio: CGFloat,
        orientation: CGImagePropertyOrientation
    ) throws -> CGImage? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
        try handler.perform([request])
        guard let result = request.results?.first else { return nil }
        let mask = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        guard let bounds = maskBounds(of: mask) else { return nil }

        // Mask is image-sized here, so bounds map 1:1 into image pixels.
        let imageRect = CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height)
        // Vision/CoreImage use bottom-left origin; mask rows are top-first.
        let crop = CGRect(
            x: bounds.minX,
            y: imageRect.height - bounds.maxY,
            width: bounds.width,
            height: bounds.height
        ).intersection(imageRect)
        guard !crop.isNull, crop.width >= 1, crop.height >= 1 else { return nil }

        let maskImage = CIImage(cvPixelBuffer: mask)
        let filter = CIFilter.blendWithMask()
        filter.inputImage = CIImage(cgImage: cgImage)
        filter.maskImage = maskImage
        filter.backgroundImage = CIImage(color: .clear).cropped(to: imageRect)
        guard let cutout = filter.outputImage?.cropped(to: crop) else { return nil }

        // Pad to a transparent square so foreground fills `ratio` of frame.
        let side = CGFloat(squareCanvasSide(foregroundWidth: Int(crop.width), height: Int(crop.height), ratio: ratio))
        let dx = (side - crop.width) / 2 - crop.minX
        let dy = (side - crop.height) / 2 - crop.minY
        let placed = cutout.transformed(by: CGAffineTransform(translationX: dx, y: dy))
        let canvas = CIImage(color: .clear).cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        guard let composite = CIFilter(name: "CISourceOverCompositing") else { return nil }
        composite.setValue(placed, forKey: kCIInputImageKey)
        composite.setValue(canvas, forKey: kCIInputBackgroundImageKey)
        guard let output = composite.outputImage else { return nil }
        return CIContext().createCGImage(output, from: CGRect(x: 0, y: 0, width: side, height: side))
    }

    /// Row-major top-first alpha bounds. Pure helper (unit-tested).
    nonisolated public static func alphaBounds(data: [UInt8], width: Int, height: Int, threshold: UInt8) -> CGRect? {
        guard width > 0, height > 0, data.count >= width * height else { return nil }
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where data[y * width + x] > threshold {
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
    }

    /// Square canvas side so the foreground's long edge fills `ratio`.
    nonisolated public static func squareCanvasSide(foregroundWidth w: Int, height h: Int, ratio: CGFloat) -> Int {
        guard w > 0, h > 0, ratio > 0 else { return max(w, h, 1) }
        return Int((CGFloat(max(w, h)) / ratio).rounded())
    }

    nonisolated private static func maskBounds(of mask: CVPixelBuffer) -> CGRect? {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(mask) else { return nil }
        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)
        let rowBytes = CVPixelBufferGetBytesPerRow(mask)
        // Vision returns one-component Float32 ('L00f') on current OSes and
        // UInt8 ('L008') on older ones; read the actual element type.
        var data: [UInt8] = []
        data.reserveCapacity(width * height)
        switch CVPixelBufferGetPixelFormatType(mask) {
        case kCVPixelFormatType_OneComponent32Float:
            for y in 0..<height {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float.self)
                for x in 0..<width {
                    data.append(UInt8(max(0, min(255, row[x] * 255))))
                }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<height {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                data.append(contentsOf: UnsafeBufferPointer(start: row, count: width))
            }
        default:
            return nil
        }
        return alphaBounds(data: data, width: width, height: height, threshold: 10)
    }

    // MARK: - Convenience overloads per platform
    #if canImport(UIKit)
    /// Maps `UIImage.Orientation` onto `CGImagePropertyOrientation`.
    ///
    /// The cases share names and not raw values. `UIImage.Orientation.up` is 0,
    /// which is not a valid ImageIO orientation (`CGImagePropertyOrientation.up` is 1).
    static func cgImagePropertyOrientation(for orientation: UIImage.Orientation) -> CGImagePropertyOrientation {
        switch orientation {
        case .up:
            return .up
        case .down:
            return .down
        case .left:
            return .left
        case .right:
            return .right
        case .upMirrored:
            return .upMirrored
        case .downMirrored:
            return .downMirrored
        case .leftMirrored:
            return .leftMirrored
        case .rightMirrored:
            return .rightMirrored
        @unknown default:
            return .up
        }
    }

    /// iOS/tvOS/watchOS convenience: accepts UIImage.
    /// The image's orientation is passed to Vision; it is not a raw-value cast.
    public static func removeBackground(from uiImage: UIImage) async throws -> CGImage? {
        guard let cgImage = uiImage.cgImage else { return nil }
        let orientation = cgImagePropertyOrientation(for: uiImage.imageOrientation)
        return try await removeBackground(from: cgImage, orientation: orientation)
    }

    /// Bitmap with `uiImage.imageOrientation` already applied, so the result is `.up`.
    /// Uses the explicit orientation switch above. `.up` returns the original bitmap.
    static func uprightCGImage(from uiImage: UIImage) throws -> CGImage {
        guard let cgImage = uiImage.cgImage else {
            throw MLX3DRuntimeError.imageUnreadable("UIImage has no CGImage.")
        }
        let orientation = cgImagePropertyOrientation(for: uiImage.imageOrientation)
        guard orientation != .up else {
            return cgImage
        }
        let oriented = CIImage(cgImage: cgImage).oriented(orientation)
        guard let upright = CIContext().createCGImage(oriented, from: oriented.extent) else {
            throw MLX3DRuntimeError.imageUnsupported("Could not apply UIImage orientation.")
        }
        return upright
    }
    #endif

    #if canImport(AppKit)
    /// macOS convenience: accepts NSImage
    public static func removeBackground(from nsImage: NSImage) async throws -> CGImage? {
        guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        // NSImage doesn't carry EXIF orientation the same way; assume up.
        return try await removeBackground(from: cgImage, orientation: .up)
    }
    #endif
}
