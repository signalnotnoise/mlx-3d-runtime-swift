import CoreGraphics
import Foundation
import MLX
import Testing

#if canImport(UIKit)
import CoreImage
import UIKit
#endif

@testable import MLX3DRuntime

struct ImageBackgroundRemovalTests {
    @Test func alphaBoundsFindsForeground() {
        let mask: [UInt8] = [
            0, 255, 255, 0,
            0, 200, 0, 0,
            0, 0, 0, 0,
        ]
        let bounds = ImageBackgroundRemoval.alphaBounds(data: mask, width: 4, height: 3, threshold: 10)
        #expect(bounds == CGRect(x: 1, y: 0, width: 2, height: 2))
    }

    @Test func alphaBoundsEmptyMaskIsNil() {
        let bounds = ImageBackgroundRemoval.alphaBounds(
            data: [UInt8](repeating: 0, count: 16), width: 4, height: 4, threshold: 10)
        #expect(bounds == nil)
    }

    @Test func squareCanvasMatchesUpstreamRatio() {
        #expect(ImageBackgroundRemoval.squareCanvasSide(foregroundWidth: 170, height: 100, ratio: 0.85) == 200)
        #expect(ImageBackgroundRemoval.squareCanvasSide(foregroundWidth: 100, height: 85, ratio: 0.85) == 118)
    }

    @Test func skippingCutoutDoesNotRequireASubjectMask() async {
        let image = solidImage(width: 8, height: 8, red: 255, green: 0, blue: 0, alpha: 255)
        await #expect(
            throws: MLX3DRuntimeError.weightsNotFound("/nonexistent/triposr_swift_ready.safetensors")
        ) {
            try await MLX3DRuntime.imageToMesh(
                cgImage: image,
                weightsURL: URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors"),
                options: ImageToMeshOptions(gridSize: 8, applyForegroundCutout: false)
            )
        }
    }

    #if os(macOS)
    @Test func foregroundCutoutIsSquareAndKeepsAlpha() async throws {
        let source = solidImage(width: 64, height: 48, red: 20, green: 20, blue: 20, alpha: 255)
        guard let cutout = try await ImageBackgroundRemoval.foregroundCutout(from: source, ratio: 0.85) else {
            // Vision found no foreground instance. The square-canvas math is covered above.
            return
        }
        #expect(cutout.width == cutout.height)
        #expect(cutout.width > 0)
        let keepsAlpha: Set<CGImageAlphaInfo> = [
            .premultipliedLast, .premultipliedFirst, .last, .first, .alphaOnly,
        ]
        #expect(keepsAlpha.contains(cutout.alphaInfo))
    }
    #endif

    @Test func cutoutFailurePolicyDefaultsToStrict() {
        #expect(ImageToMeshOptions().cutoutFailurePolicy == .strict)
    }

    @Test func strictNilThrowsAndFallbackKeepsTheOriginalImage() throws {
        let source = solidImage(width: 4, height: 4, red: 10, green: 20, blue: 30, alpha: 255)
        let cutout = solidImage(width: 4, height: 4, red: 200, green: 0, blue: 0, alpha: 255)
        #expect(
            throws: MLX3DRuntimeError.imageUnsupported("Vision foreground cutout produced no subject.")
        ) {
            try MLX3DRuntime.resolveForegroundCutout(
                source: source,
                outcome: .success(nil),
                policy: .strict
            )
        }
        let kept = try MLX3DRuntime.resolveForegroundCutout(
            source: source,
            outcome: .success(nil),
            policy: .fallbackToOriginal
        )
        #expect(kept === source)
        let strictCutout = try MLX3DRuntime.resolveForegroundCutout(
            source: source,
            outcome: .success(cutout),
            policy: .strict
        )
        #expect(strictCutout === cutout)
        let fallbackCutout = try MLX3DRuntime.resolveForegroundCutout(
            source: source,
            outcome: .success(cutout),
            policy: .fallbackToOriginal
        )
        #expect(fallbackCutout === cutout)
    }

    @Test func strictRethrowsAndFallbackKeepsTheOriginalImage() throws {
        let source = solidImage(width: 4, height: 4, red: 1, green: 2, blue: 3, alpha: 255)
        #expect(throws: PrepFailed.self) {
            try MLX3DRuntime.resolveForegroundCutout(
                source: source,
                outcome: .failure(PrepFailed()),
                policy: .strict
            )
        }
        let kept = try MLX3DRuntime.resolveForegroundCutout(
            source: source,
            outcome: .failure(PrepFailed()),
            policy: .fallbackToOriginal
        )
        #expect(kept === source)
    }

    @Test func fallbackCutoutStillReachesMissingWeights() async {
        let image = solidImage(width: 8, height: 8, red: 0, green: 0, blue: 0, alpha: 255)
        await #expect(
            throws: MLX3DRuntimeError.weightsNotFound("/nonexistent/triposr_swift_ready.safetensors")
        ) {
            try await MLX3DRuntime.imageToMesh(
                cgImage: image,
                weightsURL: URL(fileURLWithPath: "/nonexistent/triposr_swift_ready.safetensors"),
                options: ImageToMeshOptions(
                    gridSize: 8,
                    applyForegroundCutout: true,
                    cutoutFailurePolicy: .fallbackToOriginal
                )
            )
        }
    }

    @Test func shapEImageDefaultsMatchMesquetteImageMode() {
        let options = ShapEImageToMeshOptions()
        #expect(options.batchSize == 1)
        #expect(options.guidanceScale == 3)
        #expect(options.numInferenceSteps == 32)
        #expect(options.gridSize == 64)
        #expect(options.seed == 0)
        #expect(options.useMetalMarchingCubes == true)
        #expect(options.progressInterval == 0)
        #expect(options.applyWhiteBackgroundRemoval == true)
        #if os(macOS)
        #expect(options.queryBatchSize == 4096)
        #else
        #expect(options.queryBatchSize == 1024)
        #endif
    }

    @Test func whitePrepSkipDoesNotCallRemoval() async {
        let source = solidImage(width: 4, height: 4, red: 9, green: 9, blue: 9, alpha: 255)
        let calls = CallCount()
        let image = await MLX3DRuntime.prepareShapEImage(
            source,
            applyWhiteBackgroundRemoval: false
        ) { _ in
            calls.value += 1
            return nil
        }
        #expect(image === source)
        #expect(calls.value == 0)
    }

    @Test func whitePrepUsesStrippedImageAndKeepsOriginalOnNilOrThrow() async {
        let source = solidImage(width: 4, height: 4, red: 4, green: 5, blue: 6, alpha: 255)
        let stripped = solidImage(width: 4, height: 4, red: 7, green: 8, blue: 9, alpha: 255)
        let used = await MLX3DRuntime.prepareShapEImage(source, applyWhiteBackgroundRemoval: true) { _ in
            stripped
        }
        #expect(used === stripped)
        let nilKept = await MLX3DRuntime.prepareShapEImage(source, applyWhiteBackgroundRemoval: true) { _ in
            nil
        }
        #expect(nilKept === source)
        let thrownKept = await MLX3DRuntime.prepareShapEImage(source, applyWhiteBackgroundRemoval: true) { _ in
            throw PrepFailed()
        }
        #expect(thrownKept === source)
    }

    @Test func shapEImageEntryThrowsWhenTheBundleIsMissing() async {
        let image = solidImage(width: 8, height: 8, red: 255, green: 255, blue: 255, alpha: 255)
        await #expect(throws: MLX3DRuntimeError.bundleNotFound("/nonexistent/shap-e-bundle")) {
            try await MLX3DRuntime.imageToMeshShapE(
                cgImage: image,
                modelBundleURL: URL(fileURLWithPath: "/nonexistent/shap-e-bundle"),
                options: ShapEImageToMeshOptions(gridSize: 8, applyWhiteBackgroundRemoval: false)
            )
        }
    }

    private func solidImage(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        )!
        context.setFillColor(
            red: CGFloat(red) / 255,
            green: CGFloat(green) / 255,
            blue: CGFloat(blue) / 255,
            alpha: CGFloat(alpha) / 255
        )
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    #if canImport(UIKit)
    @Test func sessionUIImageCutoutPathFeedsUprightPixels() async throws {
        let reference = cornerImage()
        let expected = rgba(reference)
        let session = uiImageSession()
        for orientation in rotatedOrientations {
            let tagged = uiImage(reference, orientation: orientation)
            #expect(rgba(displayedPixels(of: tagged)) == expected)
            let input = try await session.modelInputCGImage(
                uiImage: tagged,
                options: ImageToMeshOptions(applyForegroundCutout: true, cutoutFailurePolicy: .strict),
                foregroundCutout: { image, _, imageOrientation in
                    #expect(imageOrientation == .up)
                    #expect(rgba(image) == expected)
                    return image
                }
            )
            #expect(rgba(input) == expected)
        }
    }

    @Test func sessionUIImageSkippedAndFallbackPathsFeedUprightPixels() async throws {
        let reference = cornerImage()
        let expected = rgba(reference)
        let session = uiImageSession()
        for orientation in rotatedOrientations {
            let tagged = uiImage(reference, orientation: orientation)
            #expect(rgba(displayedPixels(of: tagged)) == expected)
            let skipped = try await session.modelInputCGImage(
                uiImage: tagged,
                options: ImageToMeshOptions(applyForegroundCutout: false),
                foregroundCutout: { _, _, _ in
                    Issue.record("cutout ran when it was disabled")
                    return nil
                }
            )
            #expect(rgba(skipped) == expected)
            let fallback = try await session.modelInputCGImage(
                uiImage: tagged,
                options: ImageToMeshOptions(
                    applyForegroundCutout: true,
                    cutoutFailurePolicy: .fallbackToOriginal
                ),
                foregroundCutout: { image, _, imageOrientation in
                    #expect(imageOrientation == .up)
                    #expect(rgba(image) == expected)
                    return nil
                }
            )
            #expect(rgba(fallback) == expected)
        }
    }

    @Test func uiImageOrientationMapsToCGImagePropertyOrientation() {
        let cases: [(UIImage.Orientation, CGImagePropertyOrientation)] = [
            (.up, .up),
            (.down, .down),
            (.left, .left),
            (.right, .right),
            (.upMirrored, .upMirrored),
            (.downMirrored, .downMirrored),
            (.leftMirrored, .leftMirrored),
            (.rightMirrored, .rightMirrored),
        ]
        for (uiOrientation, imageIO) in cases {
            #expect(ImageBackgroundRemoval.cgImagePropertyOrientation(for: uiOrientation) == imageIO)
            let cast = CGImagePropertyOrientation(rawValue: UInt32(uiOrientation.rawValue))
            if uiOrientation == .rightMirrored {
                #expect(cast == .rightMirrored)
            } else {
                #expect(cast != imageIO)
            }
        }
    }

    @Test func uiImageRemoveBackgroundNormalizesEveryOrientation() async throws {
        let source = cornerImage()
        let uprightDisplay = displayedPixels(of: UIImage(cgImage: source, scale: 1, orientation: .up))

        for uiOrientation in Self.everyUIImageOrientation {
            let tagged = UIImage(cgImage: source, scale: 1, orientation: uiOrientation)
            let mapped = ImageBackgroundRemoval.cgImagePropertyOrientation(for: uiOrientation)
            let displayed = displayedPixels(of: tagged)
            let oriented = try #require(ciOriented(source, mapped))
            #expect(displayed.width == oriented.width)
            #expect(displayed.height == oriented.height)
            #expect(displayed.width == Int(tagged.size.width))
            #expect(displayed.height == Int(tagged.size.height))
            #expect(rgba(displayed) == rgba(oriented))

            let throughHelper = try await ImageBackgroundRemoval.removeBackground(from: tagged)
            let throughMapped = try await ImageBackgroundRemoval.removeBackground(
                from: source,
                orientation: mapped
            )
            #expect(rgba(throughHelper) == rgba(throughMapped))

            let uprightEquivalent = UIImage(cgImage: displayed, scale: 1, orientation: .up)
            let throughUpright = try await ImageBackgroundRemoval.removeBackground(from: uprightEquivalent)
            #expect(rgba(throughHelper) == rgba(throughUpright))
            if uiOrientation != .up {
                #expect(rgba(displayed) != rgba(uprightDisplay))
            }
        }
    }

    private var rotatedOrientations: [UIImage.Orientation] {
        [.right, .left, .down, .upMirrored]
    }

    private func uiImageSession() -> MLX3DSession {
        MLX3DSession(
            shapELoader: { _ in UIKitFakeShapE() },
            tripoLoader: { _ in UIKitFakeTripo() }
        )
    }

    /// Raw bitmap whose UIKit drawing at `orientation` is `reference`.
    private func uiImage(_ reference: CGImage, orientation: UIImage.Orientation) -> UIImage {
        let mapped = ImageBackgroundRemoval.cgImagePropertyOrientation(for: orientation)
        let raw = ciOriented(reference, inverseOrientation(mapped))!
        return UIImage(cgImage: raw, scale: 1, orientation: orientation)
    }

    private func inverseOrientation(
        _ orientation: CGImagePropertyOrientation
    ) -> CGImagePropertyOrientation {
        switch orientation {
        case .up:
            return .up
        case .upMirrored:
            return .upMirrored
        case .down:
            return .down
        case .downMirrored:
            return .downMirrored
        case .leftMirrored:
            return .leftMirrored
        case .right:
            return .left
        case .rightMirrored:
            return .rightMirrored
        case .left:
            return .right
        }
    }

    private static let everyUIImageOrientation: [UIImage.Orientation] = [
        .up, .down, .left, .right, .upMirrored, .downMirrored, .leftMirrored, .rightMirrored,
    ]

    private func cornerImage() -> CGImage {
        let pixels: [UInt8] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255,
            255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 255, 255,
        ]
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(
            width: 3,
            height: 2,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: 12,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    private func displayedPixels(of image: UIImage) -> CGImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        let rendered = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
        return rendered.cgImage!
    }

    private func ciOriented(_ image: CGImage, _ orientation: CGImagePropertyOrientation) -> CGImage? {
        let oriented = CIImage(cgImage: image).oriented(orientation)
        return CIContext().createCGImage(oriented, from: oriented.extent)
    }

    private func rgba(_ image: CGImage?) -> [UInt8]? {
        guard let image else { return nil }
        let width = image.width
        let height = image.height
        var data = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        guard let context else { return nil }
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return data
    }
    #endif
}

#if canImport(UIKit)
private struct UIKitFakeShapE: SessionShapE {
    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        [Mesh(positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], faces: [0, 1, 2])]
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        [Mesh(positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], faces: [0, 1, 2])]
    }
}

private struct UIKitFakeTripo: SessionTripoSR {
    func mesh(
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        Mesh(positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], faces: [0, 1, 2])
    }
}
#endif

private struct PrepFailed: Error {}

private final class CallCount: @unchecked Sendable {
    var value = 0
}
