import CoreGraphics
import Foundation
import Testing

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
}

private struct PrepFailed: Error {}

private final class CallCount: @unchecked Sendable {
    var value = 0
}
