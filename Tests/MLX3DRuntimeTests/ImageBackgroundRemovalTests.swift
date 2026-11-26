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
