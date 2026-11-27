import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing

@testable import MLX3DRuntime

struct ImageTensorTests {
    init() {
        Device.setDefault(device: Device(.cpu))
    }

    @Test func missingImageThrows() {
        #expect(throws: MLX3DRuntimeError.self) {
            _ = try ImageTensor.rgb(contentsOf: URL(fileURLWithPath: "/nonexistent/mlx3d-image.png"))
        }
    }

    @Test func opaqueRedStaysRed() throws {
        let url = try writePNG(pixels: [255, 0, 0, 255], width: 1, height: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let tensor = try ImageTensor.rgb(contentsOf: url, side: 1, background: 0.5)
        #expect(tensor.shape == [1, 1, 1, 3])
        let values = tensor.asArray(Float.self)
        #expect(values.count == 3)
        #expect(abs(values[0] - 1) < 0.02)
        #expect(abs(values[1]) < 0.02)
        #expect(abs(values[2]) < 0.02)
    }

    @Test func aspectFitLetterboxesAndStretchFillsAWideImage() throws {
        let image = solidImage(width: 4, height: 2, red: 255, green: 0, blue: 0)
        let fit = try ImageTensor.rgb(
            cgImage: image,
            side: 4,
            background: 0,
            resizeMode: .aspectFit
        )
        let stretch = try ImageTensor.rgb(
            cgImage: image,
            side: 4,
            background: 0,
            resizeMode: .stretch
        )
        let fitValues = fit.asArray(Float.self)
        let stretchValues = stretch.asArray(Float.self)
        #expect(fitValues[0] < 0.05)
        #expect(stretchValues[0] > 0.95)
        let center = (2 * 4 + 2) * 3
        #expect(fitValues[center] > 0.95)
        #expect(stretchValues[center] > 0.95)
        #expect(ImageToMeshOptions().resizeMode == .aspectFit)
    }

    @Test func transparentPixelUsesBackground() throws {
        let url = try writePNG(pixels: [0, 0, 0, 0], width: 1, height: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let tensor = try ImageTensor.rgb(contentsOf: url, side: 1, background: 0.5)
        let values = tensor.asArray(Float.self)
        #expect(values.count == 3)
        for value in values {
            #expect(abs(value - 0.5) < 0.02)
        }
    }

    private func solidImage(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) -> CGImage {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 255, count: height * bytesPerRow)
        for pixel in 0..<(width * height) {
            pixels[pixel * 4] = red
            pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    private func writePNG(pixels: [UInt8], width: Int, height: Int) throws -> URL {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let data = Data(pixels)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard
            let provider = CGDataProvider(data: data as CFData),
            let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        else {
            throw MLX3DRuntimeError.imageUnsupported("Could not build the test image.")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-\(UUID().uuidString).png")
        guard
            let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else {
            throw MLX3DRuntimeError.imageUnsupported("Could not create a PNG destination.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw MLX3DRuntimeError.imageUnsupported("Could not write the PNG.")
        }
        return url
    }
}
