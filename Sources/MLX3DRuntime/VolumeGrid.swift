import Foundation

public enum VolumeGrid {
    /// Packed `x, y, z` samples. X is the slowest index and Z is the fastest.
    ///
    /// Sample `i` on an axis is `min + i / (resolution - 1) * (max - min)`,
    /// so the first sample is `min` and the last sample is `max`.
    public static func positions(
        resolution: Int,
        min minCorner: SIMD3<Float>,
        max maxCorner: SIMD3<Float>
    ) -> [Float] {
        precondition(resolution >= 2, "resolution must be at least 2")
        let sampleCount = resolution * resolution * resolution
        var values = [Float](repeating: 0, count: sampleCount * 3)
        let denom = Float(resolution - 1)
        let extent = maxCorner - minCorner
        var cursor = 0
        for ix in 0..<resolution {
            let x = minCorner.x + (Float(ix) / denom) * extent.x
            for iy in 0..<resolution {
                let y = minCorner.y + (Float(iy) / denom) * extent.y
                for iz in 0..<resolution {
                    let z = minCorner.z + (Float(iz) / denom) * extent.z
                    values[cursor] = x
                    values[cursor + 1] = y
                    values[cursor + 2] = z
                    cursor += 3
                }
            }
        }
        return values
    }
}
