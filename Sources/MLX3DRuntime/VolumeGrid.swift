import Foundation

public enum VolumeGrid {
    /// Smallest grid accepted by ``validate(resolution:maximum:)``.
    public static let minimumResolution = 2

    /// Hard cap. A larger `maximum` argument is lowered to this value.
    /// 512³ density samples are the largest grid this package will allocate.
    public static let maximumResolution = 512

    /// Reject a grid before any weight load or sample buffer is created.
    ///
    /// `maximum` can be lower than ``maximumResolution`` (for example
    /// `ImageToMeshOptions.maximumGridSize`). It cannot raise the hard cap.
    public static func validate(resolution: Int, maximum: Int = maximumResolution) throws {
        let cap = min(maximum, maximumResolution)
        guard resolution >= minimumResolution, resolution <= cap else {
            throw MLX3DRuntimeError.gridSizeOutOfRange(
                requested: resolution,
                minimum: minimumResolution,
                maximum: cap
            )
        }
    }

    /// One batch of packed `x, y, z` samples. X is the slowest index and Z is the fastest.
    ///
    /// `points` counts points, not floats. Sample `i` on an axis is
    /// `min + i / (resolution - 1) * (max - min)`.
    public static func positions(
        resolution: Int,
        min minCorner: SIMD3<Float>,
        max maxCorner: SIMD3<Float>,
        points: Range<Int>
    ) -> [Float] {
        let denom = Float(resolution - 1)
        let extent = maxCorner - minCorner
        let plane = resolution * resolution
        var values = [Float]()
        values.reserveCapacity(points.count * 3)
        for point in points {
            let ix = point / plane
            let iy = (point / resolution) % resolution
            let iz = point % resolution
            values.append(minCorner.x + (Float(ix) / denom) * extent.x)
            values.append(minCorner.y + (Float(iy) / denom) * extent.y)
            values.append(minCorner.z + (Float(iz) / denom) * extent.z)
        }
        return values
    }

    /// Every sample on the grid. Call ``validate(resolution:maximum:)`` first.
    /// Meshing uses ``positions(resolution:min:max:points:)`` one batch at a time instead.
    public static func positions(
        resolution: Int,
        min minCorner: SIMD3<Float>,
        max maxCorner: SIMD3<Float>
    ) throws -> [Float] {
        try validate(resolution: resolution)
        let count = resolution * resolution * resolution
        return positions(
            resolution: resolution,
            min: minCorner,
            max: maxCorner,
            points: 0..<count
        )
    }
}
