// swift-tools-version: 6.2
import PackageDescription

// Consumer package. It depends on mlx-3d-runtime-swift by relative path and
// imports ShapERuntime and TripoSRRuntime in the same target. Clone this
// repository as `mlx-3d-runtime-swift` (the default git clone directory) so
// the path package identity matches the product references below.
//
// TODO: Switch the branch pins to `from: "0.1.0"` when the root Package.swift does.

let package = Package(
    name: "MLX3DExample",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .executable(name: "MLX3DExample", targets: ["MLX3DExample"]),
    ],
    dependencies: [
        .package(path: "../.."),
        // TODO: from: "0.1.0" once shap-e-mlx-swift is tagged.
        .package(
            url: "https://github.com/signalnotnoise/shap-e-mlx-swift.git",
            branch: "cursor/shap-e-runtime-package-ea5b"
        ),
        // TODO: from: "0.1.0" once triposr-mlx-swift is tagged.
        .package(
            url: "https://github.com/signalnotnoise/triposr-mlx-swift.git",
            branch: "cursor/triposr-mlx-swift-package-45c1"
        ),
    ],
    targets: [
        .executableTarget(
            name: "MLX3DExample",
            dependencies: [
                .product(name: "MLX3DRuntime", package: "mlx-3d-runtime-swift"),
                .product(name: "ShapERuntime", package: "shap-e-mlx-swift"),
                .product(name: "TripoSRRuntime", package: "triposr-mlx-swift"),
            ]
        ),
    ]
)
