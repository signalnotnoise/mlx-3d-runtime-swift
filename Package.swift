// swift-tools-version: 6.2
import PackageDescription

// TODO: Switch both sibling dependencies to `from: "0.1.0"` once those
// repositories are tagged. They are still branch-pinned because 0.1.0
// does not exist yet.
//
// shap-e-mlx-swift: `.package(url: "https://github.com/signalnotnoise/shap-e-mlx-swift.git", from: "0.1.0")`
// triposr-mlx-swift: `.package(url: "https://github.com/signalnotnoise/triposr-mlx-swift.git", from: "0.1.0")`

let package = Package(
    name: "MLX3DRuntime",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLX3DRuntime",
            targets: ["MLX3DRuntime"]
        ),
        .executable(
            name: "MLX3DCLI",
            targets: ["MLX3DCLI"]
        ),
    ],
    dependencies: [
        // TripoSRRuntime allows only mlx-swift 0.31.3..<0.31.5 (0.31.5 needs
        // Swift tools 6.3). ShapERuntime allows 0.29.1 and later. This range
        // is the overlap, so one mlx-swift version satisfies both packages.
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.31.3"..<"0.31.5"),
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
        .target(
            name: "MLX3DRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "ShapERuntime", package: "shap-e-mlx-swift"),
                .product(name: "TripoSRRuntime", package: "triposr-mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "MLX3DCLI",
            dependencies: ["MLX3DRuntime"]
        ),
        .testTarget(
            name: "MLX3DRuntimeTests",
            dependencies: [
                "MLX3DRuntime",
                "MLX3DCLI",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ShapERuntime", package: "shap-e-mlx-swift"),
                .product(name: "TripoSRRuntime", package: "triposr-mlx-swift"),
            ]
        ),
    ]
)
