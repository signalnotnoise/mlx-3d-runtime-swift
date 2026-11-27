# MLX3DRuntime

Swift package for text-to-3D and image-to-3D on Apple Silicon. Text prompts go through [shap-e-mlx-swift](https://github.com/signalnotnoise/shap-e-mlx-swift) (`ShapERuntime`). Images go through [triposr-mlx-swift](https://github.com/signalnotnoise/triposr-mlx-swift) (`TripoSRRuntime`). Both paths return one `Mesh` and can write OBJ or PLY.

signalnotnoise maintains this project. It is an integration layer over those two packages. It does not modify them.

## Install

There is no version tag yet. The sibling packages are public and not tagged yet either.

TODO: switch both sibling dependencies to `from: "0.1.0"` once `shap-e-mlx-swift` and `triposr-mlx-swift` are tagged. Until then `Package.swift` pins both to public branch `main`. The same TODO is in `Package.swift`.

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Example",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    dependencies: [
        .package(url: "https://github.com/signalnotnoise/mlx-3d-runtime-swift.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "Example",
            dependencies: [
                .product(name: "MLX3DRuntime", package: "mlx-3d-runtime-swift"),
            ]
        ),
    ]
)
```

`mlx-swift` is declared as `0.31.3..<0.31.5`. Both public sibling manifests use that same range. `0.31.5` and later set Swift tools 6.3.

Dependencies are the public GitHub URLs, not local paths. CI deletes the SwiftPM cache and resolves `main` from GitHub before it builds.

| Product | Kind |
| --- | --- |
| `MLX3DRuntime` | Library. Import this. |
| `MLX3DCLI` | Executable. `text` and `image` subcommands. |

## Quick start

Weights are not in this repository.

Shap-E reads a runtime bundle from [signalNotNoise/shap-e-mlx](https://huggingface.co/signalNotNoise/shap-e-mlx) (revision `d71f70a09b9c03535188d006a719edda5c19a6eb` in the Shap-E package README). `manifest.json` and `bpe_simple_vocab_16e6.txt.gz` sit in that directory.

TripoSR reads `triposr_swift_ready.safetensors` from [signalNotNoise/triposr-mlx](https://huggingface.co/signalNotNoise/triposr-mlx) (revision `76f3028746d0b5f39b6ba0205ede9e28835f106d`).

```swift
import MLX3DRuntime

let mesh = try await MLX3DRuntime.textToMesh(
    prompt: "a red chair",
    modelBundleURL: bundleURL
)
try MeshExport.writeOBJ(mesh, to: URL(fileURLWithPath: "chair.obj"))

let scanned = try MLX3DRuntime.imageToMesh(
    imageURL: imageURL,
    weightsURL: weightsURL
)
try MeshExport.writePLY(scanned, to: URL(fileURLWithPath: "scan.ply"))
```

Those static calls load weights on every request. Hold an `MLX3DSession` when the same weights should stay loaded. A second call with the same key reuses the model. Concurrent calls for that key share one in-flight load. A failed load is not cached. `unload` drops the model and, if a load is still running, does not put it back; waiters throw `MLX3DRuntimeError.modelUnloaded`. `evict` drops a model that is already loaded and leaves an in-flight load alone. The session does not download weights or decide when to preload.

```swift
let session = MLX3DSession()
let key = MLX3DModelKey.shapE(modelBundleURL: bundleURL)
try await session.load(key)

let first = try await session.textToMesh(prompt: "a red chair", modelBundleURL: bundleURL)
let second = try await session.textToMesh(prompt: "a blue lamp", modelBundleURL: bundleURL)
await session.unload(key)
```

The Shap-E key is the bundle directory plus the BPE file (default `bpe_simple_vocab_16e6.txt.gz` inside the bundle). The TripoSR key is the safetensors URL. Prompt, seed, grid size, and steps are not part of the key.

`Mesh.positions` and `Mesh.vertexColors` are packed `x, y, z` and `r, g, b`. `Mesh.faces` are packed triangle indices. `vertexColors` is `nil` when a path has no colors. Shap-E's RGBA channel is stored as RGB.

Image-to-mesh decodes the file, composites alpha over gray `0.5`, and aspect-fits it into a 512×512 RGB tensor. It then queries TripoSR density on a 128³ grid in batches of 8192, runs marching cubes at threshold 25, and queries TripoSR for a color at each vertex. Positions for each density batch are built when that batch runs. `--grid-size` must be from 2 through 512 (`VolumeGrid.maximumResolution`). A larger size fails before weights load. `ImageToMeshOptions.maximumGridSize` can set a lower cap. It cannot raise 512. The default radius is the model's renderer radius, `0.87`.

Marching cubes is Shap-E's public `MarchingCubes.extract`. The density grid is converted with `density - threshold`, so the interior is positive and the isosurface is 0. That is the same field TripoSR passes to marching cubes after its double negation (`-(density - threshold)` is negated again inside the helper). Vertices stay in X, Y, Z. Shap-E's extractor already returns that order.

```bash
MLX3DCLI text --prompt "a red chair" --bundle runtime_bundle --output chair.obj
MLX3DCLI image --input photo.png --weights triposr_swift_ready.safetensors --output scan.ply
```

`text` also reads `SHAP_E_BUNDLE`. `image` also reads `TRIPOSR_WEIGHTS_PATH` and `./triposr_swift_ready.safetensors`. A missing bundle, missing weights, or any thrown error prints `Failed:` on stderr and exits nonzero.

`Examples/Integration` is a separate executable package. It depends on this repository with `path: "../.."` and imports `MLX3DRuntime`, `ShapERuntime`, and `TripoSRRuntime` in one target. Clone this repo so the directory name is `mlx-3d-runtime-swift` (git's default). From that checkout:

```bash
cd Examples/Integration
xcodebuild build \
  -scheme MLX3DExample \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  CODE_SIGNING_ALLOWED=NO
```

The example does not load weights. It is the check that both sibling products link together.

## Public API

```swift
public struct Mesh: Equatable, Sendable {
    public var positions: [Float]
    public var faces: [Int32]
    public var vertexColors: [Float]?
    public var vertexCount: Int { get }
    public var faceCount: Int { get }
    public var boundaryEdgeCount: Int { get }
    public var invalidFaceCount: Int { get }
    public var isClosed: Bool { get }
}

public enum MeshExport {
    public static func writeOBJ(_ mesh: Mesh, to url: URL, comment: String? = nil) throws
    public static func writePLY(_ mesh: Mesh, to url: URL) throws
    public static func write(_ mesh: Mesh, to url: URL) throws
}

public struct TextToMeshOptions: Sendable { /* guidance 15, 64 steps, grid 128, batch 65536, seed 0 */ }
public struct ImageToMeshOptions: Sendable { /* grid 128, threshold 25, batch 8192, radius nil, side 512, background 0.5, foreground cutout on at ratio 0.85, cutoutFailurePolicy .strict */ }
public struct ShapEImageToMeshOptions: Sendable { /* guidance 3, 32 steps, grid 64, seed 0, white-background removal on */ }

public enum MLX3DRuntime {
    public static func textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:) async throws -> Mesh
    public static func imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:) async throws -> [Mesh]
    public static func imageToMesh(imageURL:weightsURL:options:) throws -> Mesh
    public static func imageToMesh(cgImage:weightsURL:options:) async throws -> Mesh
    public static func imageToMesh(image:weightsURL:options:) throws -> Mesh
}

public enum MLX3DBackend { case shapE, tripoSR }

public struct MLX3DModelKey: Sendable, Hashable {
    public let backend: MLX3DBackend
    public let bundleURL: URL
    public let auxiliaryURLs: [URL]
    public static func shapE(modelBundleURL:bpeVocabularyURL:) -> MLX3DModelKey
    public static func tripoSR(weightsURL:) -> MLX3DModelKey
}

public final class MLX3DSession: Sendable {
    public init()
    public func load(_ key: MLX3DModelKey) async throws
    public func unload(_ key: MLX3DModelKey) async
    public func unloadAll() async
    public func evict(_ key: MLX3DModelKey) async
    public func evictAll() async
    public func textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:) async throws -> Mesh
    public func imageToMesh(imageURL:weightsURL:options:) async throws -> Mesh
    public func imageToMesh(image:weightsURL:options:) async throws -> Mesh
}

public enum ImageTensor {
    public static func rgb(contentsOf:side:background:) throws -> MLXArray
    public static func rgb(cgImage:side:background:) throws -> MLXArray
}

public enum VolumeGrid {
    public static func positions(resolution:min:max:) -> [Float]
}

public enum MarchingCubesExtractor {
    public static func extract(field:resolution:threshold:bboxMin:bboxMax:vertexColorsAt:) throws -> Mesh
}
```

`image` on the tensor overload is float RGB in `0...1`, shaped `[height, width, 3]` or `[1, height, width, 3]`. That overload does not resize.

## Build and test

CI uses `xcodebuild` on macOS. mlx-swift's Metal library is produced by the Xcode build, so `swift test` is not the path this repo uses.

```bash
xcodebuild test \
  -scheme MLX3DRuntime-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

Unit tests cover mesh topology, OBJ and PLY export, a synthetic ellipsoid through marching cubes (closed, 0 boundary edges), image decoding, CLI exits when the bundle or weights are missing, and the loaded-model cache. Cache tests inject a fake loader. They do not read weight files. They check reuse, one shared in-flight load, a failed load that is retried, and `unload` discarding a load that was already running. Loading real weights is optional: those tests skip when `TRIPOSR_WEIGHTS_PATH` / `triposr_swift_ready.safetensors` or `SHAP_E_BUNDLE` is absent. `xcodebuild` only forwards variables prefixed with `TEST_RUNNER_` into the test process.

## Scope

Platforms in `Package.swift` are macOS 14, iOS 17, and visionOS 1.

Only macOS has been tested. iOS and visionOS are declared but not yet run. CI builds and runs the tests on a macOS runner. It does not download weights, and it does not build an iOS or visionOS destination.

Shap-E and TripoSR weight files stay outside git. A full text-to-mesh or image-to-mesh run needs those files and an Apple Silicon GPU.

## License

MIT, Copyright (c) 2026 Ismael Alex Otero. See [LICENSE](LICENSE).

Upstream Shap-E is MIT, Copyright (c) 2023 OpenAI. Upstream TripoSR is MIT, Copyright (c) 2024 Tripo AI & Stability AI. Attribution for both, and for MLX, is in [NOTICE](NOTICE).
