# MLX3DRuntime

Swift package for text-to-3D and image-to-3D on Apple Silicon. Text prompts go through [shap-e-mlx-swift](https://github.com/signalnotnoise/shap-e-mlx-swift) (`ShapERuntime`). Images go through [triposr-mlx-swift](https://github.com/signalnotnoise/triposr-mlx-swift) (`TripoSRRuntime`). Both paths return one `Mesh`. Export writes OBJ+MTL, PLY, or USDZ.

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
try MeshExport.writeUSDZ(
    mesh,
    to: URL(fileURLWithPath: "chair.usdz"),
    options: MeshExportOptions(upAxis: .yUp)
)
```

`writeOBJ` also writes `chair.mtl` beside the OBJ. The MTL `Kd` is the average vertex color. `upAxis` defaults to `.native`, so this sample's OBJ and PLY keep the mesh's Z-up coordinates. `.yUp` is opt-in and applies only to the file being written.

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

`textToMesh` still returns one `Mesh`. `textToMeshBatch` takes a `batchSize` and returns one `Mesh` per row. `useMetalMarchingCubes` on text defaults to false, and `progressInterval` defaults to 0. Pass a `progress` closure to receive Shap-E phase and step events. Partial meshes are included only when `progressInterval` is greater than zero. A nil closure does not ask Shap-E to build that callback.

TripoSR `imageToMesh` takes the same kind of closure. It reports `encoding`, `volume`, `meshing`, and `colors`, and the partial-mesh field stays nil. `ImageToMeshOptions.gridSize` defaults to 128. `useMetalMarchingCubes` defaults to false. `resizeMode` defaults to `.aspectFit`; `.stretch` fills the square the way Mesquette's previous TripoSR facade did. The tensor overload does not resize.

Shap-E image generation already defaults Metal marching cubes to true and `progressInterval` to 0. Its `progress` closure is the same event type. Labels, percentages, and preview drawing stay with the caller.

`session.imageToMesh(cgImage:)` runs the same foreground cutout as `MLX3DRuntime.imageToMesh(cgImage:)`, including `cutoutFailurePolicy` (`.strict` by default), then reuses the cached TripoSR model. `session.imageToMeshShapE` runs the same white-background `removeBackground` prep, then reuses the cached Shap-E runtime. On UIKit, `imageToMesh(uiImage:)` uses that same switch to render an upright bitmap before the cutout. A skipped cutout and a fallback both send those upright pixels to TripoSR.

`Mesh.positions` and `Mesh.vertexColors` are packed `x, y, z` and `r, g, b`. `Mesh.faces` are packed triangle indices. `vertexColors` is `nil` when a path has no colors. Shap-E's RGBA channel is stored as RGB.

Shap-E and TripoSR meshes are Z-up: the body runs from the legs to the back along Z, and the back is toward +Z. Export does not change that unless you pass `upAxis: .yUp`. That option rotates `(x, y, z)` to `(x, z, -y)` while writing OBJ, PLY, or USDZ. It is a right-handed turn, so winding stays the same. The `Mesh` value itself is not modified. There is no implicit axis conversion.

OBJ colors follow the Mesquette export invariant. `writeOBJ` writes positions only and a companion `.mtl` whose `Kd` is the average albedo (`0.75 0.75 0.75` when the mesh has no colors). Per-vertex color is kept in USDZ and in PLY. The older `v x y z r g b` OBJ is still available as `MeshExport.objText` or `MeshExportOptions(objColors: .vertexRGB)`. That option does not write an MTL, because the extra `v` components are not a standard OBJ file. The CLI writes the default OBJ+MTL (and PLY) in native coordinates. It does not write USDZ: Model I/O does not register that exporter in a command-line process. An app calls `MeshExport.writeUSDZ`. `.yUp` is a library option on every format the library writes. Share sheets and scene placement stay in the app. Package tests skip USDZ for the same process limitation. `Examples/ExportHost` is the app-hosted check, and CI runs it on macOS and on the iOS Simulator.

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

public enum MeshExportUpAxis: Sendable, Equatable { case native, yUp }
public enum MeshOBJColors: Sendable, Equatable { case materialAverage, vertexRGB }
public struct MeshExportOptions: Sendable, Equatable {
    public var upAxis: MeshExportUpAxis       // default .native
    public var objColors: MeshOBJColors       // default .materialAverage
}

public enum MeshExport {
    public static let materialName: String
    public static func writeOBJ(_ mesh: Mesh, to url: URL, comment: String? = nil, options: MeshExportOptions = .init()) throws
    public static func writePLY(_ mesh: Mesh, to url: URL, options: MeshExportOptions = .init()) throws
    public static func writeUSDZ(_ mesh: Mesh, to url: URL, options: MeshExportOptions = .init()) throws
    public static func write(_ mesh: Mesh, to url: URL, options: MeshExportOptions = .init()) throws
    public static func materialURL(forOBJ url: URL) -> URL
    public static func objText(_ mesh: Mesh, comment: String? = nil, options: MeshExportOptions = .init()) throws -> String
    public static func wavefrontText(_ mesh: Mesh, mtlFileName: String, objectName: String, comment: String? = nil, options: MeshExportOptions = .init()) throws -> String
    public static func materialText(for mesh: Mesh) throws -> String
}

public struct MLX3DGenerationEvent: Sendable, Equatable {
    public var phase: String
    public var step: Int
    public var totalSteps: Int
    public var partialMeshes: [Mesh]?
}

public typealias MLX3DProgressHandler = @Sendable (MLX3DGenerationEvent) -> Void

public enum ImageResizeMode { case aspectFit, stretch }

public struct TextToMeshOptions: Sendable { /* guidance 15, 64 steps, grid 128, batch 65536, seed 0, CPU marching cubes, progressInterval 0 */ }
public struct ImageToMeshOptions: Sendable { /* grid 128, threshold 25, batch 8192, radius nil, side 512, background 0.5, CPU marching cubes, aspectFit, foreground cutout on at ratio 0.85, cutoutFailurePolicy .strict */ }
public struct ShapEImageToMeshOptions: Sendable { /* guidance 3, 32 steps, grid 64, seed 0, Metal marching cubes, progressInterval 0, white-background removal on */ }

public enum MLX3DRuntime {
    public static func textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> Mesh
    public static func textToMeshBatch(prompt:modelBundleURL:bpeVocabularyURL:batchSize:options:progress:) async throws -> [Mesh]
    public static func imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> [Mesh]
    public static func imageToMesh(imageURL:weightsURL:options:progress:) throws -> Mesh
    public static func imageToMesh(cgImage:weightsURL:options:progress:) async throws -> Mesh
    public static func imageToMesh(image:weightsURL:options:progress:) throws -> Mesh
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
    public func textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> Mesh
    public func textToMeshBatch(prompt:modelBundleURL:bpeVocabularyURL:batchSize:options:progress:) async throws -> [Mesh]
    public func imageToMesh(imageURL:weightsURL:options:progress:) async throws -> Mesh
    public func imageToMesh(cgImage:weightsURL:options:progress:) async throws -> Mesh
    public func imageToMesh(image:weightsURL:options:progress:) async throws -> Mesh
    public func imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> [Mesh]
}

public enum ImageTensor {
    public static func rgb(contentsOf:side:background:resizeMode:) throws -> MLXArray
    public static func rgb(cgImage:side:background:resizeMode:) throws -> MLXArray
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

Unit tests cover mesh topology, OBJ+MTL and PLY export, a Y-up conversion that keeps winding, a synthetic ellipsoid through marching cubes (closed, 0 boundary edges), image decoding, aspect-fit versus stretch, CLI exits when the bundle or weights are missing, the loaded-model cache, and generation progress. USDZ round-trip runs in the app host, not in this package process. A missing bundle or weights file still throws, and the progress closure is not called. A test seam checks that phase events arrive in order and that a nil closure does not build a Shap-E callback. Cache tests inject a fake loader. They do not read weight files. They check reuse, one shared in-flight load, a failed load that is retried, and `unload` discarding a load that was already running. Loading real weights is optional: those tests skip when `TRIPOSR_WEIGHTS_PATH` / `triposr_swift_ready.safetensors` or `SHAP_E_BUNDLE` is absent. `xcodebuild` only forwards variables prefixed with `TEST_RUNNER_` into the test process.

## Scope

Platforms in `Package.swift` are macOS 14, iOS 17, and visionOS 1.

Only macOS has been tested. iOS and visionOS are declared but not yet run. CI builds and runs the tests on a macOS runner. It does not download weights, and it does not build an iOS or visionOS destination.

Shap-E and TripoSR weight files stay outside git. A full text-to-mesh or image-to-mesh run needs those files and an Apple Silicon GPU.

## License

MIT, Copyright (c) 2026 Ismael Alex Otero. See [LICENSE](LICENSE).

Upstream Shap-E is MIT, Copyright (c) 2023 OpenAI. Upstream TripoSR is MIT, Copyright (c) 2024 Tripo AI & Stability AI. The OBJ+MTL and USDZ writer is adapted from Mesquette, MIT, Copyright (c) 2026 signalnotnoise. Attribution is in [NOTICE](NOTICE).
