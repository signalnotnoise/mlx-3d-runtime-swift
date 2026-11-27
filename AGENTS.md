# AGENTS.md

Router for `mlx-3d-runtime-swift`. Read `docs/knowledge-graph.json` before a broad search.

## Identity

Products: library `MLX3DRuntime`, executable `MLX3DCLI`, and the separate example package `Examples/Integration` (`MLX3DExample`).

Platforms in `Package.swift:13-16`: macOS 14, iOS 17, visionOS 1. Only the macOS `xcodebuild` path has been run. iOS and visionOS are declared and have not been run.

`swift-tools-version: 6.2`. `mlx-swift` is `"0.31.3"..<"0.31.5"` (`Package.swift:31`).

Public entry points:

- `MLX3DSession` at `Sources/MLX3DRuntime/MLX3DSession.swift:38-345`. The caller creates and holds it. Loaded Shap-E and TripoSR models stay in that session until `unload` or `evict`.
- `textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:) async throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:172-184`. Builds a throwaway `MLX3DSession`, so a second static call loads again. The session method checks the grid first (`Sources/MLX3DRuntime/MLX3DSession.swift:119`), then reuses the cached `ShapERuntime`.
- `imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:) async throws -> [Mesh]` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:260-265`. Shap-E image generation only. Default `applyWhiteBackgroundRemoval` runs `removeBackground` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:270-273`). This method is not used by TripoSR. The static entry constructs a new runtime on every call. `MLX3DSession.imageToMeshShapE` (`Sources/MLX3DRuntime/MLX3DSession.swift:219-232`) uses the same white prep and the cached runtime.
- `imageToMesh(imageURL:weightsURL:options:) throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:193-197`. Grid size is checked before weights load (`Sources/MLX3DRuntime/MLX3DRuntime.swift:198-199`). Decodes a file to `[1, side, side, 3]` float RGB (`Sources/MLX3DRuntime/ImageTensor.swift:20`). Does not run a cutout.
- `imageToMesh(cgImage:weightsURL:options:) async throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:217-221`. Default `applyForegroundCutout` runs `foregroundCutout` at ratio 0.85 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:224-240`), then gray compositing. `cutoutFailurePolicy` defaults to `.strict` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:152`). `MLX3DSession.imageToMesh(cgImage:)` (`Sources/MLX3DRuntime/MLX3DSession.swift:149-161`) does the same cutout, then uses the cached TripoSR model.
- `imageToMesh(image:weightsURL:options:) throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:307-311`. `image` is `[height, width, 3]` or `[1, height, width, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:440-461`). This overload does not resize or cut out. It loads a new `TripoSRModel` on every call.
- Session file and tensor `imageToMesh` overloads at `Sources/MLX3DRuntime/MLX3DSession.swift:129-142` and `Sources/MLX3DRuntime/MLX3DSession.swift:314-322` reuse the loaded TripoSR model. They do not run a cutout. On UIKit, `imageToMesh(uiImage:)` renders an upright bitmap with the explicit orientation switch before the cutout (`Sources/MLX3DRuntime/MLX3DSession.swift:164-212`).

`Mesh` packs positions as `x,y,z`, faces as triangle indices, and optional vertex colors as `r,g,b` (`Sources/MLX3DRuntime/Mesh.swift:5-10`).

TripoSR image defaults live on `ImageToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:142-153`): grid 128, threshold 25, query batch 8192, `radius` nil (then `model.renderer.radius`), image side 512, background 0.5, `maximumGridSize` 512, foreground cutout on, ratio 0.85, `cutoutFailurePolicy` `.strict`. Shap-E image defaults live on `ShapEImageToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:86-98`): guidance 3, 32 steps, grid 64, seed 0, white-background removal on. The hard cap is `VolumeGrid.maximumResolution` (`Sources/MLX3DRuntime/VolumeGrid.swift:9`). Density positions are built one batch at a time (`Sources/MLX3DRuntime/TripoSRMeshing.swift:62-67`). The numeric renderer radius is not hardcoded in this repo.

## Out of scope

- Model internals of Shap-E and TripoSR. Those symbols are external. They live in `shap-e-mlx-swift` and `triposr-mlx-swift`.
- Training, fine-tuning, and Python parity or weight-conversion scripts.
- iOS and visionOS execution. The platforms are declared only.
- Downloads, on-disk weight locations, and preload or unload policy. `MLX3DSession` only exposes `load`, `unload`, and `evict`. The caller decides when to use them.

## Documentation index

- [docs/engineering_standards.md](docs/engineering_standards.md)
- [docs/knowledge-graph.json](docs/knowledge-graph.json) — read this before a broad grep
- [docs/knowledge-graph.md](docs/knowledge-graph.md)
- [docs/wiki/01-Architecture-and-Dataflow.md](docs/wiki/01-Architecture-and-Dataflow.md)
- [docs/wiki/02-Dependencies-and-Weights.md](docs/wiki/02-Dependencies-and-Weights.md)
- [docs/wiki/03-Build-CI-and-Testing.md](docs/wiki/03-Build-CI-and-Testing.md)
- [docs/wiki/04-Memory-and-Performance.md](docs/wiki/04-Memory-and-Performance.md)

## Build and test

Do not use `swift build` or `swift test`. `mlx-swift` needs Xcode to compile Metal shaders and bundle `default.metallib`, including when a test sets the MLX device to CPU.

From the repo root:

```bash
xcodebuild build \
  -scheme MLX3DCLI \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO

xcodebuild test \
  -scheme MLX3DRuntime-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

CI also passes `-skipMacroValidation` (`.github/workflows/ci.yml:104-112`). The test target depends on `MLX3DCLI` (`Package.swift:58-66`), so `MLX3DRuntime-Package` builds the CLI. Optional weight tests read `TRIPOSR_WEIGHTS_PATH` and `SHAP_E_BUNDLE`, and also the `TEST_RUNNER_` prefixed copies (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-75`). Prefix those variables with `TEST_RUNNER_` when invoking `xcodebuild`.

Example package (directory name of this checkout must be `mlx-3d-runtime-swift`):

```bash
cd Examples/Integration
xcodebuild build \
  -scheme MLX3DExample \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO
```

CI resolves the public sibling `main` branches from GitHub after deleting the SwiftPM cache (`.github/workflows/ci.yml:49-96`). There is no `Vendor/` mirror.

## Zero-drift

A commit that adds, moves, renames, or changes files, public APIs, or `Package.swift` constraints updates `docs/knowledge-graph.json`, `docs/knowledge-graph.md`, and the affected `docs/wiki/*.md` in the same commit. `scripts/verify_docs.py` hashes the cited lines in the working tree. `metadata.verified_commit` is a label only. It does not have to be an ancestor of `HEAD`.
