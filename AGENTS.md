# AGENTS.md

Router for `mlx-3d-runtime-swift`. Read `docs/knowledge-graph.json` before a broad search.

## Identity

Products: library `MLX3DRuntime`, executable `MLX3DCLI`, and the separate example package `Examples/Integration` (`MLX3DExample`).

Platforms in `Package.swift:13-16`: macOS 14, iOS 17, visionOS 1. Only the macOS `xcodebuild` path has been run. iOS and visionOS are declared and have not been run.

`swift-tools-version: 6.2`. `mlx-swift` is `"0.31.3"..<"0.31.5"` (`Package.swift:31`).

Public entry points:

- `MLX3DSession` at `Sources/MLX3DRuntime/MLX3DSession.swift:38-396`. The caller creates and holds it. Loaded Shap-E and TripoSR models stay in that session until `unload` or `evict`.
- `textToMesh(prompt:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:196-210`. Builds a throwaway `MLX3DSession`, so a second static call loads again. The session method checks the grid first (`Sources/MLX3DRuntime/MLX3DSession.swift:148-149`), then reuses the cached `ShapERuntime` and samples one mesh. `textToMeshBatch(prompt:modelBundleURL:bpeVocabularyURL:batchSize:options:progress:) async throws -> [Mesh]` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:213-229` takes the batch count. `useMetalMarchingCubes` defaults to false. `progressInterval` defaults to 0, which reports steps and does not extract partial meshes.
- `imageToMeshShapE(cgImage:modelBundleURL:bpeVocabularyURL:options:progress:) async throws -> [Mesh]` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:315-350`. Shap-E image generation only. Default `applyWhiteBackgroundRemoval` runs `removeBackground` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:326-332`). This method is not used by TripoSR. The static entry constructs a new runtime on every call. `MLX3DSession.imageToMeshShapE` (`Sources/MLX3DRuntime/MLX3DSession.swift:264-279`) uses the same white prep and the cached runtime. A nil `progress` is passed through as a nil Shap-E callback (`Sources/MLX3DRuntime/MLX3DRuntime.swift:478-498`).
- `imageToMesh(imageURL:weightsURL:options:progress:) throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:241-256`. Grid size is checked before weights load (`Sources/MLX3DRuntime/MLX3DRuntime.swift:247-248`). Decodes a file to `[1, side, side, 3]` float RGB (`Sources/MLX3DRuntime/ImageTensor.swift:20-30`). Does not run a cutout. `resizeMode` defaults to aspect-fit.
- `imageToMesh(cgImage:weightsURL:options:progress:) async throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:268-303`. Default `applyForegroundCutout` runs `foregroundCutout` at ratio 0.85 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:276-292`), then gray compositing. `cutoutFailurePolicy` defaults to `.strict` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:171-171`). `MLX3DSession.imageToMesh(cgImage:)` (`Sources/MLX3DRuntime/MLX3DSession.swift:188-202`) does the same cutout, then uses the cached TripoSR model.
- `imageToMesh(image:weightsURL:options:progress:) throws -> Mesh` at `Sources/MLX3DRuntime/MLX3DRuntime.swift:357-367`. `image` is `[height, width, 3]` or `[1, height, width, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:533-554`). This overload does not resize or cut out. It loads a new `TripoSRModel` on every call.
- Session file and tensor `imageToMesh` overloads at `Sources/MLX3DRuntime/MLX3DSession.swift:166-181` and `Sources/MLX3DRuntime/MLX3DSession.swift:364-373` reuse the loaded TripoSR model. They do not run a cutout. On UIKit, `imageToMesh(uiImage:)` renders an upright bitmap with the explicit orientation switch before the cutout (`Sources/MLX3DRuntime/MLX3DSession.swift:205-257`).
- `MLX3DGenerationEvent` at `Sources/MLX3DRuntime/GenerationProgress.swift:9-26` is the forwarded phase, step, total, and optional Shap-E partial meshes. TripoSR emits `encoding`, `volume`, `meshing`, and `colors` with nil partials (`Sources/MLX3DRuntime/MLX3DRuntime.swift:376-397`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:33-50`).

`Mesh` packs positions as `x,y,z`, faces as triangle indices, and optional vertex colors as `r,g,b` (`Sources/MLX3DRuntime/Mesh.swift:5-10`).

TripoSR image defaults live on `ImageToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:159-172`): grid 128, threshold 25, query batch 8192, `radius` nil (then `model.renderer.radius`), image side 512, background 0.5, `maximumGridSize` 512, CPU marching cubes, aspect-fit resize, foreground cutout on, ratio 0.85, `cutoutFailurePolicy` `.strict`. A caller sets `gridSize` itself; Mesquette's previous facade used 256. Shap-E image defaults live on `ShapEImageToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:96-108`): guidance 3, 32 steps, grid 64, seed 0, Metal marching cubes on, `progressInterval` 0, white-background removal on. The hard cap is `VolumeGrid.maximumResolution` (`Sources/MLX3DRuntime/VolumeGrid.swift:9`). Density positions are built one batch at a time (`Sources/MLX3DRuntime/TripoSRMeshing.swift:72-77`). The numeric renderer radius is not hardcoded in this repo.

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
