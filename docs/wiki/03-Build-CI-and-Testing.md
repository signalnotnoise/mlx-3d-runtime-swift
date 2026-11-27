# Build, CI, and testing

## Commands

Do not use `swift build` or `swift test`. Metal's `default.metallib` comes from an Xcode build. See [AGENTS.md](../../AGENTS.md).

CI runs on `macos-26` (`.github/workflows/ci.yml:18`) and selects `Xcode_26.6` when that app is present (`.github/workflows/ci.yml:26-31`). It installs the Metal toolchain if `xcodebuild -showComponent MetalToolchain` fails (`.github/workflows/ci.yml:40-47`).

The job deletes the SwiftPM and DerivedData caches, then runs `xcodebuild -resolvePackageDependencies` into an empty directory (`.github/workflows/ci.yml:49-96`). It refuses to continue if `git url.insteadOf` is set, and it checks that `shap-e-mlx-swift`, `triposr-mlx-swift`, and `mlx-swift` resolved from their GitHub URLs, with the siblings on branch `main`. `Package.swift` lists those URLs (`Package.swift:31-41`).

Test command in CI (`.github/workflows/ci.yml:104-112`):

```bash
xcodebuild test \
  -scheme MLX3DRuntime-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

The example resolves into a second empty checkout directory and then builds (`.github/workflows/ci.yml:114-147`), from `Examples/Integration`:

```bash
xcodebuild build \
  -scheme MLX3DExample \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  CODE_SIGNING_ALLOWED=NO
```

`MLX3DRuntimeTests` depends on target `MLX3DCLI` (`Package.swift:58-66`), so the test scheme builds the CLI. A direct CLI build uses `-scheme MLX3DCLI` as in [AGENTS.md](../../AGENTS.md). That scheme name was not a separate CI step in `.github/workflows/ci.yml`.

`scripts/verify_docs.py` checks doc anchors. CI runs it after the example build.

A second job, `ios-simulator`, runs on the same `macos-26` runner (`.github/workflows/ci.yml:153`). It resolves from GitHub into an empty cache, picks an available iPhone simulator, and runs `xcodebuild test -destination 'platform=iOS Simulator,...'` (`.github/workflows/ci.yml:227-239`) for `uiImageOrientationMapsToCGImagePropertyOrientation`, `uiImageRemoveBackgroundNormalizesEveryOrientation`, `sessionUIImageCutoutPathFeedsUprightPixels`, and `sessionUIImageSkippedAndFallbackPathsFeedUprightPixels`. The rest of the suite stays on the macOS job because MLX startup crashes the iOS simulator test runner. The macOS job does not compile UIKit.

## What the tests cover

| Suite | File | Covers |
| --- | --- | --- |
| `MeshTests` | `Tests/MLX3DRuntimeTests/MeshTests.swift:6-47` | Open triangle, closed tetrahedron, degenerate face |
| `MeshExportTests` | `Tests/MLX3DRuntimeTests/MeshExportTests.swift:15-613` | OBJ+MTL references, unchanged PLY, legacy vertex-RGB OBJ, USDZ round-trip, Y-up pose and winding |
| `MarchingCubesTests` | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:7-102` | Grid corners, empty fields, closed ellipsoid at resolution 32 (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:70-93`) |
| `VolumeGridBatchTests` | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:104-175` | Rejects grids outside 2...512 before weight load, and checks lazy batches match the full point list |
| `ImageTensorTests` | `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:9-80` | Missing file, opaque red, transparent pixel over 0.5 |
| `CLIExitTests` | `Tests/MLX3DRuntimeTests/CLIExitTests.swift:6-83` | Nonzero exits, including `--grid-size 9999`; locates `MLX3DCLI` via `dladdr` (`Tests/MLX3DRuntimeTests/CLIExitTests.swift:85-106`). macOS only; `Process` is unavailable on iOS. |
| `RuntimeContractTests` | `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:9-38` | Missing weights, missing bundle, short field |
| `ModelCacheTests` | `Tests/MLX3DRuntimeTests/ModelCacheTests.swift:8-232` | Fake loaders: reuse, single-flight load, failed load is retried, `unload` does not republish an in-flight model |
| `SessionLoadTests` | `Tests/MLX3DRuntimeTests/ModelCacheTests.swift:234-272` | Missing bundle and weights throw on `MLX3DSession.load` and are not kept as a successful load |
| `PublishedWeightTests` | `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:40-65` | Skipped unless weights or a Shap-E bundle are present |
| `SessionImageCacheTests` | `Tests/MLX3DRuntimeTests/ModelCacheTests.swift:276-369` | Session image paths: missing Shap-E bundle, strict and fallback cutout policy, and a second call reuses the loaded slot |
| `ImageBackgroundRemovalTests` | `Tests/MLX3DRuntimeTests/ImageBackgroundRemovalTests.swift:13-424` | Mask bounds, square canvas ratio 0.85, TripoSR skip flag, strict versus fallback cutout policy, Shap-E white-prep skip, a macOS Vision cutout when one is produced, the eight UIImage orientations on iOS, and the session UIImage orientation mapping |

`PublishedWeightTests` only calls `loadWeights` or `ShapERuntime` init. It does not generate a mesh (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:47-64`).

The green macOS run on commit `4d5c73ee1ad4581ede2e87fae6678b4ca51abf69` passed these suites and skipped the two weight tests. That run URL is `https://github.com/signalnotnoise/mlx-3d-runtime-swift/actions/runs/45852915411`.
