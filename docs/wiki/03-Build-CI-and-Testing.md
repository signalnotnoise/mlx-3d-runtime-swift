# Build, CI, and testing

## Commands

Do not use `swift build` or `swift test`. Metal's `default.metallib` comes from an Xcode build. See [AGENTS.md](../../AGENTS.md).

CI runs on `macos-26` (`.github/workflows/ci.yml:18`) and selects `Xcode_26.6` when that app is present (`.github/workflows/ci.yml:26-31`). It installs the Metal toolchain if `xcodebuild -showComponent MetalToolchain` fails (`.github/workflows/ci.yml:41-47`).

Package resolution cannot clone the private sibling URLs with the Actions token. The job turns `Vendor/` into bare git mirrors and rewrites those URLs (`.github/workflows/ci.yml:49-72`). `Package.swift` itself still lists the GitHub URLs (`Package.swift:34-42`).

Test command in CI (`.github/workflows/ci.yml:116-127`):

```bash
xcodebuild test \
  -scheme MLX3DRuntime-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -skipMacroValidation \
  -parallel-testing-enabled NO \
  CODE_SIGNING_ALLOWED=NO
```

Example command in CI (`.github/workflows/ci.yml:129-141`), from `Examples/Integration`:

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

## What the tests cover

| Suite | File | Covers |
| --- | --- | --- |
| `MeshTests` | `Tests/MLX3DRuntimeTests/MeshTests.swift:6-47` | Open triangle, closed tetrahedron, degenerate face |
| `MeshExportTests` | `Tests/MLX3DRuntimeTests/MeshExportTests.swift:6-98` | OBJ, PLY, empty mesh, bad indices |
| `MarchingCubesTests` | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:7-102` | Grid corners, empty fields, closed ellipsoid at resolution 32 (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:70-93`) |
| `VolumeGridBatchTests` | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:104-175` | Rejects grids outside 2...512 before weight load, and checks lazy batches match the full point list |
| `ImageTensorTests` | `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:9-80` | Missing file, opaque red, transparent pixel over 0.5 |
| `CLIExitTests` | `Tests/MLX3DRuntimeTests/CLIExitTests.swift:5-82` | Nonzero exits, including `--grid-size 9999`; locates `MLX3DCLI` via `dladdr` (`Tests/MLX3DRuntimeTests/CLIExitTests.swift:84-105`) |
| `RuntimeContractTests` | `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:9-38` | Missing weights, missing bundle, short field |
| `PublishedWeightTests` | `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:40-65` | Skipped unless weights or a Shap-E bundle are present |
| `ImageBackgroundRemovalTests` | `Tests/MLX3DRuntimeTests/ImageBackgroundRemovalTests.swift:7-210` | Mask bounds, square canvas ratio 0.85, TripoSR skip flag, strict versus fallback cutout policy, Shap-E white-prep skip, and a macOS Vision cutout when one is produced |

`PublishedWeightTests` only calls `loadWeights` or `ShapERuntime` init. It does not generate a mesh (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:47-64`).

The green macOS run on commit `4d5c73ee1ad4581ede2e87fae6678b4ca51abf69` passed these suites and skipped the two weight tests. That run URL is `https://github.com/signalnotnoise/mlx-3d-runtime-swift/actions/runs/45852915411`.
