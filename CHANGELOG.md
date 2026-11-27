# Changelog

All notable changes to this package are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Changed

- `Package.swift` pins `shap-e-mlx-swift` and `triposr-mlx-swift` to
  `from: "0.1.0"` instead of branch `main`. `Package.resolved` records
  shap-e `962403e` (the `v0.1.0` tag, which includes the `7ed124d`
  tensor-byte mesh accessors) and triposr `5199405`. CI checks those
  version pins after resolving from GitHub.

### Added

- Optional `progress` on `textToMesh`, `imageToMesh`, and `imageToMeshShapE`,
  on both `MLX3DRuntime` and `MLX3DSession`. The closure receives
  `MLX3DGenerationEvent`: the runtime phase name, step, total, and optional
  partial meshes. Shap-E partial meshes are forwarded only when
  `progressInterval` is greater than zero. A nil closure is passed through as
  a nil Shap-E callback, so those extracts do not run. TripoSR reports
  `encoding`, `volume`, `meshing`, and `colors` and leaves partial meshes nil.
  Labels, percentages, and preview drawing stay with the caller.
- `textToMeshBatch`, which takes a batch count and returns `[Mesh]`.
  `textToMesh` still returns one `Mesh`. `useMetalMarchingCubes` on text
  defaults to false. `progressInterval` on text defaults to 0.
- TripoSR `useMetalMarchingCubes` (default false, the CPU extractor) and
  `resizeMode` (default `.aspectFit`; `.stretch` matches the previous
  Mesquette facade). `gridSize` stays caller-set and still defaults to 128.
- `Examples/ExportHost`, a small macOS and iOS app whose hosted XCTest
  target writes and reads USDZ. Model I/O does not register the USD exporter
  in the SwiftPM test process or in `MLX3DCLI`, so the package tests skip
  USDZ and the CLI does not offer it.
- OBJ+MTL and USDZ export, adapted from Mesquette's `MeshExporter`.
  `writeOBJ` writes positions-only `v` lines, `mtllib`, `usemtl VertexColorAverage`,
  and a companion `.mtl` whose `Kd` is the average albedo (gray `0.75` when the
  mesh has no colors). USDZ keeps per-vertex RGB through Model I/O. PLY is unchanged.
  `MeshExportOptions.upAxis` defaults to `.native`. `.yUp` rotates Z-up meshes to
  Y-up at export only: `(x, y, z)` becomes `(x, z, -y)`. Shap-E and TripoSR stay
  Z-up in memory, legs to back along Z, back toward +Z. The historical
  `v x y z r g b` OBJ remains `objText` and `objColors: .vertexRGB` (no MTL).
- `MLX3DSession`, a caller-owned in-memory cache of loaded Shap-E and TripoSR
  models. The key is the backend, the bundle URL, and any other file the
  loader reads (the Shap-E BPE vocabulary). A second request reuses the loaded
  model. Concurrent requests for that key share one load. A failed load is not
  cached. `unload` bumps a generation token so an in-flight load cannot
  re-enter the cache; its waiters throw `MLX3DRuntimeError.modelUnloaded`.
  `evict` drops a ready model and leaves an in-flight load alone. The session
  does not download weights or choose a preload policy. Static `textToMesh`
  uses a throwaway session. Static `imageToMesh` still loads on every call.
- `MLX3DSession.imageToMesh(cgImage:)` runs the foreground cutout with
  `CutoutFailurePolicy` (`.strict` by default) and then the cached TripoSR
  model. `MLX3DSession.imageToMeshShapE` runs the white-background prep and
  then the cached Shap-E runtime. On UIKit, `imageToMesh(uiImage:)` renders an
  upright bitmap with the explicit `UIImage.Orientation` switch before the
  cutout, so a skipped cutout or a fallback does not send a sideways photo.
- Swift package `MLX3DRuntime` with library product `MLX3DRuntime` and
  executable product `MLX3DCLI`.
- `Mesh` (positions, faces, optional vertex colors) plus OBJ and PLY export.
- Text-to-mesh through `ShapERuntime`, and image-to-mesh through
  `TripoSRRuntime`.
- TripoSR mesh extraction: density on a 128³ grid in batches, marching cubes
  at threshold 25 via Shap-E's public `MarchingCubes`, and vertex colors from
  TripoSR's color query.
- `Examples/Integration`, a separate package that depends on this repo by
  relative path and imports `MLX3DRuntime`, `ShapERuntime`, and `TripoSRRuntime`.
- `imageToMesh(cgImage:)` runs Mesquette's `foregroundCutout` (ratio 0.85)
  before TripoSR preprocessing. Pass `applyForegroundCutout: false` to skip it.
  `cutoutFailurePolicy` defaults to `.strict`. `.fallbackToOriginal` logs and
  uses the original image, matching Mesquette. The file-URL and CLI image
  paths do not cut out.
- `imageToMeshShapE(cgImage:)` runs the white-background `removeBackground`
  prep, then Shap-E image generation. The prep is skippable and is not used
  by TripoSR. Defaults match Mesquette image mode: guidance 3, 32 steps, grid 64.
- Unit tests for marching cubes, mesh export, CLI failure exits, foreground
  cutout helpers, and the example package wiring. Real-weight tests skip when
  the files are absent.
- GitHub Actions workflow that builds, tests, and builds the example on macOS.
- `NOTICE`, `.spi.yml`, and this changelog.

### Changed

- `MeshExport.writeOBJ` and `MeshExport.write` for a `.obj` URL no longer emit
  vertex RGB on `v` lines. They write the Mesquette OBJ+MTL pair instead.
  Pass `MeshExportOptions(objColors: .vertexRGB)` for the previous file.

### Fixed

- USDZ export no longer depends on `MDLAsset.canExportFileExtension("usdz")`.
  That check is false even inside an app, including the Xcode 26.6 GitHub
  runner. The file is an uncompressed, 64-byte-aligned zip of a Model I/O
  `usdc`/`usda` crate, a SceneKit USDZ package, or an ASCII USDA layer.
- Shap-E progress partials no longer crash the process when a sampler step
  extracts an empty or non-contiguous mesh. `mesh(from:)` validates the
  vertex, face, and color `MLXArray`s (`size > 0`, rank, last dimension 3 or
  4, dtype) and calls `asType` and `contiguous()` before `asArray`. A partial
  that fails that check is omitted, so `partialMeshes` may be empty. The
  finished mesh still throws `emptyMesh` when extraction produces no
  triangles. `Package.resolved` pins `shap-e-mlx-swift` at `7ed124d`, where
  the scalar accessors read tensor bytes instead of `MLXArray.asArray`.
  Int64 face indices that do not fit in `Int32` throw `emptyMesh` instead of
  truncating.
- `removeBackground(from: UIImage)` maps `UIImage.Orientation` onto
  `CGImagePropertyOrientation` with an explicit switch. The two enums do not
  share raw values, so the previous cast dropped upright images and
  mis-labeled the rest. `uiImageOrientationMapsToCGImagePropertyOrientation`
  and `uiImageRemoveBackgroundNormalizesEveryOrientation` cover all eight
  orientations and run on the iOS Simulator CI job. CLI process
  tests stay on macOS because `Foundation.Process` is unavailable on iOS.

### Notes

- `shap-e-mlx-swift` and `triposr-mlx-swift` are public and pinned to
  `from: "0.1.0"`. `mlx-swift` is constrained to `0.31.3..<0.31.5`.
- CI resolves those URLs from GitHub with an empty SwiftPM cache. There is
  no `Vendor/` mirror.
