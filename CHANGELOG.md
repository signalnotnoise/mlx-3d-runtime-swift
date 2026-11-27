# Changelog

All notable changes to this package are documented in this file.

The format is based on Keep a Changelog, and this package does not have a
version tag yet.

## [Unreleased]

### Added

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

### Fixed

- `removeBackground(from: UIImage)` maps `UIImage.Orientation` onto
  `CGImagePropertyOrientation` with an explicit switch. The two enums do not
  share raw values, so the previous cast dropped upright images and
  mis-labeled the rest. `uiImageOrientationMapsToCGImagePropertyOrientation`
  and `uiImageRemoveBackgroundNormalizesEveryOrientation` cover all eight
  orientations and run on the iOS Simulator CI job.

### Notes

- `shap-e-mlx-swift` and `triposr-mlx-swift` are public and pinned to
  branch `main` until `0.1.0` is tagged. `mlx-swift` is constrained to
  `0.31.3..<0.31.5`.
- CI resolves those URLs from GitHub with an empty SwiftPM cache. There is
  no `Vendor/` mirror.
