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
- Unit tests for marching cubes, mesh export, CLI failure exits, and the
  example package wiring. Real-weight tests skip when the files are absent.
- GitHub Actions workflow that builds, tests, and builds the example on macOS.
- `NOTICE`, `.spi.yml`, and this changelog.

### Notes

- `shap-e-mlx-swift` and `triposr-mlx-swift` are pinned to branches until
  `0.1.0` is tagged. `mlx-swift` is constrained to `0.31.3..<0.31.5`.
