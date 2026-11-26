# Memory and performance

Numbers below are either code defaults or sibling README measurements. This wrapper has no end-to-end timing of its own.

## What this repo sets

`ImageToMeshOptions` defaults are grid 128, threshold 25, and `queryBatchSize` 8192 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:38-42`). `TextToMeshOptions` defaults include `queryBatchSize` 65536 and `gridSize` 128 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:14-18`).

Density and color queries loop `min(batchSize, remaining)` points (`Sources/MLX3DRuntime/TripoSRMeshing.swift:59-71` and `Sources/MLX3DRuntime/TripoSRMeshing.swift:85-97`). Each chunk is `eval`'d before the next one.

A search of `Sources/` finds no `Memory.cacheLimit` and no `Device.setDefault` on the library or CLI path. Tests set `Device(.cpu)` (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:8-10`, `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:10-12`, `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:48`).

This package has not been timed on device. There is no measured wall clock, resident set, or `Memory.peakMemory` for `textToMesh` or `imageToMesh` in this repo.

## Sibling measurements (not this binary)

Copied from `Vendor/triposr-mlx-swift/README.md`, which records a harness outside that package, on an Apple M3 Max:

- `TripoSRCLI` smoke with `Memory.cacheLimit` 512 MiB: 0.75 seconds real, maximum resident set size 1,733,754,880 bytes, peak memory footprint 2,231,355,360 bytes. Printed triplanes `[1, 3, 64, 64, 40]` (`Vendor/triposr-mlx-swift/README.md:123`).
- Chair run phase table: density query of a 128³ grid in batches of 8192 took 2.375 seconds; marching cubes 0.058 seconds; vertex colors 0.016 seconds (`Vendor/triposr-mlx-swift/README.md:132-134`). The same section says marching cubes ran outside that package.
- That process: 5.26 seconds real, maximum resident set size 1,773,879,296 bytes, peak memory footprint 3,503,343,872 bytes, MLX `Memory.peakMemory` 2,629,506,977 bytes (`Vendor/triposr-mlx-swift/README.md:136`).

Copied from `Vendor/shap-e-mlx-swift/README.md:109`, a sibling CLI run on an Apple M3 Max: 39.42 seconds real, peak memory footprint 5,489,169,280 bytes, maximum resident set size 5,856,296,960 bytes, 9,882 vertices, 19,684 faces, closed, 0 boundary edges. Metal marching cubes 0.052 seconds. CPU marching cubes 0.816 seconds. That is not a run of `MLX3DCLI`.

The ellipsoid unit test in this repo uses a 32³ synthetic field and checks closure (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:51-67`). It does not record time or memory.
