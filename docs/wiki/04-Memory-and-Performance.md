# Memory and performance

Numbers below are either code defaults or sibling README measurements. This wrapper has no end-to-end timing of its own.

## What this repo sets

`ImageToMeshOptions` defaults are grid 128, threshold 25, and `queryBatchSize` 8192 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:144-147`). `TextToMeshOptions` defaults include `queryBatchSize` 65536 and `gridSize` 128 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:26-29`). `ShapEImageToMeshOptions` uses guidance 3, 32 steps, and grid 64 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:88-90`), with query batch 4096 on macOS and 1024 elsewhere (`Sources/MLX3DRuntime/MLX3DRuntime.swift:114-120`). Both options carry `maximumGridSize`, which cannot exceed 512 (`Sources/MLX3DRuntime/VolumeGrid.swift:9-23`).

Density queries build only the current batch of positions (`Sources/MLX3DRuntime/TripoSRMeshing.swift:59-68`) and still store the full density field, because marching cubes needs every sample (`Sources/MLX3DRuntime/TripoSRMeshing.swift:56-58`). Color queries loop mesh vertices (`Sources/MLX3DRuntime/TripoSRMeshing.swift:90-96`). Each chunk is `eval`'d before the next one.

A search of `Sources/` finds no `Memory.cacheLimit` and no `Device.setDefault` on the library or CLI path. Tests set `Device(.cpu)` (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:8-10`, `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:10-12`, `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:48`).

`MLX3DSession` keeps loaded Shap-E and TripoSR objects in process memory until the caller `unload`s or `evict`s them (`Sources/MLX3DRuntime/MLX3DSession.swift:59-100`). There is no global cache. `MLX3DRuntime.textToMesh` allocates a session for that call and drops it (`Sources/MLX3DRuntime/MLX3DRuntime.swift:178-183`). The static image methods construct a new `TripoSRModel` per call (`Sources/MLX3DRuntime/MLX3DRuntime.swift:318-323`). The session does not write a weight cache to disk.

This package has not been timed on device. There is no measured wall clock, resident set, or `Memory.peakMemory` for `textToMesh` or `imageToMesh` in this repo.

## Sibling measurements (not this binary)

Recorded in public `triposr-mlx-swift` at `51994056febf3be5ed726e89de169037a2fbf6a6`, a harness outside that package, on an Apple M3 Max:

- `TripoSRCLI` smoke with `Memory.cacheLimit` 512 MiB: 0.75 seconds real, maximum resident set size 1,733,754,880 bytes, peak memory footprint 2,231,355,360 bytes. Printed triplanes `[1, 3, 64, 64, 40]`.
- Chair run phase table: density query of a 128³ grid in batches of 8192 took 2.375 seconds; marching cubes 0.058 seconds; vertex colors 0.016 seconds. The same section says marching cubes ran outside that package.
- That process: 5.26 seconds real, maximum resident set size 1,773,879,296 bytes, peak memory footprint 3,503,343,872 bytes, MLX `Memory.peakMemory` 2,629,506,977 bytes.

Recorded in public `shap-e-mlx-swift` at `d7b6b7aa88d11e265799fc81e5fea1b734571d2b`, a sibling CLI run on an Apple M3 Max: 39.42 seconds real, peak memory footprint 5,489,169,280 bytes, maximum resident set size 5,856,296,960 bytes, 9,882 vertices, 19,684 faces, closed, 0 boundary edges. Metal marching cubes 0.052 seconds. CPU marching cubes 0.816 seconds. That is not a run of `MLX3DCLI`.

The ellipsoid unit test in this repo uses a 32³ synthetic field and checks closure (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:51-67`). It does not record time or memory.
