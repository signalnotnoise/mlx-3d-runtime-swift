# Engineering standards

Invariants of this repository. Sibling model code is not specified here.

## Tools and mlx-swift

`Package.swift:1` sets `swift-tools-version: 6.2`.

`Package.swift:29-32` caps `mlx-swift` at `0.31.3..<0.31.5`. The comment states that `0.31.5` and later set Swift tools 6.3, and that Shap-E's own manifest allows `0.29.1` and later, so this range is the overlap. The vendored sibling manifests match that description: `Vendor/shap-e-mlx-swift/Package.swift:22` uses `from: "0.29.1"` and `Vendor/triposr-mlx-swift/Package.swift:25-27` uses the same `0.31.3..<0.31.5` cap. Do not widen this package past `0.31.4`.

## Tensor layout and coordinates

Images accepted by `imageToMesh(image:)` are float RGB in `0...1`, rank 3 `[height, width, 3]` or rank 4 `[1, height, width, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:171-186`). Rank 3 is reshaped to `[1, H, W, 3]`. The file decoder returns `[1, side, side, 3]` (`Sources/MLX3DRuntime/ImageTensor.swift:20`). Row 0 is the top of the image (`Sources/MLX3DRuntime/ImageTensor.swift:79-80`). Alpha is composited over `background` (`Sources/MLX3DRuntime/ImageTensor.swift:52-59`). This path does not apply ImageNet normalization; that stays inside the sibling `forwardTriplanes`.

`VolumeGrid.positions` is packed `x, y, z` with X slowest and Z fastest. Sample `i` is `min + i / (resolution - 1) * (max - min)` (`Sources/MLX3DRuntime/VolumeGrid.swift:3-11`). TripoSR meshing uses the cube `[-radius, radius]` on each axis (`Sources/MLX3DRuntime/TripoSRMeshing.swift:23-25`). `radius` is `options.radius` or `model.renderer.radius` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:17`). This repo does not hardcode `0.87`.

Density queries are reshaped to `[count, 3]` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:62`). The flattened density array must have `count` values (`Sources/MLX3DRuntime/TripoSRMeshing.swift:66-68`). Color queries use the same position shape and require `count * 3` floats (`Sources/MLX3DRuntime/TripoSRMeshing.swift:88-93`). This repo does not assert the MLX rank of `evaluate`'s return value beyond that flat count.

Marching cubes subtracts `threshold` and reshapes the field to `[resolution, resolution, resolution]` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:33-37`). Values above the threshold are the interior. The bbox passed to Shap-E is the same min and size as the sample grid (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:38-44`). Positions come back packed `x,y,z`. Faces are `Int32` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:45-53`).

`Mesh.positions` length is `3V`. `Mesh.faces` length is `3F`. `Mesh.vertexColors`, when present, length is `3V` (`Sources/MLX3DRuntime/Mesh.swift:5-10`). Shap-E RGBA (`V * 4`) is stored as RGB, dropping the fourth component (`Sources/MLX3DRuntime/MLX3DRuntime.swift:145-160`).

## Weights

Never commit weights. `.gitignore` ignores `runtime_bundle/`, `triposr_swift_ready.safetensors`, and `*.safetensors`.

The library image path requires the caller to pass a regular file (`Sources/MLX3DRuntime/MLX3DRuntime.swift:163-168`). It does not search the filesystem. `loadWeights(from:)` is the sibling API. This repo does not reimplement shape or unused-key checks.

The library text path requires `manifest.json` in the bundle directory and a BPE file, defaulting to `bpe_simple_vocab_16e6.txt.gz` beside that manifest (`Sources/MLX3DRuntime/MLX3DRuntime.swift:66-74`).

CLI image search order is `--weights`, then `TRIPOSR_WEIGHTS_PATH`, then `./triposr_swift_ready.safetensors`, then `./triposr_weights/triposr_swift_ready.safetensors` (`Sources/MLX3DCLI/MLX3DCLI.swift:114-117` and `Sources/MLX3DCLI/MLX3DCLI.swift:159-166`). It does not search the older monorepo paths that the sibling CLI searches.

CLI text search is `--bundle` or `SHAP_E_BUNDLE` (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`). Optional `--bpe` overrides the bundle default (`Sources/MLX3DCLI/MLX3DCLI.swift:77`).

Tests also accept `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH` and `TEST_RUNNER_SHAP_E_BUNDLE` (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-75`).

Canonical files, cited from the vendored sibling READMEs (sibling measurements, not re-hashed here):

- TripoSR: `triposr_swift_ready.safetensors`, 1,674,793,528 bytes, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`, Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d` (`Vendor/triposr-mlx-swift/README.md:67-70` and `Vendor/triposr-mlx-swift/README.md:121`).
- Shap-E: Hub revision `d71f70a09b9c03535188d006a719edda5c19a6eb`, described as 1,104 files. The sibling README says upstream notes record 5,143,254,750 bytes and that the package does not re-measure that payload (`Vendor/shap-e-mlx-swift/README.md:63`). This repo does not record a single SHA-256 for that bundle.

## Parity

This repo does not ship Python parity dumps or a `--parity` mode. `MLX3DCLI` has `text` and `image` only (`Sources/MLX3DCLI/MLX3DCLI.swift:49-58`).

The image grid matches the TripoSR extraction convention used by this code: density minus threshold, positive inside, no axis swap (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:6-12`). Whether that matches a given upstream commit beyond that comment is not re-checked in this repo.

`DINOv2` interpolation and Shap-E diffusion numerics are sibling behavior. Do not change them from this package.

## Memory and Metal

This repo does not set `Memory.cacheLimit` and does not call `Device.setDefault` on the library path. `MLX.eval` is used after triplane construction, each density batch, each color batch, and marching-cubes outputs (`Sources/MLX3DRuntime/MLX3DRuntime.swift:126-130`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:64-65`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:90-91`, `Sources/MLX3DRuntime/MarchingCubesExtractor.swift:45-46`).

Point queries are chunked by `queryBatchSize` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:59-71`). The image default is 8192 and the text default passed to Shap-E is 65536 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:38-42` and `Sources/MLX3DRuntime/MLX3DRuntime.swift:14-19`). Those are code defaults, not measurements.

Build with `xcodebuild`, not `swift build` / `swift test`, so Metal can produce `default.metallib`. Tests that touch MLX set `Device(.cpu)` in `MarchingCubesTests` and `ImageTensorTests` initializers and in `tripoSRWeightsLoadWhenPresent` (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:8-10`, `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:10-12`, `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:48`).
