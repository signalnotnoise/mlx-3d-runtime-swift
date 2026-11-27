# Engineering standards

Invariants of this repository. Sibling model code is not specified here.

## Tools and mlx-swift

`Package.swift:1` sets `swift-tools-version: 6.2`.

`Package.swift:29-31` caps `mlx-swift` at `0.31.3..<0.31.5`. The comment states that `0.31.5` and later set Swift tools 6.3. Public `shap-e-mlx-swift` and `triposr-mlx-swift` `main` (commits `d7b6b7aa88d11e265799fc81e5fea1b734571d2b` and `51994056febf3be5ed726e89de169037a2fbf6a6`) both declare that same range. Do not widen this package past `0.31.4`.

## Tensor layout and coordinates

Images accepted by `imageToMesh(image:)` are float RGB in `0...1`, rank 3 `[height, width, 3]` or rank 4 `[1, height, width, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:533-554`). Rank 3 is reshaped to `[1, H, W, 3]`. The file decoder returns `[1, side, side, 3]` (`Sources/MLX3DRuntime/ImageTensor.swift:20-30`). Row 0 is the top of the image (`Sources/MLX3DRuntime/ImageTensor.swift:123-124`). Alpha is composited over `background` (`Sources/MLX3DRuntime/ImageTensor.swift:89-96`). A `CGImage` uses the same draw (`Sources/MLX3DRuntime/ImageTensor.swift:34-45`). This path does not apply ImageNet normalization; that stays inside the sibling `forwardTriplanes`.

`VolumeGrid.positions` is packed `x, y, z` with X slowest and Z fastest. Sample `i` is `min + i / (resolution - 1) * (max - min)` (`Sources/MLX3DRuntime/VolumeGrid.swift:26-47`). TripoSR meshing uses the cube `[-radius, radius]` on each axis (`Sources/MLX3DRuntime/TripoSRMeshing.swift:22-23`). `radius` is `options.radius` or `model.renderer.radius` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:16`). This repo does not hardcode `0.87`.

`VolumeGrid.validate` rejects sizes outside `2...maximum` before weights load or a sample buffer is allocated (`Sources/MLX3DRuntime/VolumeGrid.swift:15-24`). `maximum` cannot exceed `VolumeGrid.maximumResolution` (512) (`Sources/MLX3DRuntime/VolumeGrid.swift:9`). `maximumGridSize` on the text, TripoSR image, and Shap-E image options can set a lower cap (`Sources/MLX3DRuntime/MLX3DRuntime.swift:22-23`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:144-145`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:89-90`). The static text entry forwards to a session that checks first (`Sources/MLX3DRuntime/MLX3DRuntime.swift:203-228`, `Sources/MLX3DRuntime/MLX3DSession.swift:148-149`). `imageToMeshShapE` and the `imageToMesh` overloads check at `Sources/MLX3DRuntime/MLX3DRuntime.swift:247-248`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:274-275`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:322-323`, and `Sources/MLX3DRuntime/MLX3DRuntime.swift:363-364`. The CLI runs the same check before it looks up a bundle or weights (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`, `Sources/MLX3DCLI/MLX3DCLI.swift:117-119`).

Density queries build one batch of positions with `points:` and reshape that batch to `[count, 3]` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:72-78`). The flattened density array must have `count` values (`Sources/MLX3DRuntime/TripoSRMeshing.swift:82-84`). The full density volume is still `resolution`³ floats because marching cubes needs the field. Position storage is one batch. Color queries use mesh vertices, not the volume grid, and require `count * 3` floats (`Sources/MLX3DRuntime/TripoSRMeshing.swift:113-119`). This repo does not assert the MLX rank of `evaluate`'s return value beyond that flat count.

Marching cubes subtracts `threshold` and reshapes the field to `[resolution, resolution, resolution]` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:34-38`). Values above the threshold are the interior. The bbox passed to Shap-E is the same min and size as the sample grid (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:39-54`). `useMetalMarchingCubes` defaults to false and selects `MetalMarchingCubes.extract`; false keeps `MarchingCubes.extract` on that same field. Positions come back packed `x,y,z`. Faces are `Int32` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:56-64`).

`Mesh.positions` length is `3V`. `Mesh.faces` length is `3F`. `Mesh.vertexColors`, when present, length is `3V` (`Sources/MLX3DRuntime/Mesh.swift:5-10`). Shap-E RGBA (`V * 4`) is stored as RGB, dropping the fourth component (`Sources/MLX3DRuntime/MLX3DRuntime.swift:507-523`).

## Weights

Never commit weights. `.gitignore` ignores `runtime_bundle/`, `triposr_swift_ready.safetensors`, and `*.safetensors`.

The library TripoSR image path requires the caller to pass a regular file (`Sources/MLX3DRuntime/MLX3DRuntime.swift:525-531`). It does not search the filesystem. `loadWeights(from:)` is the sibling API. This repo does not reimplement shape or unused-key checks. Grid validation runs before that file check (`Sources/MLX3DRuntime/MLX3DRuntime.swift:247-248`).

The library text path and `imageToMeshShapE` require `manifest.json` in the bundle directory and a BPE file, defaulting to `bpe_simple_vocab_16e6.txt.gz` beside that manifest (`Sources/MLX3DRuntime/MLX3DSession.swift:379-390`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:453-464`). Grid validation runs before the bundle is opened (`Sources/MLX3DRuntime/MLX3DSession.swift:148-149`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:322-323`).

`imageToMesh(cgImage:)` runs `foregroundCutout` at ratio 0.85 unless `applyForegroundCutout` is false (`Sources/MLX3DRuntime/MLX3DRuntime.swift:268-303`). `cutoutFailurePolicy` defaults to `.strict` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:52-55`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:171-171`): a nil cutout throws, and a thrown error propagates (`Sources/MLX3DRuntime/MLX3DRuntime.swift:409-416`). `.fallbackToOriginal` logs Mesquette's two messages and passes the original image (`Sources/MLX3DRuntime/MLX3DRuntime.swift:411-423`). The file-URL and tensor entries do not cut out (`Sources/MLX3DRuntime/MLX3DRuntime.swift:239-240`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:355-355`). The CLI image command calls the file-URL entry (`Sources/MLX3DCLI/MLX3DCLI.swift:130-139`), so it does not cut out either. The cutout itself is the Mesquette copy (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:82-88` and `Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:90-131`). The UIImage helper maps `UIImage.Orientation` with an explicit switch (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:189-210`) because those raw values are not `CGImagePropertyOrientation` values.

`imageToMeshShapE` is the only caller of the white `removeBackground` helper (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:38-70`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:326-329`). TripoSR never calls it. `applyWhiteBackgroundRemoval` defaults to true and can be set false (`Sources/MLX3DRuntime/MLX3DRuntime.swift:108-108`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:434-436`). A nil removal keeps the original image with no log; a throw logs "Background removal failed:" and keeps the original (`Sources/MLX3DRuntime/MLX3DRuntime.swift:440-449`).

## Loaded models

`MLX3DSession` caches loaded models in memory for the caller that holds the session (`Sources/MLX3DRuntime/MLX3DSession.swift:38-68`). The key is the backend, the bundle URL, and any other file the loader reads (`Sources/MLX3DRuntime/MLX3DModelKey.swift:18-54`). Shap-E's bundle holds the neural weights. The BPE vocabulary is part of the key because `ShapERuntime` loads it into the cached object. TripoSR's bundle URL is the safetensors file. Prompt, seed, grid, steps, and threshold are not part of the key.

A second request for a ready key does not call the loader. Concurrent requests for one key share one in-flight load. A thrown load is dropped, and the next request calls the loader again (`Sources/MLX3DRuntime/MLX3DModelCache.swift:46-65`). `unload` increments a per-key generation and drops the slot (`Sources/MLX3DRuntime/MLX3DModelCache.swift:83-96`). A load that already started cannot write the model back. Waiters throw `modelUnloaded` (`Sources/MLX3DRuntime/MLX3DRuntimeError.swift:17-20`). `evict` drops a ready model and leaves an in-flight load alone (`Sources/MLX3DRuntime/MLX3DModelCache.swift:69-74`). This package does not download weights or choose when to unload. Cutout types stay in `ImageBackgroundRemoval` and `MLX3DRuntime`.

CLI image search order is `--weights`, then `TRIPOSR_WEIGHTS_PATH`, then `./triposr_swift_ready.safetensors`, then `./triposr_weights/triposr_swift_ready.safetensors` (`Sources/MLX3DCLI/MLX3DCLI.swift:120-123` and `Sources/MLX3DCLI/MLX3DCLI.swift:165-172`). It does not search the older monorepo paths that the sibling CLI searches. An illegal `--grid-size` fails before that search (`Sources/MLX3DCLI/MLX3DCLI.swift:117-119`).

CLI text search is `--bundle` or `SHAP_E_BUNDLE` (`Sources/MLX3DCLI/MLX3DCLI.swift:76-78`). Optional `--bpe` overrides the bundle default (`Sources/MLX3DCLI/MLX3DCLI.swift:80`). Grid validation runs first (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`).

Tests also accept `TEST_RUNNER_TRIPOSR_WEIGHTS_PATH` and `TEST_RUNNER_SHAP_E_BUNDLE` (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-75`).

Canonical files, cited from the vendored sibling READMEs (sibling measurements, not re-hashed here):

- TripoSR: `triposr_swift_ready.safetensors`, 1,674,793,528 bytes, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`, Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d`. Those figures are in the public `triposr-mlx-swift` README at `51994056febf3be5ed726e89de169037a2fbf6a6`. This repo does not re-hash that file.
- Shap-E: Hub revision `d71f70a09b9c03535188d006a719edda5c19a6eb`, described as 1,104 files. The public `shap-e-mlx-swift` README at `d7b6b7aa88d11e265799fc81e5fea1b734571d2b` says upstream notes record 5,143,254,750 bytes and that the package does not re-measure that payload. This repo does not record a single SHA-256 for that bundle.

## Parity

This repo does not ship Python parity dumps or a `--parity` mode. `MLX3DCLI` has `text` and `image` only (`Sources/MLX3DCLI/MLX3DCLI.swift:49-58`).

The image grid matches the TripoSR extraction convention used by this code: density minus threshold, positive inside, no axis swap (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:6-12`). Whether that matches a given upstream commit beyond that comment is not re-checked in this repo.

`DINOv2` interpolation and Shap-E diffusion numerics are sibling behavior. Do not change them from this package.

## Memory and Metal

This repo does not set `Memory.cacheLimit` and does not call `Device.setDefault` on the library path. `MLX.eval` is used after triplane construction, each density batch, each color batch, and marching-cubes outputs (`Sources/MLX3DRuntime/MLX3DRuntime.swift:385-385`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:80-80`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:116-116`, `Sources/MLX3DRuntime/MarchingCubesExtractor.swift:57-57`).

Point queries are chunked by `queryBatchSize` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:68-94`). Each density batch allocates only that batch of positions (`Sources/MLX3DRuntime/TripoSRMeshing.swift:72-77`). The TripoSR image default is 8192 and the text default passed to Shap-E is 65536 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:161-164` and `Sources/MLX3DRuntime/MLX3DRuntime.swift:32-36`). The Shap-E image entry passes its own query batch, 4096 on macOS and 1024 elsewhere (`Sources/MLX3DRuntime/MLX3DRuntime.swift:124-130`). Those are code defaults, not measurements. Grids larger than 512 are rejected (`Sources/MLX3DRuntime/VolumeGrid.swift:9-23`).

Build with `xcodebuild`, not `swift build` / `swift test`, so Metal can produce `default.metallib`. Tests that touch MLX set `Device(.cpu)` in `MarchingCubesTests` and `ImageTensorTests` initializers and in `tripoSRWeightsLoadWhenPresent` (`Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:8-10`, `Tests/MLX3DRuntimeTests/ImageTensorTests.swift:10-12`, `Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:48`).

## Export

Stored meshes stay in the generator's coordinates. Shap-E and TripoSR are Z-up: the body runs from the legs to the back along Z, and the back is toward +Z (`Sources/MLX3DRuntime/MeshExport.swift:40-51`). `MeshExportOptions.upAxis` defaults to `.native` (`Sources/MLX3DRuntime/MeshExport.swift:64-76`). `.yUp` allocates a new position buffer at export and leaves the `Mesh` value alone (`Sources/MLX3DRuntime/MeshExport.swift:329-348`). The map is `(x, y, z)` to `(x, z, -y)`, a rotation by -90 degrees about X. Its determinant is +1. Face order is copied (`Sources/MLX3DRuntime/MeshExport.swift:426-437`), so geometric normals rotate with the vertices.

`writeOBJ` defaults to Mesquette's OBJ+MTL (`Sources/MLX3DRuntime/MeshExport.swift:84-111`): `mtllib` names the sibling `.mtl` (`Sources/MLX3DRuntime/MeshExport.swift:109-111`), `usemtl` is `VertexColorAverage`, `v` lines are positions, and `Kd` is the average RGB or `0.75 0.75 0.75` (`Sources/MLX3DRuntime/MeshExport.swift:197-234`). `objColors: .vertexRGB` and `objText` keep `v x y z r g b` and do not write an MTL (`Sources/MLX3DRuntime/MeshExport.swift:171-194`). PLY remains ASCII, with optional float colors (`Sources/MLX3DRuntime/MeshExport.swift:236-271`). USDZ is a Model I/O mesh; per-vertex RGB is `MDLVertexAttributeColor` (`Sources/MLX3DRuntime/MeshExport.swift:125-149`, `Sources/MLX3DRuntime/MeshExport.swift:350-393`). Every index passes `validatedFaceIndices` (`Sources/MLX3DRuntime/MeshExport.swift:312-326`). Share sheets, ZIP packaging, file names, and scene placement are not written here. The USD exporter has to be registered in the process; a bare command-line tool reports that it cannot export `usdz`. `MLX3DCLI` therefore does not offer USDZ. The SwiftPM test process does not register it either, so `MeshExportTests` skips USDZ unless `MDLAsset.canExportFileExtension("usdz")` is true. CI proves USDZ inside `ExportHost`, an app-hosted XCTest bundle (`Examples/ExportHost/ExportHostTests/USDZHostTests.swift:11-259`), on macOS (`.github/workflows/ci.yml:149-162`) and on the iOS Simulator (`.github/workflows/ci.yml:275-285`).
