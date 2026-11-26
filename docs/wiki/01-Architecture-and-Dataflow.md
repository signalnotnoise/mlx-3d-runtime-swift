# Architecture and dataflow

Text-to-mesh and TripoSR image-to-mesh each return one `Mesh` (`Sources/MLX3DRuntime/Mesh.swift:5-16`). Shap-E image-to-mesh returns one `Mesh` per batch row. Sibling modules are called, not reimplemented.

## Text to mesh

`MLX3DRuntime.textToMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:172-206`) rejects an illegal grid size, checks `manifest.json` and the BPE file, constructs `ShapERuntime`, and awaits `generate` with the fields from `TextToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:26-31`): guidance 15, 64 steps, grid 128, query batch 65536, seed 0, maximum grid 512.

`mesh(from:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:416-421`) copies `verticesArray()` and `facesArray()`. `rgbColors` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:423-439`) turns Shap-E RGBA (`count` divisible by 4) into packed RGB and drops the fourth component. An empty face list throws `emptyMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:202-204`).

Shap-E's latent, diffusion, and SDF sampler are external. This file does not name their tensor ranks.

## Image to mesh

`imageToMesh(imageURL:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:215-228`) rejects an illegal grid size before it checks the weights file, then calls `ImageTensor.rgb` (`Sources/MLX3DRuntime/ImageTensor.swift:11-21`). It does not run `foregroundCutout` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:214`).

`imageToMesh(cgImage:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:239-272`) optionally calls `foregroundCutout` (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:80-86`) and then `ImageTensor.rgb(cgImage:)` (`Sources/MLX3DRuntime/ImageTensor.swift:24-34`). `applyForegroundCutout` defaults to true and `foregroundRatio` to 0.85 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:150-151`). `cutoutFailurePolicy` defaults to `.strict` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:152`). `.fallbackToOriginal` is the Mesquette behavior: nil or a thrown error logs and the original image is composited (`Sources/MLX3DRuntime/MLX3DRuntime.swift:361-372`). The white-background `removeBackground` (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:36-68`) is not on this path.

`ImageTensor` draws into a `side` by `side` RGBA bitmap, fills `background` first, aspect-fits the source, and flips Y so row 0 is the top (`Sources/MLX3DRuntime/ImageTensor.swift:68-96`). The returned tensor is `[1, side, side, 3]` float RGB in `0...1`. Default `side` is 512 and default background is 0.5 (`Sources/MLX3DRuntime/ImageTensor.swift:11-14`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:142-149`).

`imageToMesh(image:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:329-347`) again checks the grid, then loads `TripoSRModel` weights, accepts `[H, W, 3]` or `[1, H, W, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:449-470`), calls `forwardTriplanes`, and passes the triplanes to `TripoSRMeshing.extract` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:6-46`). This repo does not read or check the triplane shape. The sibling README's `[1, 3, 64, 64, 40]` figure is a sibling measurement from public `triposr-mlx-swift` at `51994056febf3be5ed726e89de169037a2fbf6a6`, not a check in this package.

`TripoSRMeshing` places the cube on `[-radius, radius]` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:21-22`). Density positions are generated per batch (`Sources/MLX3DRuntime/VolumeGrid.swift:30-50`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:59-68`) and stored as one float per point (`Sources/MLX3DRuntime/TripoSRMeshing.swift:48-79`). Default batch size is 8192 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:146`). Grids outside 2...512 are rejected before that allocation (`Sources/MLX3DRuntime/VolumeGrid.swift:9-24`).

`MarchingCubesExtractor.extract` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:13-73`) subtracts `threshold` (default 25 on the image options, default 0 on the extractor itself) and reshapes to `[resolution, resolution, resolution]`. It calls sibling `MarchingCubes.extract` and, when the closure is set, colors vertices from `queryColors` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:81-105`). A mesh with no faces from the high-level TripoSR image API throws `emptyMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:343-345`). The extractor itself can return an empty `Mesh` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:48-49`).

## Shap-E image to mesh

`imageToMeshShapE` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:282-323`) is separate from TripoSR. It validates the grid, optionally runs white-background `removeBackground` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:379-401`), checks the same Shap-E bundle as `textToMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:403-413`), and calls `ShapERuntime.generate(image:)`. Defaults are guidance 3, 32 steps, grid 64, seed 0, and white prep on (`Sources/MLX3DRuntime/MLX3DRuntime.swift:86-98`). It returns one `Mesh` per batch row. `removeBackground` is not called from any TripoSR function.

## Export and the example

`MeshExport` writes OBJ (`v` / `f`, faces 1-based, optional `r g b` on `v`) and ASCII PLY (`Sources/MLX3DRuntime/MeshExport.swift:23-46`, `Sources/MLX3DRuntime/MeshExport.swift:72-78`, `Sources/MLX3DRuntime/MeshExport.swift:113-118`).

`MLX3DCLI` dispatches `text` and `image` (`Sources/MLX3DCLI/MLX3DCLI.swift:39-58`). `MLX3DExample` only references `Mesh`, `MarchingCubes`, and `TripoSRModel` and does not load weights (`Examples/Integration/Sources/MLX3DExample/MLX3DExample.swift:7-17`).
