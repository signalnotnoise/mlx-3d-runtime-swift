# Architecture and dataflow

Two paths return one `Mesh` (`Sources/MLX3DRuntime/Mesh.swift:5-16`). Sibling modules are called, not reimplemented.

## Text to mesh

`MLX3DRuntime.textToMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:77-111`) rejects an illegal grid size, checks `manifest.json` and the BPE file, constructs `ShapERuntime`, and awaits `generate` with the fields from `TextToMeshOptions` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:17-23`): guidance 15, 64 steps, grid 128, query batch 65536, seed 0, maximum grid 512.

`mesh(from:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:190-195`) copies `verticesArray()` and `facesArray()`. `rgbColors` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:197-213`) turns Shap-E RGBA (`count` divisible by 4) into packed RGB and drops the fourth component. An empty face list throws `emptyMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:107-109`).

Shap-E's latent, diffusion, and SDF sampler are external. This file does not name their tensor ranks.

## Image to mesh

`imageToMesh(imageURL:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:119-132`) rejects an illegal grid size before it checks the weights file, then calls `ImageTensor.rgb` (`Sources/MLX3DRuntime/ImageTensor.swift:11-21`).

`imageToMesh(cgImage:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:140-164`) optionally calls `foregroundCutout` (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:80-86`) and then `ImageTensor.rgb(cgImage:)` (`Sources/MLX3DRuntime/ImageTensor.swift:24-34`). `applyForegroundCutout` defaults to true and `foregroundRatio` to 0.85 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:57-58`). The white-background `removeBackground` (`Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:36-68`) is not on this path.

`ImageTensor` draws into a `side` by `side` RGBA bitmap, fills `background` first, aspect-fits the source, and flips Y so row 0 is the top (`Sources/MLX3DRuntime/ImageTensor.swift:68-96`). The returned tensor is `[1, side, side, 3]` float RGB in `0...1`. Default `side` is 512 and default background is 0.5 (`Sources/MLX3DRuntime/ImageTensor.swift:11-14`, `Sources/MLX3DRuntime/MLX3DRuntime.swift:49-56`).

`imageToMesh(image:)` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:170-188`) again checks the grid, then loads `TripoSRModel` weights, accepts `[H, W, 3]` or `[1, H, W, 3]` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:223-244`), calls `forwardTriplanes`, and passes the triplanes to `TripoSRMeshing.extract` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:6-46`). This repo does not read or check the triplane shape. The sibling README's `[1, 3, 64, 64, 40]` figure is a sibling measurement (`Vendor/triposr-mlx-swift/README.md:123`), not a check in this package.

`TripoSRMeshing` places the cube on `[-radius, radius]` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:21-22`). Density positions are generated per batch (`Sources/MLX3DRuntime/VolumeGrid.swift:30-50`, `Sources/MLX3DRuntime/TripoSRMeshing.swift:59-68`) and stored as one float per point (`Sources/MLX3DRuntime/TripoSRMeshing.swift:48-79`). Default batch size is 8192 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:52`). Grids outside 2...512 are rejected before that allocation (`Sources/MLX3DRuntime/VolumeGrid.swift:9-24`).

`MarchingCubesExtractor.extract` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:13-73`) subtracts `threshold` (default 25 on the image options, default 0 on the extractor itself) and reshapes to `[resolution, resolution, resolution]`. It calls sibling `MarchingCubes.extract` and, when the closure is set, colors vertices from `queryColors` (`Sources/MLX3DRuntime/TripoSRMeshing.swift:81-105`). A mesh with no faces from the high-level image API throws `emptyMesh` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:184-186`). The extractor itself can return an empty `Mesh` (`Sources/MLX3DRuntime/MarchingCubesExtractor.swift:48-49`).

## Export and the example

`MeshExport` writes OBJ (`v` / `f`, faces 1-based, optional `r g b` on `v`) and ASCII PLY (`Sources/MLX3DRuntime/MeshExport.swift:23-46`, `Sources/MLX3DRuntime/MeshExport.swift:72-78`, `Sources/MLX3DRuntime/MeshExport.swift:113-118`).

`MLX3DCLI` dispatches `text` and `image` (`Sources/MLX3DCLI/MLX3DCLI.swift:39-58`). `MLX3DExample` only references `Mesh`, `MarchingCubes`, and `TripoSRModel` and does not load weights (`Examples/Integration/Sources/MLX3DExample/MLX3DExample.swift:7-17`).
