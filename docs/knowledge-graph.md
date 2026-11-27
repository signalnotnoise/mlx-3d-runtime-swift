# Knowledge graph

`verified_commit` `4d5c73ee1ad4581ede2e87fae6678b4ca51abf69` on `cursor/mlx-3d-runtime-swift-5003` is a label. `scripts/verify_docs.py` does not require it to be an ancestor of `HEAD`. It hashes the cited lines in the working tree. Sibling modules are external and have no line numbers here.

```mermaid
flowchart TD
  subgraph products [This package]
    CLI[MLX3DCLI]
    API[MLX3DRuntime]
    SESSION[MLX3DSession]
    IMG[ImageTensor]
    GRID[VolumeGrid]
    MESH_EXT[TripoSRMeshing]
    MC[MarchingCubesExtractor]
    MESH[Mesh]
    EXP[MeshExport]
  end
  subgraph external [Sibling packages]
    SHAPE[ShapERuntime]
    TRIPO[TripoSRRuntime]
  end
  CLI --> API
  API -->|throwaway session| SESSION
  SESSION -->|cached text prompt| SHAPE
  SHAPE -->|positions and faces| MESH
  API --> IMG
  IMG -->|image 1 x side x side x 3| API
  API -->|triplanes unchecked| MESH_EXT
  MESH_EXT --> GRID
  GRID -->|packed xyz| MESH_EXT
  MESH_EXT -->|positions count x 3| TRIPO
  TRIPO -->|density and color| MESH_EXT
  MESH_EXT --> MC
  MC -->|SDF resolution cubed| SHAPE
  MC --> MESH
  API --> EXP
  CLI --> EXP
```

## Where to find things

| Task | Location |
| --- | --- |
| Package pins and products | `Package.swift:1-68` |
| Text-to-mesh options and call | `Sources/MLX3DRuntime/MLX3DRuntime.swift:16-184` |
| Caller-owned loaded-model session | `Sources/MLX3DRuntime/MLX3DSession.swift:2-413` |
| Model key | `Sources/MLX3DRuntime/MLX3DModelKey.swift:1-78` |
| In-memory load cache | `Sources/MLX3DRuntime/MLX3DModelCache.swift:1-168` |
| Shap-E image options and call | `Sources/MLX3DRuntime/MLX3DRuntime.swift:68-122` |
| Image decode | `Sources/MLX3DRuntime/ImageTensor.swift:6-107` |
| CGImage cutout then TripoSR | `Sources/MLX3DRuntime/MLX3DRuntime.swift:217-250` |
| Cutout failure policy | `Sources/MLX3DRuntime/MLX3DRuntime.swift:46-56` |
| Copied Mesquette cutout | `Sources/MLX3DRuntime/ImageBackgroundRemoval.swift:6-246` |
| Shap-E white prep then image generation | `Sources/MLX3DRuntime/MLX3DRuntime.swift:260-301` |
| Image-to-mesh and image rank check | `Sources/MLX3DRuntime/MLX3DRuntime.swift:186-462` |
| Density grid and batches | `Sources/MLX3DRuntime/TripoSRMeshing.swift:5-106` |
| Sample coordinates and the 512 cap | `Sources/MLX3DRuntime/VolumeGrid.swift:3-68` |
| Grid rejection and lazy-batch tests | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:104-175` |
| Loaded-model cache tests | `Tests/MLX3DRuntimeTests/ModelCacheTests.swift:8-272` |
| Marching cubes | `Sources/MLX3DRuntime/MarchingCubesExtractor.swift:5-74` |
| Mesh topology | `Sources/MLX3DRuntime/Mesh.swift:5-90` |
| OBJ, PLY, and USDZ | `Sources/MLX3DRuntime/MeshExport.swift:17-465` |
| CLI | `Sources/MLX3DCLI/MLX3DCLI.swift:18-168` |
| Example link check | `Examples/Integration/Sources/MLX3DExample/MLX3DExample.swift:1-18` |
| CI | `.github/workflows/ci.yml:16-285` |
| App-hosted USDZ tests | `Examples/ExportHost/ExportHostTests/USDZHostTests.swift:11-259` |
| Public sibling branch pins | `Package.swift:33-41` |
| Ellipsoid test | `Tests/MLX3DRuntimeTests/MarchingCubesTests.swift:51-92` |
