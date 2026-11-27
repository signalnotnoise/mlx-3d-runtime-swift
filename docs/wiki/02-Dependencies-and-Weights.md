# Dependencies and weights

## Package pins

`Package.swift:21-32` depends on:

- `mlx-swift` `0.31.3..<0.31.5` (`Package.swift:24`).
- `shap-e-mlx-swift` `from: "0.1.0"` (`Package.swift:25-28`).
- `triposr-mlx-swift` `from: "0.1.0"` (`Package.swift:29-32`).

The example manifest uses the same version pins (`Examples/Integration/Package.swift:21-28`).

`Package.resolved` records the revisions resolved from those `0.1.0` tags. The Shap-E pin is `962403e2ccef75a10c084c09c70c92899e93d328` (`v0.1.0`, which includes `7ed124d`), whose `verticesArray`, `facesArray`, and `colorsArray` read tensor bytes instead of `MLXArray.asArray`. CI resolves again from GitHub with an empty cache (`.github/workflows/ci.yml:49-96`). There is no `Vendor/` mirror.

`Examples/Integration/Package.swift:20` depends on this repo by `path: "../.."`. The package identity is the checkout directory name `mlx-3d-runtime-swift` (`Examples/Integration/Package.swift:5-7` and `Examples/Integration/Package.swift:34`).

Products of this package are `MLX3DRuntime` and `MLX3DCLI` (`Package.swift:11-19`). The example product is `MLX3DExample` (`Examples/Integration/Package.swift:16-17`).

## What each path needs

Text-to-mesh needs a Shap-E runtime bundle directory containing `manifest.json` and, unless `--bpe` / `bpeVocabularyURL` is set, `bpe_simple_vocab_16e6.txt.gz` (`Sources/MLX3DRuntime/MLX3DSession.swift:379-390`). `imageToMeshShapE` checks the same two files (`Sources/MLX3DRuntime/MLX3DRuntime.swift:453-464`). The session image entry uses that same loader. The grid must already be inside 2...512 (`Sources/MLX3DRuntime/MLX3DSession.swift:148-149`). A session reuses that loaded runtime for the same bundle and BPE URL. The static `textToMesh` builds a new session per call (`Sources/MLX3DRuntime/MLX3DRuntime.swift:203-228`). The README points that bundle at Hugging Face `signalNotNoise/shap-e-mlx` revision `d71f70a09b9c03535188d006a719edda5c19a6eb` (`README.md:51`). The public `shap-e-mlx-swift` README at `d7b6b7aa88d11e265799fc81e5fea1b734571d2b` says that revision lists 1,104 files and that a byte size of 5,143,254,750 is an upstream note the sibling package did not re-measure. This repo does not store a bundle SHA-256.

Image-to-mesh needs `triposr_swift_ready.safetensors` passed as a file URL (`Sources/MLX3DRuntime/MLX3DRuntime.swift:369-373`). The grid check at `Sources/MLX3DRuntime/MLX3DRuntime.swift:363-364` runs before `loadWeights`. The static image methods load that file on every call. `MLX3DSession.imageToMesh`, including the `CGImage` entry, keeps the loaded `TripoSRModel` for the weights URL (`Sources/MLX3DRuntime/MLX3DSession.swift:392-395`). The public `triposr-mlx-swift` README at `51994056febf3be5ed726e89de169037a2fbf6a6` records 1,674,793,528 bytes, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`, and Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d`. This repo does not re-hash that file.

Weights are gitignored (`.gitignore` lists `runtime_bundle/`, `triposr_swift_ready.safetensors`, and `*.safetensors`). Do not commit them.

## Search order

The library does not search. Image calls require the file the caller passed (`Sources/MLX3DRuntime/MLX3DRuntime.swift:525-531`).

`MLX3DCLI` image order is `--weights`, then `TRIPOSR_WEIGHTS_PATH`, then `triposr_swift_ready.safetensors`, then `triposr_weights/triposr_swift_ready.safetensors` (`Sources/MLX3DCLI/MLX3DCLI.swift:120-123`, `Sources/MLX3DCLI/MLX3DCLI.swift:165-172`). Text order is `--bundle` or `SHAP_E_BUNDLE` (`Sources/MLX3DCLI/MLX3DCLI.swift:76-78`). Both commands reject `--grid-size` outside 2...512 before that lookup (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`, `Sources/MLX3DCLI/MLX3DCLI.swift:117-119`). `--max-grid-size` can only lower the cap (`Sources/MLX3DRuntime/VolumeGrid.swift:15-17`).

Unit tests additionally honor `TEST_RUNNER_` prefixes (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-90`).
