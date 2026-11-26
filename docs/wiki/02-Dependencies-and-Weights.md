# Dependencies and weights

## Package pins

`Package.swift:28-41` depends on:

- `mlx-swift` `0.31.3..<0.31.5` (`Package.swift:31`).
- `shap-e-mlx-swift` public branch `main` (`Package.swift:33-36`).
- `triposr-mlx-swift` public branch `main` (`Package.swift:38-41`).

TODO in `Package.swift:4-9` and again at `Package.swift:32` and `Package.swift:37`: switch both sibling pins to `from: "0.1.0"` once those repositories are tagged. The example manifest repeats public `main` and the same TODO (`Examples/Integration/Package.swift:9-10` and `Examples/Integration/Package.swift:24-33`).

`Package.resolved` records the revisions resolved from those `main` branches. CI resolves again from GitHub with an empty cache (`.github/workflows/ci.yml:49-96`). There is no `Vendor/` mirror.

`Examples/Integration/Package.swift:22` depends on this repo by `path: "../.."`. The package identity is the checkout directory name `mlx-3d-runtime-swift` (`Examples/Integration/Package.swift:5-7` and `Examples/Integration/Package.swift:38`).

Products of this package are `MLX3DRuntime` and `MLX3DCLI` (`Package.swift:18-26`). The example product is `MLX3DExample` (`Examples/Integration/Package.swift:18-19`).

## What each path needs

Text-to-mesh needs a Shap-E runtime bundle directory containing `manifest.json` and, unless `--bpe` / `bpeVocabularyURL` is set, `bpe_simple_vocab_16e6.txt.gz` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:75-83`). The grid must already be inside 2...512 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:74`). The README points that bundle at Hugging Face `signalNotNoise/shap-e-mlx` revision `d71f70a09b9c03535188d006a719edda5c19a6eb` (`README.md:51`). The public `shap-e-mlx-swift` README at `d7b6b7aa88d11e265799fc81e5fea1b734571d2b` says that revision lists 1,104 files and that a byte size of 5,143,254,750 is an upstream note the sibling package did not re-measure. This repo does not store a bundle SHA-256.

Image-to-mesh needs `triposr_swift_ready.safetensors` passed as a file URL (`Sources/MLX3DRuntime/MLX3DRuntime.swift:134-138`). The grid check at `Sources/MLX3DRuntime/MLX3DRuntime.swift:134` runs before `loadWeights`. The public `triposr-mlx-swift` README at `51994056febf3be5ed726e89de169037a2fbf6a6` records 1,674,793,528 bytes, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`, and Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d`. This repo does not re-hash that file.

Weights are gitignored (`.gitignore` lists `runtime_bundle/`, `triposr_swift_ready.safetensors`, and `*.safetensors`). Do not commit them.

## Search order

The library does not search. Image calls require the file the caller passed (`Sources/MLX3DRuntime/MLX3DRuntime.swift:174-179`).

`MLX3DCLI` image order is `--weights`, then `TRIPOSR_WEIGHTS_PATH`, then `triposr_swift_ready.safetensors`, then `triposr_weights/triposr_swift_ready.safetensors` (`Sources/MLX3DCLI/MLX3DCLI.swift:120-123`, `Sources/MLX3DCLI/MLX3DCLI.swift:165-172`). Text order is `--bundle` or `SHAP_E_BUNDLE` (`Sources/MLX3DCLI/MLX3DCLI.swift:76-78`). Both commands reject `--grid-size` outside 2...512 before that lookup (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`, `Sources/MLX3DCLI/MLX3DCLI.swift:117-119`). `--max-grid-size` can only lower the cap (`Sources/MLX3DRuntime/VolumeGrid.swift:15-17`).

Unit tests additionally honor `TEST_RUNNER_` prefixes (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-90`).
