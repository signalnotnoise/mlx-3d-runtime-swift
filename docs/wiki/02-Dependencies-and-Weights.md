# Dependencies and weights

## Package pins

`Package.swift:28-42` depends on:

- `mlx-swift` `0.31.3..<0.31.5` (`Package.swift:32`).
- `shap-e-mlx-swift` branch `cursor/shap-e-runtime-package-ea5b` (`Package.swift:34-37`).
- `triposr-mlx-swift` branch `cursor/triposr-mlx-swift-package-45c1` (`Package.swift:39-42`).

TODO in `Package.swift:4-9` and again at `Package.swift:33` and `Package.swift:38`: switch both sibling pins to `from: "0.1.0"` once those repositories are tagged. The example manifest repeats the same branches and the same TODO (`Examples/Integration/Package.swift:9` and `Examples/Integration/Package.swift:23-32`).

The checkout snapshots used by CI are recorded in `Vendor/SNAPSHOT.txt:1-2`:

- shap-e commit `9d042666fbfbd40b07417cd23dfe34d0731d673f`
- triposr commit `aafdeee6081dd780986479a72f34baab1a4f3c09`

`Examples/Integration/Package.swift:22` depends on this repo by `path: "../.."`. The package identity is the checkout directory name `mlx-3d-runtime-swift` (`Examples/Integration/Package.swift:5-7` and `Examples/Integration/Package.swift:38`).

Products of this package are `MLX3DRuntime` and `MLX3DCLI` (`Package.swift:18-26`). The example product is `MLX3DExample` (`Examples/Integration/Package.swift:18-19`).

## What each path needs

Text-to-mesh needs a Shap-E runtime bundle directory containing `manifest.json` and, unless `--bpe` / `bpeVocabularyURL` is set, `bpe_simple_vocab_16e6.txt.gz` (`Sources/MLX3DRuntime/MLX3DRuntime.swift:183-187`). `imageToMeshShapE` checks the same two files (`Sources/MLX3DRuntime/MLX3DRuntime.swift:403-413`). The grid must already be inside 2...512 (`Sources/MLX3DRuntime/MLX3DRuntime.swift:178`). The README points that bundle at Hugging Face `signalNotNoise/shap-e-mlx` revision `d71f70a09b9c03535188d006a719edda5c19a6eb` (`README.md:53`). The sibling snapshot says that revision lists 1,104 files and that a byte size of 5,143,254,750 is an upstream note the sibling package did not re-measure (`Vendor/shap-e-mlx-swift/README.md:63`). This repo does not store a bundle SHA-256.

Image-to-mesh needs `triposr_swift_ready.safetensors` passed as a file URL (`Sources/MLX3DRuntime/MLX3DRuntime.swift:335-338`). The grid check at `Sources/MLX3DRuntime/MLX3DRuntime.swift:334` runs before `loadWeights`. The sibling snapshot records 1,674,793,528 bytes, SHA-256 `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`, and Hub commit `76f3028746d0b5f39b6ba0205ede9e28835f106d` (`Vendor/triposr-mlx-swift/README.md:67-70`, `Vendor/triposr-mlx-swift/README.md:121`). This repo does not re-hash that file.

Weights are gitignored (`.gitignore` lists `runtime_bundle/`, `triposr_swift_ready.safetensors`, and `*.safetensors`). Do not commit them.

## Search order

The library does not search. Image calls require the file the caller passed (`Sources/MLX3DRuntime/MLX3DRuntime.swift:441-446`).

`MLX3DCLI` image order is `--weights`, then `TRIPOSR_WEIGHTS_PATH`, then `triposr_swift_ready.safetensors`, then `triposr_weights/triposr_swift_ready.safetensors` (`Sources/MLX3DCLI/MLX3DCLI.swift:120-123`, `Sources/MLX3DCLI/MLX3DCLI.swift:165-172`). Text order is `--bundle` or `SHAP_E_BUNDLE` (`Sources/MLX3DCLI/MLX3DCLI.swift:76-78`). Both commands reject `--grid-size` outside 2...512 before that lookup (`Sources/MLX3DCLI/MLX3DCLI.swift:73-75`, `Sources/MLX3DCLI/MLX3DCLI.swift:117-119`). `--max-grid-size` can only lower the cap (`Sources/MLX3DRuntime/VolumeGrid.swift:15-17`).

Unit tests additionally honor `TEST_RUNNER_` prefixes (`Tests/MLX3DRuntimeTests/RuntimeContractTests.swift:67-90`).
