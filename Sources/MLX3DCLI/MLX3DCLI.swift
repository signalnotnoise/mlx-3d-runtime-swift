import Foundation
import MLX3DRuntime

enum CLIError: Error, CustomStringConvertible {
    case usage(String)
    case failed(String)

    var description: String {
        switch self {
        case .usage(let text):
            return text
        case .failed(let text):
            return text
        }
    }
}

@main
struct MLX3DCLI {
    static func main() async {
        do {
            try await run()
        } catch let error as CLIError {
            switch error {
            case .usage(let text):
                FileHandle.standardError.write(Data("Failed: \(text)".utf8))
                exit(1)
            case .failed(let text):
                FileHandle.standardError.write(Data("Failed: \(text)\n".utf8))
                exit(1)
            }
        } catch {
            let message = "Failed: \(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            exit(1)
        }
    }

    static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.isEmpty {
            throw CLIError.usage(helpText)
        }
        if arguments == ["--help"] || arguments == ["-h"] {
            print(helpText)
            return
        }

        switch arguments[0] {
        case "text":
            try await runText(Array(arguments.dropFirst()))
        case "image":
            try runImage(Array(arguments.dropFirst()))
        case "--help", "-h":
            print(helpText)
        default:
            throw CLIError.usage(helpText)
        }
    }

    static func runText(_ arguments: [String]) async throws {
        let parser = try FlagParser(
            arguments: arguments,
            valued: ["prompt", "bundle", "bpe", "output", "grid-size", "max-grid-size", "steps", "guidance", "seed", "query-batch-size", "format"]
        )
        if parser.help {
            print(textHelp)
            return
        }
        guard let prompt = parser.values["prompt"], !prompt.isEmpty else {
            throw CLIError.failed("text requires --prompt")
        }
        let gridSize = try parser.integer("grid-size", default: 128)
        let maximumGridSize = try parser.integer("max-grid-size", default: VolumeGrid.maximumResolution)
        try VolumeGrid.validate(resolution: gridSize, maximum: maximumGridSize)
        guard let bundlePath = parser.values["bundle"] ?? environment("SHAP_E_BUNDLE") else {
            throw CLIError.failed("text requires --bundle or SHAP_E_BUNDLE")
        }
        let bundle = URL(fileURLWithPath: bundlePath)
        let bpe = parser.values["bpe"].map { URL(fileURLWithPath: $0) }
        let steps = try parser.integer("steps", default: 64)
        let seed = try parser.integer("seed", default: 0)
        let queryBatchSize = try parser.integer("query-batch-size", default: 65536)
        let guidance = try parser.float("guidance", default: 15)
        let format = try parser.format(defaultOutput: "mesh.obj")
        let output = URL(fileURLWithPath: parser.values["output"] ?? format.defaultFileName)

        let mesh = try await MLX3DRuntime.textToMesh(
            prompt: prompt,
            modelBundleURL: bundle,
            bpeVocabularyURL: bpe,
            options: TextToMeshOptions(
                guidanceScale: guidance,
                numInferenceSteps: steps,
                gridSize: gridSize,
                queryBatchSize: queryBatchSize,
                seed: seed,
                maximumGridSize: maximumGridSize
            )
        )
        try write(mesh, to: output, format: format.kind)
        report(mesh, url: output)
    }

    static func runImage(_ arguments: [String]) throws {
        let parser = try FlagParser(
            arguments: arguments,
            valued: ["input", "weights", "output", "grid-size", "max-grid-size", "threshold", "query-batch-size", "format"]
        )
        if parser.help {
            print(imageHelp)
            return
        }
        guard let input = parser.values["input"], !input.isEmpty else {
            throw CLIError.failed("image requires --input")
        }
        let gridSize = try parser.integer("grid-size", default: 128)
        let maximumGridSize = try parser.integer("max-grid-size", default: VolumeGrid.maximumResolution)
        try VolumeGrid.validate(resolution: gridSize, maximum: maximumGridSize)
        guard let weightsPath = parser.values["weights"] ?? environment("TRIPOSR_WEIGHTS_PATH") ?? relativeWeights() else {
            throw CLIError.failed(
                "weights not found (pass --weights, set TRIPOSR_WEIGHTS_PATH, or place triposr_swift_ready.safetensors in the working directory)"
            )
        }
        let queryBatchSize = try parser.integer("query-batch-size", default: 8192)
        let threshold = try parser.float("threshold", default: 25)
        let format = try parser.format(defaultOutput: "mesh.obj")
        let output = URL(fileURLWithPath: parser.values["output"] ?? format.defaultFileName)

        let mesh = try MLX3DRuntime.imageToMesh(
            imageURL: URL(fileURLWithPath: input),
            weightsURL: URL(fileURLWithPath: weightsPath),
            options: ImageToMeshOptions(
                gridSize: gridSize,
                threshold: threshold,
                queryBatchSize: queryBatchSize,
                maximumGridSize: maximumGridSize
            )
        )
        try write(mesh, to: output, format: format.kind)
        report(mesh, url: output)
    }

    static func write(_ mesh: Mesh, to url: URL, format: MeshKind) throws {
        switch format {
        case .obj:
            try MeshExport.writeOBJ(mesh, to: url)
        case .ply:
            try MeshExport.writePLY(mesh, to: url)
        }
    }

    static func report(_ mesh: Mesh, url: URL) {
        print("vertices \(mesh.vertexCount)")
        print("faces \(mesh.faceCount)")
        print("wrote \(url.path)")
    }

    static func environment(_ key: String) -> String? {
        let value = ProcessInfo.processInfo.environment[key]
        guard let value, !value.isEmpty else { return nil }
        return value
    }

    static func relativeWeights() -> String? {
        let candidates = [
            "triposr_swift_ready.safetensors",
            "triposr_weights/triposr_swift_ready.safetensors",
        ]
        return candidates.first {
            FileManager.default.fileExists(atPath: $0)
        }
    }
}

enum MeshKind {
    case obj
    case ply
}

struct ResolvedFormat {
    var kind: MeshKind
    var defaultFileName: String
}

struct FlagParser {
    var values: [String: String] = [:]
    var help = false

    init(arguments: [String], valued: Set<String>) throws {
        var index = 0
        while index < arguments.count {
            let token = arguments[index]
            if token == "--help" || token == "-h" {
                help = true
                index += 1
                continue
            }
            guard token.hasPrefix("--"), token.count > 2 else {
                throw CLIError.failed("Unexpected argument \(token)")
            }
            let name = String(token.dropFirst(2))
            guard valued.contains(name) else {
                throw CLIError.failed("Unknown flag --\(name)")
            }
            let next = index + 1
            guard next < arguments.count, !arguments[next].hasPrefix("--") else {
                throw CLIError.failed("Missing value for --\(name)")
            }
            values[name] = arguments[next]
            index = next + 1
        }
    }

    func integer(_ name: String, default defaultValue: Int) throws -> Int {
        guard let raw = values[name] else { return defaultValue }
        guard let value = Int(raw) else {
            throw CLIError.failed("--\(name) must be an integer, got \(raw)")
        }
        return value
    }

    func float(_ name: String, default defaultValue: Float) throws -> Float {
        guard let raw = values[name] else { return defaultValue }
        guard let value = Float(raw) else {
            throw CLIError.failed("--\(name) must be a number, got \(raw)")
        }
        return value
    }

    func format(defaultOutput: String) throws -> ResolvedFormat {
        let explicit = values["format"]?.lowercased()
        let output = values["output"] ?? defaultOutput
        let extensionName = URL(fileURLWithPath: output).pathExtension.lowercased()
        let chosen = explicit ?? (extensionName.isEmpty ? "obj" : extensionName)
        switch chosen {
        case "obj":
            return ResolvedFormat(kind: .obj, defaultFileName: "mesh.obj")
        case "ply":
            return ResolvedFormat(kind: .ply, defaultFileName: "mesh.ply")
        default:
            throw CLIError.failed("Unknown --format \(chosen). Use obj or ply.")
        }
    }
}

let helpText = """
mlx-3d — text-to-3D (Shap-E) and image-to-3D (TripoSR) on Apple Silicon

Usage:
  MLX3DCLI text --prompt <text> --bundle <runtime-bundle> [options]
  MLX3DCLI image --input <image> --weights <triposr_swift_ready.safetensors> [options]
  MLX3DCLI --help

text options:
  --prompt <text>              Required
  --bundle <dir>               Shap-E bundle with manifest.json (or SHAP_E_BUNDLE)
  --bpe <file>                 CLIP BPE file. Default: <bundle>/bpe_simple_vocab_16e6.txt.gz
  --output <file>              Default mesh.obj
  --format obj|ply             Default follows --output, otherwise obj
  --grid-size <int>            Default 128. Must be 2...512 (VolumeGrid.maximumResolution)
  --max-grid-size <int>        Optional lower cap. Cannot raise the hard cap of 512
  --steps <int>                Default 64
  --guidance <float>           Default 15
  --seed <int>                 Default 0
  --query-batch-size <int>     Default 65536

image options:
  --input <image>              Required
  --weights <file>             safetensors file, TRIPOSR_WEIGHTS_PATH, or ./triposr_swift_ready.safetensors
  --output <file>              Default mesh.obj
  --format obj|ply             Default follows --output, otherwise obj
  --grid-size <int>            Default 128. Must be 2...512 (VolumeGrid.maximumResolution)
  --max-grid-size <int>        Optional lower cap. Cannot raise the hard cap of 512
  --threshold <float>          Default 25
  --query-batch-size <int>     Default 8192

Missing weights, a missing bundle, and a failed run exit nonzero.

"""

let textHelp = """
MLX3DCLI text --prompt <text> --bundle <runtime-bundle> [--output mesh.obj]

Runs Shap-E and writes an OBJ or PLY mesh. See MLX3DCLI --help.

"""

let imageHelp = """
MLX3DCLI image --input <image> --weights <triposr_swift_ready.safetensors> [--output mesh.obj]

Runs TripoSR, samples density on a grid, and writes an OBJ or PLY mesh. See MLX3DCLI --help.

"""
