import Darwin
import Foundation
import Testing

#if os(macOS)
struct CLIExitTests {
    @Test func missingSubcommandExitsNonzero() throws {
        let result = try runCLI([])
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"), Comment(rawValue: result.output))
    }

    @Test func helpExitsZero() throws {
        let result = try runCLI(["--help"])
        #expect(result.status == 0, Comment(rawValue: result.output))
        #expect(result.output.contains("Usage:"), Comment(rawValue: result.output))
    }

    @Test func textExitsNonzeroWhenBundleIsMissing() throws {
        let result = try runCLI(["text", "--prompt", "a chair", "--bundle", "/nonexistent/runtime_bundle"])
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"), Comment(rawValue: result.output))
    }

    @Test func imageExitsNonzeroWhenGridSizeIsInvalid() throws {
        let result = try runCLI(
            ["image", "--input", "/nonexistent/photo.png", "--grid-size", "9999"]
        )
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"), Comment(rawValue: result.output))
        #expect(result.output.contains("9999"), Comment(rawValue: result.output))
    }

    @Test func imageExitsNonzeroWhenWeightsAreMissing() throws {
        let result = try runCLI(["image", "--input", "/nonexistent/photo.png"])
        #expect(result.status != 0)
        #expect(result.output.contains("Failed:"), Comment(rawValue: result.output))
        #expect(result.output.contains("weights"), Comment(rawValue: result.output))
    }

    private func runCLI(_ arguments: [String]) throws -> (status: Int32, output: String) {
        let cli = try #require(
            builtExecutable("MLX3DCLI"),
            "MLX3DCLI was not next to the test image"
        )
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx3d-cli-\(UUID().uuidString)", isDirectory: true)
        let workingDirectory = root.appendingPathComponent("cwd", isDirectory: true)
        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let outputURL = root.appendingPathComponent("stdout.txt")
        let errorURL = root.appendingPathComponent("stderr.txt")
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        FileManager.default.createFile(atPath: errorURL.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: outputURL)
        let errorHandle = try FileHandle(forWritingTo: errorURL)
        defer {
            try? outputHandle.close()
            try? errorHandle.close()
        }

        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "TRIPOSR_WEIGHTS_PATH")
        environment.removeValue(forKey: "TEST_RUNNER_TRIPOSR_WEIGHTS_PATH")
        environment.removeValue(forKey: "SHAP_E_BUNDLE")
        environment.removeValue(forKey: "TEST_RUNNER_SHAP_E_BUNDLE")

        let process = Process()
        process.executableURL = cli
        process.arguments = arguments
        process.currentDirectoryURL = workingDirectory
        process.environment = environment
        process.standardOutput = outputHandle
        process.standardError = errorHandle
        try process.run()
        process.waitUntilExit()

        let stdout = (try? String(contentsOf: outputURL, encoding: .utf8)) ?? ""
        let stderr = (try? String(contentsOf: errorURL, encoding: .utf8)) ?? ""
        return (process.terminationStatus, stdout + stderr)
    }
}

func builtExecutable(_ name: String) -> URL? {
    var info = Dl_info()
    guard dladdr(#dsohandle, &info) != 0, let fileName = info.dli_fname else {
        return nil
    }
    var directory = URL(fileURLWithPath: String(cString: fileName))
        .resolvingSymlinksInPath()
        .deletingLastPathComponent()
    let fileManager = FileManager.default
    for _ in 0..<8 {
        let candidate = directory.appendingPathComponent(name)
        if fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate
        }
        let parent = directory.deletingLastPathComponent()
        if parent.path == directory.path {
            break
        }
        directory = parent
    }
    return nil
}
#endif
