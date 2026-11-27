import CoreGraphics
import Foundation
import MLX
import Testing

@testable import MLX3DRuntime

/// Cache rules for ``MLX3DModelCache``, the store ``MLX3DSession`` uses.
/// Loaders are fakes. These tests do not read weights.
struct ModelCacheTests {
    @Test func secondCallReusesTheLoadedModel() async throws {
        let loader = ScriptedLoader(block: false)
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))

        let first = try await cache.model(for: key)
        let second = try await cache.model(for: key)

        #expect(first == 1)
        #expect(second == 1)
        #expect(loader.callCount == 1)
        #expect(await cache.isReady(key))
    }

    @Test func concurrentRequestsShareOneLoad() async throws {
        let loader = ScriptedLoader(block: true)
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.shapE(
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle"),
            bpeVocabularyURL: URL(fileURLWithPath: "/tmp/mlx3d/bpe.txt")
        )

        let first = Task { try await cache.model(for: key) }
        await loader.waitUntilCalls(atLeast: 1)
        let second = Task { try await cache.model(for: key) }
        try await waitForJoins(cache, count: 1)

        loader.releaseOne()
        let firstModel = try await first.value
        let secondModel = try await second.value

        #expect(firstModel == 1)
        #expect(secondModel == 1)
        #expect(loader.callCount == 1)
        #expect(await cache.isReady(key))
    }

    @Test func failedLoadIsNotCachedAndTheNextRequestRetries() async throws {
        let loader = ScriptedLoader(block: false)
        loader.failNext()
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))

        await #expect(throws: MLX3DRuntimeError.weightsNotFound("simulated")) {
            try await cache.model(for: key)
        }
        #expect(await cache.isReady(key) == false)

        let retried = try await cache.model(for: key)
        #expect(retried == 1)
        #expect(loader.callCount == 2)
        #expect(await cache.isReady(key))
    }

    @Test func coalescedFailureIsNotCached() async throws {
        let loader = ScriptedLoader(block: true)
        loader.failNext()
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))

        let first = Task { try await cache.model(for: key) }
        await loader.waitUntilCalls(atLeast: 1)
        let second = Task { try await cache.model(for: key) }
        try await waitForJoins(cache, count: 1)
        loader.releaseOne()

        await #expect(throws: MLX3DRuntimeError.weightsNotFound("simulated")) {
            try await first.value
        }
        await #expect(throws: MLX3DRuntimeError.weightsNotFound("simulated")) {
            try await second.value
        }
        #expect(await cache.isReady(key) == false)
        #expect(loader.callCount == 1)

        loader.disableBlocking()
        let retried = try await cache.model(for: key)
        #expect(retried == 1)
        #expect(loader.callCount == 2)
    }

    @Test func unloadDiscardsAnInFlightLoadAndWaitersSeeModelUnloaded() async throws {
        let loader = ScriptedLoader(block: true, value: 9)
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))
        let unloaded = MLX3DRuntimeError.modelUnloaded(backend: .tripoSR, bundlePath: key.bundleURL.path)

        let owner = Task { try await cache.model(for: key) }
        await loader.waitUntilCalls(atLeast: 1)
        let joiner = Task { try await cache.model(for: key) }
        try await waitForJoins(cache, count: 1)

        await cache.unload(key)
        loader.releaseOne()

        await #expect(throws: unloaded) {
            try await owner.value
        }
        await #expect(throws: unloaded) {
            try await joiner.value
        }
        #expect(await cache.isReady(key) == false)

        loader.disableBlocking()
        let reloaded = try await cache.model(for: key)
        #expect(reloaded == 9)
        #expect(loader.callCount == 2)
        #expect(await cache.isReady(key))
    }

    @Test func inFlightLoadCannotReenterTheCacheAfterUnload() async throws {
        let loader = ScriptedLoader(block: true)
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.shapE(
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle"),
            bpeVocabularyURL: URL(fileURLWithPath: "/tmp/mlx3d/bpe.txt")
        )
        let unloaded = MLX3DRuntimeError.modelUnloaded(backend: .shapE, bundlePath: key.bundleURL.path)

        let stale = Task { try await cache.model(for: key) }
        await loader.waitUntilCalls(atLeast: 1)
        await cache.unload(key)

        let replacement = Task { try await cache.model(for: key) }
        await loader.waitUntilCalls(atLeast: 2)

        loader.releaseOne()
        await #expect(throws: unloaded) {
            try await stale.value
        }
        #expect(await cache.isReady(key) == false)

        loader.releaseOne()
        let model = try await replacement.value
        #expect(model == 1)
        #expect(await cache.isReady(key))
        #expect(loader.callCount == 2)
    }

    @Test func evictDropsAReadyModelAndLeavesAnInFlightLoadAlone() async throws {
        let loader = ScriptedLoader(block: false)
        let cache = MLX3DModelCache<Int> { _ in try await loader.load() }
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))

        _ = try await cache.model(for: key)
        await cache.evict(key)
        #expect(await cache.isReady(key) == false)
        _ = try await cache.model(for: key)
        #expect(loader.callCount == 2)

        let blocking = ScriptedLoader(block: true)
        let inFlight = MLX3DModelCache<Int> { _ in try await blocking.load() }
        let pending = Task { try await inFlight.model(for: key) }
        await blocking.waitUntilCalls(atLeast: 1)
        await inFlight.evict(key)
        blocking.releaseOne()
        #expect(try await pending.value == 1)
        #expect(await inFlight.isReady(key))
        _ = try await inFlight.model(for: key)
        #expect(blocking.callCount == 1)
    }

    @Test func distinctKeysDoNotShareALoad() async throws {
        let loader = ScriptedLoader(block: false)
        let cache = MLX3DModelCache<Int> { key in
            try await loader.load()
            return key.backend == .shapE ? 3 : 4
        }
        let tripo = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors"))
        let otherFile = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/other.safetensors"))
        let shape = MLX3DModelKey.shapE(
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle"),
            bpeVocabularyURL: URL(fileURLWithPath: "/tmp/mlx3d/a.txt.gz")
        )
        let otherBPE = MLX3DModelKey.shapE(
            modelBundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle"),
            bpeVocabularyURL: URL(fileURLWithPath: "/tmp/mlx3d/b.txt.gz")
        )

        #expect(try await cache.model(for: tripo) == 4)
        #expect(try await cache.model(for: otherFile) == 4)
        #expect(try await cache.model(for: shape) == 3)
        #expect(try await cache.model(for: otherBPE) == 3)
        #expect(loader.callCount == 4)
        #expect(shape != otherBPE)
        #expect(tripo != otherFile)
    }

    @Test func canonicalPathsShareOneKey() {
        let dotted = MLX3DModelKey.tripoSR(
            weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/models/../weights.safetensors")
        )
        let plain = MLX3DModelKey.tripoSR(
            weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/weights.safetensors")
        )
        let slashed = MLX3DModelKey(
            backend: .shapE,
            bundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle/")
        )
        let directory = MLX3DModelKey(
            backend: .shapE,
            bundleURL: URL(fileURLWithPath: "/tmp/mlx3d/bundle")
        )

        #expect(dotted == plain)
        #expect(slashed == directory)
        #expect(dotted.bundleURL.path == "/tmp/mlx3d/weights.safetensors")
    }

    @Test func shapEKeyIncludesTheBPEFile() {
        let bundle = URL(fileURLWithPath: "/tmp/mlx3d/bundle")
        let explicit = MLX3DModelKey.shapE(
            modelBundleURL: bundle,
            bpeVocabularyURL: URL(fileURLWithPath: "/tmp/mlx3d/custom-bpe.txt")
        )
        let `default` = MLX3DModelKey.shapE(modelBundleURL: bundle)
        #expect(explicit.auxiliaryURLs.map(\.path) == ["/tmp/mlx3d/custom-bpe.txt"])
        #expect(
            `default`.auxiliaryURLs.map(\.path)
                == [bundle.appendingPathComponent("bpe_simple_vocab_16e6.txt.gz").standardizedFileURL.path]
        )
        #expect(explicit != `default`)
    }
}

struct SessionLoadTests {
    @Test func missingShapEBundleIsNotCachedAsSuccess() async {
        let session = MLX3DSession()
        let key = MLX3DModelKey.shapE(
            modelBundleURL: URL(fileURLWithPath: "/nonexistent/mlx3d-session-bundle"),
            bpeVocabularyURL: URL(fileURLWithPath: "/nonexistent/mlx3d-bpe.txt")
        )
        let missing = MLX3DRuntimeError.bundleNotFound(key.bundleURL.path)
        await #expect(throws: missing) {
            try await session.load(key)
        }
        await #expect(throws: missing) {
            try await session.load(key)
        }
    }

    @Test func missingTripoSRWeightsAreNotCachedAsSuccess() async {
        let session = MLX3DSession()
        let key = MLX3DModelKey.tripoSR(
            weightsURL: URL(fileURLWithPath: "/nonexistent/mlx3d-session-weights.safetensors")
        )
        let missing = MLX3DRuntimeError.weightsNotFound(key.bundleURL.path)
        await #expect(throws: missing) {
            try await session.load(key)
        }
        await #expect(throws: missing) {
            try await session.load(key)
        }
    }

    @Test func unloadOfAMissingKeyDoesNotThrow() async {
        let session = MLX3DSession()
        let key = MLX3DModelKey.tripoSR(weightsURL: URL(fileURLWithPath: "/tmp/mlx3d/never-loaded.safetensors"))
        await session.unload(key)
        await session.unloadAll()
        await session.evict(key)
        await session.evictAll()
    }
}

struct SessionImageCacheTests {
    init() {
        Device.setDefault(device: Device(.cpu))
    }

    @Test func shapEImageThrowsWhenTheBundleIsMissing() async {
        let session = MLX3DSession()
        let bundle = URL(fileURLWithPath: "/nonexistent/shap-e-bundle")
        let key = MLX3DModelKey.shapE(modelBundleURL: bundle)
        await #expect(throws: MLX3DRuntimeError.bundleNotFound(key.bundleURL.path)) {
            try await session.imageToMeshShapE(
                cgImage: solidCGImage(width: 4, height: 4),
                modelBundleURL: bundle,
                options: ShapEImageToMeshOptions(gridSize: 8, applyWhiteBackgroundRemoval: false)
            )
        }
    }

    @Test func tripoCutoutStrictNilDoesNotLoadAndFallbackDoes() async throws {
        let image = solidCGImage(width: 4, height: 4)
        let weights = URL(fileURLWithPath: "/tmp/mlx3d/session-weights.safetensors")
        let strict = testSession()
        await #expect(
            throws: MLX3DRuntimeError.imageUnsupported("Vision foreground cutout produced no subject.")
        ) {
            try await strict.tripoCGImage(
                image,
                weightsURL: weights,
                options: ImageToMeshOptions(gridSize: 8, cutoutFailurePolicy: .strict),
                orientation: .up
            ) { _, _, _ in
                nil
            }
        }
        #expect(await strict.loadsStarted(for: .tripoSR) == 0)

        let thrown = testSession()
        await #expect(throws: SessionPrepFailed.self) {
            try await thrown.tripoCGImage(
                image,
                weightsURL: weights,
                options: ImageToMeshOptions(gridSize: 8, cutoutFailurePolicy: .strict),
                orientation: .up
            ) { _, _, _ in
                throw SessionPrepFailed()
            }
        }
        #expect(await thrown.loadsStarted(for: .tripoSR) == 0)

        let fallback = testSession()
        let mesh = try await fallback.tripoCGImage(
            image,
            weightsURL: weights,
            options: ImageToMeshOptions(gridSize: 8, cutoutFailurePolicy: .fallbackToOriginal),
            orientation: .up
        ) { _, _, _ in
            nil
        }
        #expect(mesh.faceCount == 1)
        #expect(await fallback.loadsStarted(for: .tripoSR) == 1)
    }

    @Test func secondShapEImageCallReusesTheSessionSlot() async throws {
        let session = testSession()
        let image = solidCGImage(width: 4, height: 4)
        let bundle = URL(fileURLWithPath: "/tmp/mlx3d/session-bundle")
        let options = ShapEImageToMeshOptions(gridSize: 8, applyWhiteBackgroundRemoval: false)
        let first = try await session.imageToMeshShapE(
            cgImage: image,
            modelBundleURL: bundle,
            options: options
        )
        let second = try await session.imageToMeshShapE(
            cgImage: image,
            modelBundleURL: bundle,
            options: options
        )
        #expect(first.count == 1)
        #expect(second == first)
        #expect(await session.loadsStarted(for: .shapE) == 1)
    }

    @Test func secondTripoImageCallReusesTheSessionSlot() async throws {
        let session = testSession()
        let image = solidCGImage(width: 4, height: 4)
        let weights = URL(fileURLWithPath: "/tmp/mlx3d/session-weights.safetensors")
        let options = ImageToMeshOptions(gridSize: 8, applyForegroundCutout: false)
        let first = try await session.imageToMesh(cgImage: image, weightsURL: weights, options: options)
        let second = try await session.imageToMesh(cgImage: image, weightsURL: weights, options: options)
        #expect(first.faceCount == 1)
        #expect(second == first)
        #expect(await session.loadsStarted(for: .tripoSR) == 1)
    }
}

private func testSession() -> MLX3DSession {
    MLX3DSession(
        shapELoader: { _ in FakeSessionShapE() },
        tripoLoader: { _ in FakeSessionTripo() }
    )
}

private struct FakeSessionShapE: SessionShapE {
    func textMesh(
        prompt: String,
        options: TextToMeshOptions,
        batchSize: Int,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        Array(repeating: sessionTriangle, count: max(batchSize, 1))
    }

    func imageMeshes(
        image: CGImage,
        options: ShapEImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) async throws -> [Mesh] {
        [sessionTriangle]
    }
}

private struct FakeSessionTripo: SessionTripoSR {
    func mesh(
        image: MLXArray,
        options: ImageToMeshOptions,
        progress: MLX3DProgressHandler?
    ) throws -> Mesh {
        sessionTriangle
    }
}

private let sessionTriangle = Mesh(
    positions: [0, 0, 0, 1, 0, 0, 0, 1, 0],
    faces: [0, 1, 2]
)

private struct SessionPrepFailed: Error {}

private func solidCGImage(width: Int, height: Int) -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
        CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    )
    let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
    )!
    context.setFillColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()!
}

private func waitForJoins(_ cache: MLX3DModelCache<Int>, count: Int) async throws {
    for _ in 0..<10_000 {
        if await cache.coalescedJoinCount >= count {
            return
        }
        await Task.yield()
    }
    Issue.record("Timed out waiting for \(count) coalesced load(s).")
    throw CancellationError()
}

private final class ScriptedLoader: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private var parked = 0
    private var blocked: [CheckedContinuation<Void, Never>] = []
    private var waiters: [(target: Int, continuation: CheckedContinuation<Void, Never>)] = []
    private var block: Bool
    private let value: Int
    private var pendingFailure = false

    init(block: Bool, value: Int = 1) {
        self.block = block
        self.value = value
    }

    var callCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }

    func failNext() {
        lock.lock()
        pendingFailure = true
        lock.unlock()
    }

    func disableBlocking() {
        lock.lock()
        block = false
        lock.unlock()
    }

    func load() async throws -> Int {
        let (call, shouldFail, shouldBlock) = lock.withLock { () -> (Int, Bool, Bool) in
            calls += 1
            let call = calls
            let shouldFail = pendingFailure
            pendingFailure = false
            return (call, shouldFail, block)
        }
        if shouldBlock {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let ready: [CheckedContinuation<Void, Never>] = lock.withLock {
                    blocked.append(continuation)
                    parked = call
                    let ready = waiters.filter { parked >= $0.target }.map(\.continuation)
                    waiters.removeAll { parked >= $0.target }
                    return ready
                }
                for waiter in ready {
                    waiter.resume()
                }
            }
        }
        if shouldFail {
            throw MLX3DRuntimeError.weightsNotFound("simulated")
        }
        return value
    }

    func waitUntilCalls(atLeast target: Int) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let already: Bool = lock.withLock {
                if parked >= target {
                    return true
                }
                waiters.append((target, continuation))
                return false
            }
            if already {
                continuation.resume()
            }
        }
    }

    func releaseOne() {
        let continuation: CheckedContinuation<Void, Never>? = lock.withLock {
            guard !blocked.isEmpty else { return nil }
            return blocked.removeFirst()
        }
        continuation?.resume()
    }
}
