import Foundation

/// In-memory cache of loaded models for one ``MLX3DSession``.
///
/// Callers do not use this type. ``MLX3DSession`` owns one cache per backend.
/// Tests inject a loader so the four cache rules can run without weights:
///
/// - A second request for a ready key returns the cached model and does not
///   call the loader again.
/// - Concurrent requests for a key that is still loading share that one task.
/// - A thrown load is removed. The next request calls the loader again.
/// - ``unload(_:)`` bumps a per-key generation. A load that was already in
///   flight cannot write the model back. Every waiter of that load throws
///   ``MLX3DRuntimeError/modelUnloaded(backend:bundlePath:)``.
///
/// ``evict(_:)`` drops a model that is already ready. It does not bump the
/// generation, so a load that is still in flight can still be cached when it
/// finishes. ``unload(_:)`` is the call that cancels that publication.
actor MLX3DModelCache<Model: Sendable> {
    typealias Loader = @Sendable (MLX3DModelKey) async throws -> Model

    private enum SlotState {
        case loading(Task<Model, Error>)
        case ready(Model)
    }

    private struct Slot {
        var generation: UInt64
        var state: SlotState
    }

    private let loader: Loader
    private var slots: [MLX3DModelKey: Slot] = [:]
    /// Last generation published by ``unload(_:)``. A load records the value
    /// it observed at start. A later unload increments it, and the load
    /// refuses to store its model.
    private var generations: [MLX3DModelKey: UInt64] = [:]
    private var coalescedJoins = 0
    /// New loader tasks started. A cache hit or a joined in-flight load does not increment this.
    private(set) var loadsStarted = 0

    init(loader: @escaping Loader) {
        self.loader = loader
    }

    func model(for key: MLX3DModelKey) async throws -> Model {
        if let slot = slots[key] {
            switch slot.state {
            case .ready(let model):
                return model
            case .loading(let task):
                coalescedJoins += 1
                return try await join(key: key, generation: slot.generation, task: task)
            }
        }

        let generation = generations[key] ?? 0
        loadsStarted += 1
        let load = loader
        let task = Task.detached {
            try await load(key)
        }
        slots[key] = Slot(generation: generation, state: .loading(task))
        return try await finishOwning(key: key, generation: generation, task: task)
    }

    /// Drops a ready model. An in-flight load is left in place and may still
    /// enter the cache when it succeeds.
    func evict(_ key: MLX3DModelKey) {
        guard let slot = slots[key] else { return }
        if case .ready = slot.state {
            slots[key] = nil
        }
    }

    func evictAll() {
        let keys = Array(slots.keys)
        for key in keys {
            evict(key)
        }
    }

    /// Drops the key and invalidates an in-flight load for it.
    ///
    /// The loader task is cancelled. Cancellation is cooperative: if the
    /// loader returns a model anyway, that model is not stored, and every
    /// waiter receives ``MLX3DRuntimeError/modelUnloaded(backend:bundlePath:)``
    /// rather than the loaded value. A load that starts after this call uses
    /// the new generation and may cache normally.
    func unload(_ key: MLX3DModelKey) {
        generations[key] = (generations[key] ?? 0) &+ 1
        guard let slot = slots.removeValue(forKey: key) else { return }
        if case .loading(let task) = slot.state {
            task.cancel()
        }
    }

    func unloadAll() {
        let keys = Set(slots.keys).union(generations.keys)
        for key in keys {
            unload(key)
        }
    }

    func isReady(_ key: MLX3DModelKey) -> Bool {
        if case .ready = slots[key]?.state {
            return true
        }
        return false
    }

    var coalescedJoinCount: Int {
        coalescedJoins
    }

    private func join(
        key: MLX3DModelKey,
        generation: UInt64,
        task: Task<Model, Error>
    ) async throws -> Model {
        do {
            let model = try await task.value
            guard generationIsCurrent(key, generation) else {
                throw Self.unloaded(key)
            }
            return model
        } catch {
            guard generationIsCurrent(key, generation) else {
                throw Self.unloaded(key)
            }
            throw error
        }
    }

    private func finishOwning(
        key: MLX3DModelKey,
        generation: UInt64,
        task: Task<Model, Error>
    ) async throws -> Model {
        do {
            let model = try await task.value
            guard generationIsCurrent(key, generation),
                let slot = slots[key],
                slot.generation == generation
            else {
                throw Self.unloaded(key)
            }
            slots[key] = Slot(generation: generation, state: .ready(model))
            return model
        } catch {
            guard generationIsCurrent(key, generation) else {
                throw Self.unloaded(key)
            }
            if let slot = slots[key], slot.generation == generation {
                slots[key] = nil
            }
            throw error
        }
    }

    private func generationIsCurrent(_ key: MLX3DModelKey, _ generation: UInt64) -> Bool {
        (generations[key] ?? 0) == generation
    }

    private static func unloaded(_ key: MLX3DModelKey) -> MLX3DRuntimeError {
        .modelUnloaded(backend: key.backend, bundlePath: key.bundleURL.path)
    }
}
