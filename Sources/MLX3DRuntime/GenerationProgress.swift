/// One phase or step from Shap-E or TripoSR.
///
/// `phase` is the underlying runtime's name (`embedding`, `sampling`, `volume`,
/// and the rest). `step` and `totalSteps` are that runtime's counters.
/// `partialMeshes` is non-nil only on Shap-E sampling ticks when
/// `progressInterval` is greater than zero. A nil progress handler is forwarded
/// as a nil Shap-E callback, so those extracts do not run. Labels, percentages,
/// and preview drawing stay with the caller.
public struct MLX3DGenerationEvent: Sendable, Equatable {
    public var phase: String
    public var step: Int
    public var totalSteps: Int
    public var partialMeshes: [Mesh]?

    public init(
        phase: String,
        step: Int,
        totalSteps: Int,
        partialMeshes: [Mesh]? = nil
    ) {
        self.phase = phase
        self.step = step
        self.totalSteps = totalSteps
        self.partialMeshes = partialMeshes
    }
}

public typealias MLX3DProgressHandler = @Sendable (MLX3DGenerationEvent) -> Void

enum GenerationProgress {
    static func emit(
        _ progress: MLX3DProgressHandler?,
        phase: String,
        step: Int,
        totalSteps: Int
    ) {
        guard let progress else { return }
        progress(
            MLX3DGenerationEvent(
                phase: phase,
                step: step,
                totalSteps: totalSteps,
                partialMeshes: nil
            )
        )
    }
}
