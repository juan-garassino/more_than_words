import Foundation

struct ConvergenceTracker {
    static func score(dimensions: [Float]) -> Float {
        dimensions.min() ?? 0.0
    }
}
