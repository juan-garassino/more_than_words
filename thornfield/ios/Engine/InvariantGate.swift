import Foundation

struct InvariantGate {
    static func isOpen(convergenceScore: Float) -> Bool {
        convergenceScore >= 0.75
    }
}
