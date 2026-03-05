import Foundation

struct CasebookState {
    var placedTriads: [GridPosition: Triad] = [:]
    var convergenceDimensions: [Float]
    var turnCount: Int = 0
    var grid: [[Triad?]]

    init(convergenceDimensions: [Float], rows: Int = 8, cols: Int = 6) {
        self.convergenceDimensions = convergenceDimensions
        self.grid = Array(repeating: Array(repeating: nil, count: cols), count: rows)
    }

    var convergenceScore: Float {
        convergenceDimensions.min() ?? 0.0
    }

    var invariantGateOpen: Bool {
        convergenceScore >= 0.75
    }

    var allPlacedTokens: [Token] {
        placedTriads.values.flatMap(\.tokens)
    }

    var activeAffinityTags: Set<String> {
        Set(allPlacedTokens.flatMap(\.affinityTags))
    }

    var placedTokenIds: Set<String> {
        Set(allPlacedTokens.map(\.id))
    }

    func spatiallyAdjacentTokens(to position: GridPosition, radius: Int = 2) -> [Token] {
        placedTriads.compactMap { gridPos, triad in
            let distance = max(abs(gridPos.row - position.row), abs(gridPos.col - position.col))
            return distance <= radius ? triad : nil
        }.flatMap(\.tokens)
    }

    mutating func placeTriad(_ triad: Triad, at position: GridPosition) {
        var t = triad
        t.suggestedPosition = position
        placedTriads[position] = t
        turnCount += 1

        let contribution = triad.attractorContribution
        for i in convergenceDimensions.indices {
            convergenceDimensions[i] = min(1.0, convergenceDimensions[i] + contribution[i] * 0.25)
        }
    }
}
