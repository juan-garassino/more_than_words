import Foundation

struct GridPosition: Hashable, Codable {
    let row: Int
    let col: Int
}

struct Triad: Identifiable, Hashable {
    let id: UUID
    let tokens: [Token]
    var jointEnergy: Float
    var convergenceDelta: [Float]
    var suggestedPosition: GridPosition

    init(tokens: [Token], jointEnergy: Float = 0.0, convergenceDelta: [Float] = [], suggestedPosition: GridPosition = GridPosition(row: 0, col: 0)) {
        self.id = UUID()
        self.tokens = tokens
        self.jointEnergy = jointEnergy
        self.convergenceDelta = convergenceDelta
        self.suggestedPosition = suggestedPosition
    }

    var isClassDiverse: Bool {
        Set(tokens.map(\.tokenClass)).count == 3
    }

    var attractorContribution: [Float] {
        let n = Float(tokens.count)
        guard let dims = tokens.first?.attractorWeights.count else { return [] }
        return (0..<dims).map { dim in
            tokens.reduce(0) { $0 + $1.attractorWeights[dim] } / n
        }
    }
}
