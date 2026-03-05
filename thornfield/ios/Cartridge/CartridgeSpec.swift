import Foundation

enum CartridgeType: String, Codable {
    case mystery = "MYSTERY"
    case tamagotchi = "TAMAGOTCHI"
}

struct CartridgeSpec {
    let type: CartridgeType
    let caseId: String
    let title: String
    let nAttractorDims: Int
    let convergenceThreshold: Float
    let convergenceRate: Float
    let minTurns: Int
    let maxTurns: Int
    let tokens: [Token]
    let affinityTable: AffinityTable
    let invariantTokens: [Token]
    let openingTokenIds: [String]

    func tokenById(_ id: String) -> Token? {
        tokens.first { $0.id == id }
    }
}
