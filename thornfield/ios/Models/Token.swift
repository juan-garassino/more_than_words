import Foundation

enum TokenClass: String, Codable, CaseIterable {
    case suspect = "SUSPECT"
    case location = "LOCATION"
    case object = "OBJECT"
    case emotion = "EMOTION"
    case modifier = "MODIFIER"
    case action = "ACTION"
    case time = "TIME"
    case motive = "MOTIVE"
    case witness = "WITNESS"
    case event = "EVENT"
    case need = "NEED"
    case state = "STATE"
    case offering = "OFFERING"
    case unknown = "UNKNOWN"
}

enum TokenPhase: String, Codable, CaseIterable {
    case early = "EARLY"
    case mid = "MID"
    case late = "LATE"
    case invariant = "INVARIANT"
    case any = "ANY"

    func isAvailable(at turn: Int) -> Bool {
        switch self {
        case .any:
            return true
        case .invariant:
            return false
        case .early:
            return turn <= 8
        case .mid:
            return turn >= 5 && turn <= 14
        case .late:
            return turn >= 10
        }
    }
}

struct Token: Identifiable, Codable, Hashable {
    let id: String
    let tokenClass: TokenClass
    let phase: TokenPhase
    let attractorWeights: [Float]
    let affinityTags: [String]
    let repulsionTags: [String]
    let temperature: Float
    let narrativeGradient: Float
    let isInvariant: Bool
    let surfaceExpression: String
}

struct UnknownToken {
    let id: String = "unknown:\(UUID().uuidString)"
    let playerDescription: String

    func asToken(nDims: Int) -> Token {
        Token(
            id: id,
            tokenClass: .unknown,
            phase: .any,
            attractorWeights: Array(repeating: 0.0, count: nDims),
            affinityTags: [],
            repulsionTags: [],
            temperature: 1.0,
            narrativeGradient: 0.0,
            isInvariant: false,
            surfaceExpression: playerDescription
        )
    }
}
