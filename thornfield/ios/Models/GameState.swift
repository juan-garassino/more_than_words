import Foundation
import SwiftUI

enum GamePhase: Equatable {
    case opening
    case exploration
    case convergenceZone
    case invariantOpen
    case denouement
    case closed
}

@MainActor
final class GameState: ObservableObject {
    @Published var phase: GamePhase = .opening
    @Published var casebook: CasebookState
    @Published var currentTriads: [Triad] = []
    @Published var playerHand: [UnknownToken]
    @Published var selectedTriad: Triad?

    private let engine: AffinityEngine
    private let cartridge: CartridgeSpec

    init(cartridge: CartridgeSpec) {
        self.cartridge = cartridge
        self.casebook = CasebookState(
            convergenceDimensions: Array(repeating: 0.0, count: cartridge.nAttractorDims)
        )
        let index = TriadIndex()
        index.build(from: cartridge.tokens, affinityTable: cartridge.affinityTable)
        self.engine = AffinityEngine(affinityTable: cartridge.affinityTable, triadIndex: index)
        self.playerHand = (0..<5).map { _ in UnknownToken(playerDescription: "Unknown") }

        placeOpeningTriad()
    }

    func beginTurn() {
        if casebook.invariantGateOpen {
            currentTriads = [buildInvariantTriad()]
            phase = .invariantOpen
        } else {
            currentTriads = engine.surfaceTriads(count: 3, casebook: casebook)
            updatePhase()
        }
    }

    func placeTriad(_ triad: Triad, at position: GridPosition) {
        casebook.placeTriad(triad, at: position)

        if triad.tokens.allSatisfy(\.isInvariant) {
            phase = .denouement
        } else {
            beginTurn()
        }
    }

    func playUnknown(_ unknown: UnknownToken, replacing replacedToken: Token, in triad: Triad) -> Triad {
        var tokens = triad.tokens
        if let idx = tokens.firstIndex(where: { $0.id == replacedToken.id }) {
            tokens[idx] = unknown.asToken(nDims: cartridge.nAttractorDims)
        }
        return Triad(
            tokens: tokens,
            jointEnergy: triad.jointEnergy,
            convergenceDelta: triad.convergenceDelta,
            suggestedPosition: triad.suggestedPosition
        )
    }

    private func updatePhase() {
        let score = casebook.convergenceScore
        if score >= 0.75 { phase = .invariantOpen }
        else if score >= 0.60 { phase = .convergenceZone }
        else { phase = .exploration }
    }

    private func buildInvariantTriad() -> Triad {
        Triad(
            tokens: cartridge.invariantTokens,
            jointEnergy: 0.0,
            convergenceDelta: Array(repeating: 1.0, count: cartridge.nAttractorDims),
            suggestedPosition: GridPosition(row: 7, col: 2)
        )
    }

    private func placeOpeningTriad() {
        let opening = cartridge.openingTokenIds.compactMap { cartridge.tokenById($0) }
        if opening.count == 3 {
            let triad = Triad(tokens: opening)
            casebook.placeTriad(triad, at: GridPosition(row: 0, col: 0))
        }
        beginTurn()
    }
}
