import SwiftUI

struct ContentView: View {
    @StateObject private var gameState: GameState

    init() {
        let bundleURL = Bundle.main.bundleURL
        let cartridgeURL = bundleURL.appendingPathComponent("TheAmberCipher.cartridge")
        let loader = CartridgeLoader()
        if let spec = try? loader.load(from: cartridgeURL) {
            _gameState = StateObject(wrappedValue: GameState(cartridge: spec))
        } else {
            let emptySpec = CartridgeSpec(
                type: .mystery,
                caseId: "empty",
                title: "Empty",
                nAttractorDims: 3,
                convergenceThreshold: 0.75,
                convergenceRate: 0.25,
                minTurns: 10,
                maxTurns: 18,
                tokens: [],
                affinityTable: [:],
                invariantTokens: [],
                openingTokenIds: []
            )
            _gameState = StateObject(wrappedValue: GameState(cartridge: emptySpec))
        }
    }

    var body: some View {
        ZStack {
            Color(red: 0.05, green: 0.04, blue: 0.03).ignoresSafeArea()
            VStack(spacing: 12) {
                Text(phaseLabel)
                    .font(.caption)
                    .foregroundColor(Color(red: 0.6, green: 0.5, blue: 0.4))
                ConvergenceMeterView(dimensions: gameState.casebook.convergenceDimensions)
                CasebookView(casebook: gameState.casebook)
                OracleTrayView(triads: gameState.currentTriads, onSelect: { triad in
                    gameState.placeTriad(triad, at: triad.suggestedPosition)
                })
                HandView(hand: gameState.playerHand)
            }
            .padding()

            if gameState.phase == .denouement {
                Color.black.opacity(0.85).ignoresSafeArea()
                VStack(spacing: 20) {
                    DenouementView(tokens: gameState.cartridge.invariantTokens)
                    Button("Play Again") {
                        gameState.restart()
                    }
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
                    .background(Color(red: 0.4, green: 0.25, blue: 0.1))
                    .cornerRadius(8)
                }
            }
        }
    }

    private var phaseLabel: String {
        switch gameState.phase {
        case .opening: return "opening"
        case .exploration: return "exploration"
        case .convergenceZone: return "convergence zone"
        case .invariantOpen: return "invariant open"
        case .denouement: return "denouement"
        case .closed: return "closed"
        }
    }
}
