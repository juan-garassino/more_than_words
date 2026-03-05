import SwiftUI

struct ContentView: View {
    @StateObject private var gameState: GameState

    init() {
        let bundleURL = Bundle.main.bundleURL
        let cartridgeURL = bundleURL.appendingPathComponent("AmberCipher.cartridge")
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
                ConvergenceMeterView(dimensions: gameState.casebook.convergenceDimensions)
                CasebookView(casebook: gameState.casebook)
                OracleTrayView(triads: gameState.currentTriads)
                HandView(hand: gameState.playerHand)
            }
            .padding()
        }
    }
}
