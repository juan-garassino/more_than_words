import SwiftUI
import Combine

@MainActor
class GameViewModel: ObservableObject {
    @Published var pet: PetState
    @Published var world: WorldState
    @Published var history: GameHistory
    @Published var aiSuggestion: String? = nil
    
    private var timer: Timer?
    private let modelHandler = ModelHandler() // We will create this next
    
    init() {
        // Load from persistence or create new
        // For now, simple init
        self.pet = PetState()
        self.world = WorldState()
        self.history = GameHistory()
        
        startTimer()
    }
    
    func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: Constants.tickInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.tick()
            }
        }
    }
    
    func tick() {
        pet.decay()
        world.updateLambda(basedOn: pet)
        
        // Optional: Run passive inference every tick or few ticks
        // runInference()
    }
    
    func performAction(_ action: String) {
        // 1. Immediate Feedback
        pet.interact(action: action)
        
        // 2. Update History
        // TODO: Map action string to Token ID
        // history.append(token: actionToken, typeId: ACTION_TYPE)
        
        // 3. Run Inference for next state / reaction
        runInference()
    }
    
    enum Direction {
        case north, south, east, west
    }
    
    func move(_ direction: Direction) {
        // Update location logic
        // For now just simulation
        switch direction {
        case .north: world.currentLocation = "LOC_NORTH"
        case .south: world.currentLocation = "LOC_SOUTH"
        case .east: world.currentLocation = "LOC_EAST"
        case .west: world.currentLocation = "LOC_WEST"
        }
        runInference()
    }
    
    private func runInference() {
        Task {
            if let prediction = await modelHandler.predict(history: history) {
                // Update State based on prediction
                await MainActor.run {
                    self.world.lambda = prediction.lambda
                    self.world.visibleTokens = prediction.visibleTokens
                    
                    if let action = prediction.suggestedAction {
                        self.aiSuggestion = "AI recommends: \(action)"
                    } else {
                        self.aiSuggestion = nil
                    }
                }
            } else {
                // Fallback if model fails or is missing (Simple rules)
                await MainActor.run {
                    if pet.hunger > 0.7 {
                         aiSuggestion = "AI recommends: FEED"
                    } else if pet.dirt > 0.6 {
                        aiSuggestion = "AI recommends: CLEAN"
                    } else {
                        aiSuggestion = nil
                    }
                }
            }
        }
    }
    
    func saveState() {
        // Persist to UserDefaults or CoreData
    }
}
