import Foundation

struct WorldState: Codable {
    // Lambda (Ambiguity) - key mechanic
    var lambda: Double = 0.2 // 0.0 = clear, 1.0 = total chaos/fog
    
    var currentLocation: String = "LOC_HOME"
    
    // Visible World Tokens
    // These are strings representing what the agent "sees" or what triggers context
    var visibleTokens: [String] = ["HOME_BASE"]
    
    mutating func updateLambda(basedOn pet: PetState) {
        // High stress/needs -> High Lambda
        let stressFactor = (pet.hunger + pet.dirt + pet.loneliness) / 3.0
        let baseLambda = 0.1
        
        // If sick, lambda spikes
        let sicknessPenalty = pet.isSick ? 0.3 : 0.0
        
        lambda = min(0.9, baseLambda + (stressFactor * 0.5) + sicknessPenalty)
    }
}
