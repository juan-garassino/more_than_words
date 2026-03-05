import Foundation

struct Constants {
    // Dimensions
    static let maxStatValue: Double = 1.0
    static let minStatValue: Double = 0.0
    
    // Decay Rates (per tick)
    static let tickInterval: TimeInterval = 15.0
    
    struct Decay {
        static let hunger: Double = 0.02
        static let dirt: Double = 0.01
        static let tiredness: Double = 0.03
        static let happiness: Double = 0.01 // decays if not played with
        static let loneliness: Double = 0.01 // increases if not interacted
    }
    
    // Thresholds
    static let sicknessHungerThreshold: Double = 0.8
    static let sicknessDirtThreshold: Double = 0.7
    static let criticalHealthThreshold: Double = 0.2
    
    // Model Config
    static let historyLength: Int = 128
    static let embeddingDim: Int = 64
    
    // Token Mappings (Simplified for now - would match model vocab)
    struct Tokens {
        static let PAD: Int64 = 0
        static let BOS: Int64 = 1
        static let EOS: Int64 = 2
        static let SPECIAL: Int64 = 3
        
        // Actions
        static let ACTION_Start: Int64 = 100 // Placeholder base
        // World
        static let WORLD_Start: Int64 = 200 // Placeholder base
    }
}
