import Foundation

struct PetState: Codable {
    // Stats (0.0 - 1.0)
    var hunger: Double = 0.5
    var dirt: Double = 0.0
    var tiredness: Double = 0.0
    var happiness: Double = 0.8
    var health: Double = 1.0
    var loneliness: Double = 0.0
    
    // Derived state
    var isSick: Bool {
        return health < Constants.criticalHealthThreshold
    }
    
    var isAsleep: Bool = false
    
    // Decay Logic
    mutating func decay() {
        if isAsleep {
            tiredness = max(0, tiredness - 0.05) // Recover while sleeping
            hunger = min(1, hunger + Constants.Decay.hunger * 0.5) // Hunger slower when asleep
        } else {
            hunger = min(1, hunger + Constants.Decay.hunger)
            dirt = min(1, dirt + Constants.Decay.dirt)
            tiredness = min(1, tiredness + Constants.Decay.tiredness)
            loneliness = min(1, loneliness + Constants.Decay.loneliness)
            
            // Random happiness decay
            if Double.random(in: 0...1) > 0.8 {
                happiness = max(0, happiness - Constants.Decay.happiness)
            }
        }
        
        // Sickness Check
        if hunger > Constants.sicknessHungerThreshold || dirt > Constants.sicknessDirtThreshold {
            health = max(0, health - 0.01)
        } else {
            // Natural healing if well fed and clean
            if hunger < 0.2 && dirt < 0.2 {
                health = min(1, health + 0.005)
            }
        }
    }
    
    mutating func interact(action: String) {
        // Basic interaction logic before Model inference refinement
        switch action {
        case "FEED":
            hunger = max(0, hunger - 0.3)
            health = min(1, health + 0.05)
        case "CLEAN":
            dirt = max(0, dirt - 0.5)
            happiness = min(1, happiness + 0.1)
        case "PLAY":
            happiness = min(1, happiness + 0.25)
            tiredness = min(1, tiredness + 0.15)
            loneliness = max(0, loneliness - 0.2)
        case "SLEEP":
            isAsleep = true
        case "WAKE":
            isAsleep = false
        case "MEDICINE":
            health = min(1, health + 0.4)
        default:
            break
        }
    }
}
