import Foundation

struct GameHistory: Codable {
    // Rolling buffer of token IDs
    private(set) var tokens: [Int64]
    private(set) var typeIds: [Int64]
    
    init() {
        // Initialize with PAD tokens
        tokens = Array(repeating: Constants.Tokens.PAD, count: Constants.historyLength)
        typeIds = Array(repeating: Constants.Tokens.PAD, count: Constants.historyLength)
    }
    
    mutating func append(token: Int64, typeId: Int64) {
        // Remove first
        tokens.removeFirst()
        typeIds.removeFirst()
        
        // Append new
        tokens.append(token)
        typeIds.append(typeId)
    }
    
    // Helper to get array for Core ML input (MultiArray conversion will happen in ModelHandler)
    func getCurrentContext() -> ([Int64], [Int64]) {
        return (tokens, typeIds)
    }
}
