import Foundation

struct TokenPairKey: Hashable {
    let a: String
    let b: String

    init(_ x: String, _ y: String) {
        if x <= y {
            self.a = x
            self.b = y
        } else {
            self.a = y
            self.b = x
        }
    }
}

typealias AffinityTable = [TokenPairKey: Float]
