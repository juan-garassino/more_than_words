import XCTest
@testable import Thornfield

final class PathValidationTests: XCTestCase {
    func testTriadHasThreeTokens() {
        let token = Token(
            id: "t",
            tokenClass: .object,
            phase: .any,
            attractorWeights: [0.1, 0.1, 0.1],
            affinityTags: [],
            repulsionTags: [],
            temperature: 0.5,
            narrativeGradient: 0.0,
            isInvariant: false,
            surfaceExpression: "t"
        )
        let triad = Triad(tokens: [token, token, token])
        XCTAssertEqual(triad.tokens.count, 3)
    }
}
