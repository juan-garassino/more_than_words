import XCTest
@testable import Thornfield

final class ConvergenceProofTests: XCTestCase {
    func testConvergenceScoreUsesMin() {
        let state = CasebookState(convergenceDimensions: [0.2, 0.8, 0.5])
        XCTAssertEqual(state.convergenceScore, 0.2)
    }
}
