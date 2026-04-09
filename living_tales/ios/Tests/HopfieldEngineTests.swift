import XCTest
@testable import LivingTales

final class HopfieldEngineTests: XCTestCase {
    func testAffinityLookupSymmetry() {
        let table: AffinityTable = [TokenPairKey("a", "b"): 0.5]
        XCTAssertEqual(table[TokenPairKey("b", "a")], 0.5)
    }
}
