import SwiftUI

struct OracleTrayView: View {
    let triads: [Triad]

    var body: some View {
        HStack(spacing: 12) {
            ForEach(triads, id: \.id) { triad in
                TriadView(triad: triad)
            }
        }
        .frame(maxWidth: .infinity)
    }
}
