import SwiftUI

struct OracleTrayView: View {
    let triads: [Triad]
    var onSelect: (Triad) -> Void = { _ in }

    var body: some View {
        HStack(spacing: 12) {
            ForEach(triads, id: \.id) { triad in
                TriadView(triad: triad)
                    .onTapGesture { onSelect(triad) }
            }
        }
        .frame(maxWidth: .infinity)
    }
}
